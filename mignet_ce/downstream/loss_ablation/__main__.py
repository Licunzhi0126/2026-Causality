"""Preview table layouts or run the Formal Core two-stage loss ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mignet_ce.downstream.formal_core_analysis.run_pipeline import TIMES

from .variants import BASELINE_METHOD, SCALES, TARGET_K, TIME_PAIRS, VARIANTS, variant_flags


def _run_experiment(args: argparse.Namespace, scales: tuple[str, ...], out_dir: Path) -> None:
    from mignet_ce.io.multiscale_coarse_inputs import InputRoots, prepare_multiscale_inputs, resolve_stage_inputs
    from scripts.run_multiscale_maturity_cci_grn import build_run_specs, run_experiments

    from .plots import draw_tables
    from .tables import checkpoint_rows, table_grid, write_tables

    data_root = args.data_root.resolve()
    roots = InputRoots(
        spot_root=data_root / "spot",
        seurat_k150_root=data_root / "seurat_k150",
        seurat_k40_root=data_root / "seurat_k40",
        cci_root=args.cci_root.resolve(),
        grn_root=args.grn_root.resolve(),
        developmental_root=args.developmental_root.resolve(),
    )
    inputs = {}
    for scale in scales:
        for time in TIMES:
            stage = resolve_stage_inputs(roots, scale=scale, organ="heart", stage=time)
            # Formal Core requires the original CCI index, even where the generic runner can reconstruct it.
            for path in (
                stage.h5ad, stage.cci_total, stage.source_cci_index, stage.grn_edges,
                stage.maturity_features, stage.source_spot_features, stage.spot_domain_map,
            ):
                if path is not None:
                    stat = path.stat()
                    inputs[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    common_flags = [
        "--grn-feature-method", "legacy_nmf", "--cci-feature-method", "laplacian_hks",
        "--alpha-cci", "0.20", "--epochs", str(args.epochs), "--device", args.device,
    ]
    variants = variant_flags()
    k_by_scale = {scale: [TARGET_K[scale]] for scale in scales}
    run_plan = {
        "schema_version": 1, "baseline_method": BASELINE_METHOD, "organ": "heart",
        "time_pairs": [list(pair) for pair in TIME_PAIRS], "k_by_scale": k_by_scale,
        "seed": args.seed, "common_flags": common_flags, "variants": variants, "inputs": inputs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".running"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Output directory is locked by a run: {lock}") from exc
    try:
        config_file = out_dir / "run_config.json"
        if args.resume:
            if not config_file.is_file() or json.loads(config_file.read_text(encoding="utf-8")) != run_plan:
                raise ValueError("Resume requires matching inputs and options in the saved run_config.json.")
        else:
            if any(entry != lock for entry in out_dir.iterdir()):
                raise FileExistsError(f"Use an empty output directory or resume the same run: {out_dir}")
            config_file.write_text(json.dumps(run_plan, indent=2) + "\n", encoding="utf-8")

        prepared = prepare_multiscale_inputs(roots, out_root=out_dir, organ="heart", time_points=TIMES, scales=scales)
        rows = []
        for name, label, _ in VARIANTS:
            print(f"[loss ablation] {label}", flush=True)
            run_root = out_dir / "runs" / name / "optimal_runs"
            specs = build_run_specs(
                prepared, method=BASELINE_METHOD, out_root=run_root,
                scales=scales, pairs=TIME_PAIRS, k_by_scale=k_by_scale, seeds=[args.seed],
            )
            run_experiments(
                specs, out_root=run_root, runner_extra_args=[*common_flags, *variants[name]],
                resume=args.resume, resume_incomplete=args.resume, force=False, continue_on_error=False,
            )
            for spec in specs:
                pair = (spec.source.resolved.stage, spec.target.resolved.stage)
                rows.extend(checkpoint_rows(spec.run_dir, method=label, scale=spec.scale, pair=pair, k=spec.k))

        table_grid(rows, scales)
        write_tables(rows, out_dir)
        draw_tables(rows, out_dir, scales)
    finally:
        lock.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Draw the two layouts with example values.")
    parser.add_argument("--output-root", type=Path, help="Output folder (required for training).")
    parser.add_argument("--scales", nargs="+", choices=SCALES, default=["spot"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cci-root", type=Path)
    parser.add_argument("--grn-root", type=Path)
    parser.add_argument("--developmental-root", type=Path)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Reuse completed jobs from the same run configuration.")
    args = parser.parse_args(argv)
    scales = tuple(scale for scale in SCALES if scale in args.scales)

    if args.preview:
        from .plots import draw_tables, preview_rows

        if args.resume:
            parser.error("--resume is only for formal training runs.")
        out_dir = (args.output_root or Path("output/loss_ablation/preview")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "run_config.json").exists() or (out_dir / ".running").exists():
            parser.error("Choose a separate preview output folder.")
        draw_tables(preview_rows(scales), out_dir, scales, preview=True)
    else:
        for name in ("data_root", "cci_root", "grn_root", "developmental_root", "output_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for training.")
        if args.epochs < 2:
            parser.error("--epochs must be at least 2 for two-stage training.")
        out_dir = args.output_root.resolve()
        _run_experiment(args, scales, out_dir)
    print(f"Loss ablation tables: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
