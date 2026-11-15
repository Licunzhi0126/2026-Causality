"""Train, audit or render domain experiments using the existing formal_two_stage protocol."""

import argparse
import json
from pathlib import Path

from .inputs import load_stage, load_variant_domains
from .variants import TIME_PAIRS, variants_for_experiment, reference_for_experiment


REPO = Path(__file__).resolve().parents[3]


def _next_render_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = parent / f"render_{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def render(args, *, preview: bool = False) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from .metrics import read_run, write_metrics, write_records, history_diagnostics, pareto_report, factorial_effects
    from .plots import render_variants
    from .workflow import SCHEMA_VERSION, FINAL_ARTIFACTS, file_digest, save_json

    root = args.output_root.resolve()
    by_key, rows = {}, []
    manifest = None
    experiment = getattr(args, "experiment", None) or "factorial"
    if not preview:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        plan_path = root / "run_config.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        saved_experiment = plan.get("experiment", "single_factor")
        if getattr(args, "experiment", None) not in (None, saved_experiment):
            raise ValueError("Requested experiment differs from the saved training design")
        experiment = saved_experiment
        variants = variants_for_experiment(experiment)
        if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete"
                or manifest.get("run_config_sha256") != file_digest(plan_path)):
            raise ValueError("Rendering requires a complete, unchanged domain experiment manifest")
        for name, digest in plan["input_sha256"].items():
            if file_digest(Path(name)) != digest:
                raise ValueError(f"Training input changed since the experiment: {name}")
        # Coordinates must come from the same H5AD files as the training inputs.
        for stage in {stage for pair in TIME_PAIRS for stage in pair}:
            source = args.data_root / "spot/heart" / f"spot_heart_{stage}.h5ad"
            original = Path(plan["data_root"]) / "spot/heart" / source.name
            if file_digest(source) != plan["input_sha256"][str(original)]:
                raise ValueError(f"Figure data differ from the training H5AD: {source}")
        for item in manifest["results"]:
            key = (item["variant"], tuple(item["time_pair"]))
            if key in by_key:
                raise ValueError(f"Duplicate result in manifest: {key}")
            run = Path(item["stage2_run"])
            if set(item["artifacts"]) != set(FINAL_ARTIFACTS):
                raise ValueError(f"Incomplete domain result artifact list: {run}")
            for name, digest in item["artifacts"].items():
                if file_digest(run / name) != digest:
                    raise ValueError(f"Changed domain result: {run / name}")
            saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
            for name in ("out_dir", "stage1_only", "stage1_source"):
                saved.pop(name)
            if saved != plan["configurations"][item["variant"]]:
                raise ValueError(f"Experiment configuration mismatch: {run}")
            by_key[key] = run
        expected = {(variant.name, pair) for variant in variants for pair in TIME_PAIRS}
        if by_key.keys() != expected:
            raise ValueError(f"Expected exactly {len(variants)} versions × three time pairs")
        rows = [read_run(by_key[variant.name, pair], variant=variant.name, pair=pair)
                for variant in variants for pair in TIME_PAIRS]
    variants = variants_for_experiment(experiment)
    reference = reference_for_experiment(experiment)
    stages = {stage: load_stage(args.data_root, stage, seurat_scales=())
              for stage in sorted({stage for pair in TIME_PAIRS for stage in pair})}
    out = _next_render_directory(root / ("previews" if preview else "figures"))
    report = {"mode": "preview" if preview else "render", "status": "running", "figures": [],
              "experiment": experiment, "reference": reference}
    report_path = out / "render_manifest.json"
    save_json(report_path, report)
    try:
        if not preview:
            write_metrics(rows, out / "metrics.csv", reference)
            if experiment == "factorial":
                save_json(out / "pareto_candidates.json", pareto_report(rows, reference))
                write_records(factorial_effects(rows), out / "factorial_effects.csv")
                write_records([history_diagnostics(run, variant=name, pair=pair)
                               for (name, pair), run in by_key.items()], out / "penalty_activation.csv")
            report["source_run_manifest_sha256"] = file_digest(root / "run_manifest.json")
        for pair in TIME_PAIRS:
            stage_pair = tuple(stages[stage] for stage in pair)
            domains = None if preview else {
                variant.name: load_variant_domains(by_key[variant.name, pair], stage_pair) for variant in variants
            }
            pair_rows = None if preview else [row for row in rows if (row["source_time"], row["target_time"]) == pair]
            path = out / ("optimal_domains_spot_" + "_to_".join(pair))
            render_variants(stage_pair, domains, pair_rows, path, variants=variants, reference=reference)
            report["figures"].append(str(path))
            save_json(report_path, report)
        report["status"] = "complete"
        save_json(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        save_json(report_path, report)
        raise
    print(out, flush=True)
    return out


def audit_existing(args) -> Path:
    """Read original experiment histories without changing any saved training settings."""
    from .metrics import history_diagnostics, write_records
    from .workflow import file_digest, save_json

    source = args.history_root.resolve()
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_path = source / "run_config.json"
    if manifest.get("status") != "complete" or manifest.get("run_config_sha256") != file_digest(plan_path):
        raise ValueError("History audit requires a complete, unchanged domain experiment")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    variants = variants_for_experiment(plan.get("experiment", "single_factor"))
    keys = [(item["variant"], tuple(item["time_pair"])) for item in manifest["results"]]
    expected = {(item.name, pair) for item in variants for pair in TIME_PAIRS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate history audit results")
    records = []
    for item in manifest["results"]:
        run = Path(item["stage2_run"])
        for name in ("config.json", "metrics.csv", "summary.json"):
            if file_digest(run / name) != item["artifacts"][name]:
                raise ValueError(f"Changed history source: {run / name}")
        records.append(history_diagnostics(run, variant=item["variant"], pair=tuple(item["time_pair"])))
    out = _next_render_directory(args.output_root.resolve() / "history_audits")
    write_records(records, out / "penalty_activation.csv")
    save_json(out / "audit_manifest.json", dict(status="complete", source=str(source),
                                              source_manifest_sha256=file_digest(manifest_path), training_steps=0))
    print(f"[history audit] {out}", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("train", "resume", "render", "preview", "audit"), default="train")
    parser.add_argument("--experiment", choices=("factorial", "single_factor"), default=None,
                        help="New training defaults to factorial; render detects the saved design.")
    parser.add_argument("--history-root", type=Path, help="Completed domain experiment to audit without retraining.")
    parser.add_argument("--data-root", type=Path, default=REPO / "data/mouse_embyro/E1S1_domain_factory")
    parser.add_argument("--cci-root", type=Path)
    parser.add_argument("--grn-root", type=Path)
    parser.add_argument("--developmental-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6,
                        help="At most six training subprocesses; each task continues to Stage 2 when its Stage 1 is ready.")
    args = parser.parse_args(argv)
    if args.mode == "audit":
        if args.history_root is None:
            parser.error("--history-root is required for --mode audit")
        audit_existing(args)
        return 0
    if args.mode in {"train", "resume"}:
        args.experiment = args.experiment or "factorial"
        for name in ("cci_root", "grn_root", "developmental_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for training")
        if args.epochs < 2:
            parser.error("--epochs must be at least 2")
        from .workflow import run

        if args.history_root is not None:
            audit_existing(args)
        run(args)
    render(args, preview=args.mode == "preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
