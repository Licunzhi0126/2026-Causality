"""Run Global Optimal CG and Group Micro EI using preprocessed inputs only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..common.manifest import load_source_target
from ..optimal_cg.maturity_bridge import resolve_maturity
from ..optimal_cg.reporting import collect_summary, write_json
from ..optimal_cg.trainer_bridge import run_training
from ..optimal_cg.group_micro_analysis import run_group_micro_analysis
from ..optimal_cg.validation import validate_run
from ..paths import DEFAULT_OUTPUT_ROOT


def _inputs(prepared_root: Path, source: dict[str, object], target: dict[str, object]) -> dict[str, Path]:
    def paths(row: dict[str, object]) -> dict[str, Path]:
        sample = str(row["sample_id"])
        h5ad = Path(str(row["prepared_h5ad"]))
        if not h5ad.is_absolute():
            h5ad = prepared_root / h5ad
        return {"h5ad": h5ad, "cci": prepared_root / "cci" / f"{sample}_CCI_total.npz", "index": prepared_root / "cci" / f"{sample}_index.tsv", "grn": prepared_root / "grn" / sample / "grn_edges.csv"}
    left, right = paths(source), paths(target)
    return {f"{key}_t": value for key, value in left.items()} | {f"{key}_tp": value for key, value in right.items()}


INPUT_KEYS = ("h5ad", "cci", "index", "grn", "maturity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--maturity-root", type=Path, help="Existing source.csv and target.csv.")
    for key in INPUT_KEYS:
        for stage in ("t", "tp"):
            parser.add_argument(f"--{key}-{stage}", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--alpha-cci", type=float, default=0.20)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--checkpoint-keff-min", type=float, default=None)
    parser.add_argument("--skip-group-micro-analysis", action="store_true")
    return parser


def resolve_inputs(args, parser):
    explicit = {f"{key}_{stage}": getattr(args, f"{key}_{stage}") for key in INPUT_KEYS for stage in ("t", "tp")}
    directory_mode = args.prepared_root is not None or args.maturity_root is not None
    if directory_mode:
        if any(value is not None for value in explicit.values()):
            parser.error("Directory inputs and explicit file paths are mutually exclusive.")
        if args.prepared_root is None or args.maturity_root is None:
            parser.error("Directory mode requires --prepared-root and --maturity-root.")
        source, target = load_source_target(args.prepared_root)
        inputs = _inputs(args.prepared_root, source, target)
        mt, mtp, provenance = resolve_maturity(supplied_root=args.maturity_root)
    else:
        missing = ["--" + key.replace("_", "-") for key, value in explicit.items() if value is None]
        if missing:
            parser.error(f"Explicit mode requires all input files; missing: {', '.join(missing)}")
        mt, mtp = explicit.pop("maturity_t"), explicit.pop("maturity_tp")
        inputs = explicit
        source = {"sample_id": inputs["h5ad_t"].stem}
        target = {"sample_id": inputs["h5ad_tp"].stem}
        provenance = {"source": "user", "paths": [str(mt), str(mtp)]}
    return inputs, mt, mtp, source, target, provenance


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs, maturity_t, maturity_tp, source, target, maturity_provenance = resolve_inputs(args, parser)
    root = args.output_root / "optimal_cg"
    training_root = root / "training"
    validate_run(inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp,
                 k=args.k, out_dir=training_root, device=args.device,
                 protected_roots=[path for path in (args.prepared_root, args.maturity_root) if path is not None])
    trainer = run_training(
        python=args.python, inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp,
        out_dir=training_root, k=args.k, device=args.device, alpha_cci=args.alpha_cci,
        checkpoint_keff_min=args.checkpoint_keff_min,
    )
    summary = collect_summary(training_root)
    causal = {key: summary[key] for key in ("EI_micro_fixed", "EI_macro_best_checkpoint", "delta_EI_best_checkpoint", "best_epoch", "Keff_source", "Keff_target")}
    closure = {key: summary[key] for key in ("I_available", "I_retained", "closure_leakage", "closure_quality", "signal_status") if key in summary}
    provenance = {"k": args.k, "Keff_source": summary.get("Keff_source"), "Keff_target": summary.get("Keff_target"), "source_sample": source["sample_id"], "target_sample": target["sample_id"], "maturity": maturity_provenance, "trainer": trainer, "global_training_status": "completed", "group_analysis_status": "skipped" if args.skip_group_micro_analysis else "pending"}
    write_json(root / "causal_emergence_summary.json", causal)
    write_json(root / "dynamical_closure_summary.json", closure)
    write_json(root / "provenance.json", provenance)
    write_json(root / "summary.json", {"k": args.k, "source_sample": source["sample_id"], "target_sample": target["sample_id"], **causal, **closure})

    # Global results are durable before any optional group analysis begins.
    if not args.skip_group_micro_analysis:
        try:
            result = run_group_micro_analysis(training_root=training_root)
            provenance["group_analysis_status"] = result["status"]
            provenance["group_micro_analysis"] = result
        except Exception as error:
            provenance["group_analysis_status"] = "failed"
            provenance["group_analysis_error"] = f"{type(error).__name__}: {error}"
        write_json(root / "provenance.json", provenance)



if __name__ == "__main__":
    main()
