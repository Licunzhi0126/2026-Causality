"""Run one Optimal CG training for the explicit Plan 1 source-to-target pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..common.manifest import load_source_target
from ..optimal_cg.maturity_bridge import resolve_maturity
from ..optimal_cg.reporting import collect_summary, write_json
from ..optimal_cg.trainer_bridge import run_training
from ..optimal_cg.validation import validate_run
from ..paths import DEFAULT_OUTPUT_ROOT


def _inputs(plan1_root: Path, source: dict[str, object], target: dict[str, object]) -> dict[str, Path]:
    def paths(row: dict[str, object]) -> dict[str, Path]:
        sample = str(row["sample_id"])
        return {"h5ad": Path(str(row["prepared_h5ad"])), "cci": plan1_root / "cci" / f"{sample}_CCI_total.npz", "index": plan1_root / "cci" / f"{sample}_index.tsv", "grn": plan1_root / "grn" / sample / "grn_edges.csv"}
    left, right = paths(source), paths(target)
    return {f"{key}_t": value for key, value in left.items()} | {f"{key}_tp": value for key, value in right.items()}


def _read_index(path: Path) -> list[str]:
    import pandas as pd
    return pd.read_csv(path, sep="\t").iloc[:, 0].astype(str).tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan1-output-root", "--prepared-root", dest="plan1_output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--maturity-root", type=Path, default=None, help="Optional directory containing source.csv and target.csv maturity tables.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--k", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.k < 1:
        raise ValueError("--k must be positive.")
    source, target = load_source_target(args.plan1_output_root)
    root = args.output_root / "optimal_cg"
    inputs = _inputs(args.plan1_output_root, source, target)
    maturity_t, maturity_tp, maturity_provenance = resolve_maturity(source_sample=str(source["sample_id"]), target_sample=str(target["sample_id"]), h5ad_t=inputs["h5ad_t"], h5ad_tp=inputs["h5ad_tp"], output_root=root, supplied_root=args.maturity_root)
    training_root = root / "training"
    validate_run(inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, spot_ids_t=_read_index(inputs["index_t"]), spot_ids_tp=_read_index(inputs["index_tp"]), k=args.k, out_dir=training_root, device=args.device)
    trainer = run_training(python=args.python, inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, out_dir=training_root, k=args.k, device=args.device)
    summary = collect_summary(training_root)
    causal = {key: summary[key] for key in ("EI_micro_fixed", "EI_macro_best_checkpoint", "delta_EI_best_checkpoint", "best_epoch", "Keff_source", "Keff_target")}
    closure = {key: summary[key] for key in ("I_available", "I_retained", "closure_leakage", "closure_quality", "signal_status") if key in summary}
    provenance = {"k": args.k, "Keff_source": summary.get("Keff_source"), "Keff_target": summary.get("Keff_target"), "source_sample": source["sample_id"], "target_sample": target["sample_id"], "maturity": maturity_provenance, "trainer": trainer}
    write_json(root / "causal_emergence_summary.json", causal)
    write_json(root / "dynamical_closure_summary.json", closure)
    write_json(root / "provenance.json", provenance)
    write_json(root / "summary.json", {"k": args.k, "source_sample": source["sample_id"], "target_sample": target["sample_id"], **causal, **closure})


if __name__ == "__main__":
    main()
