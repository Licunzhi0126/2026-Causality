"""Evaluate one fixed biological-cluster partition for the Plan 1 pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..biological_cluster.analysis import contribution_table, transition_table
from ..biological_cluster.cluster_assignment import biological_assignment
from ..biological_cluster.coarse_graining import evaluate_fixed_partition
from ..biological_cluster.reporting import write_outputs
from ..biological_cluster.spot_dynamics import load_spot_dynamics
from ..common.manifest import load_source_target
from ..common.spot_reference import common_spot_reference_dir
from ..paths import DEFAULT_OUTPUT_ROOT


def _paths(root: Path, row: dict[str, object]) -> dict[str, Path]:
    sample = str(row["sample_id"])
    return {"map": Path(str(row["cluster_map"])), "sample": Path(str(row["prepared_h5ad"])), "cci": root / "cci" / f"{sample}_CCI_total.npz", "index": root / "cci" / f"{sample}_index.tsv", "grn": root / "grn" / sample / "grn_edges.csv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan1-output-root", "--prepared-root", dest="plan1_output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source, target = load_source_target(args.plan1_output_root)
    root = args.output_root / "biological_cluster"
    source_paths, target_paths = _paths(args.plan1_output_root, source), _paths(args.plan1_output_root, target)
    dynamics = load_spot_dynamics(common_spot_reference_dir(args.output_root))
    source_assignment, source_metadata = biological_assignment(source_paths["map"], list(dynamics["source_units"]))
    target_assignment, target_metadata = biological_assignment(target_paths["map"], list(dynamics["target_units"]))
    result = evaluate_fixed_partition(np.asarray(dynamics["pij"]), source_assignment, target_assignment)
    budget = result["budget"]
    closure = {key: value for key, value in budget.items() if key in {"I_available_bits", "I_macro_retained_bits", "closure_leakage_bits", "closure_quality", "information_identity_error", "signal_status", "signal_threshold_bits", "weighting"}}
    causal = {"EI_spot": result["spot_ei"], "EI_cluster": result["cluster_ei"], "delta_EI": result["delta_ei"]}
    provenance = {"source_sample": source["sample_id"], "target_sample": target["sample_id"], "spot_reference": str(common_spot_reference_dir(args.output_root)), "spot_reference_cached": dynamics["cached"]}
    write_outputs(root, pij=np.asarray(result["cluster_pij"]), source_metadata=source_metadata, target_metadata=target_metadata, transitions=transition_table(np.asarray(result["cluster_pij"]), source_metadata, target_metadata), contributions=contribution_table(np.asarray(result["cluster_ei_contributions"]), source_metadata), causal=causal, closure=closure, provenance=provenance)
    (root / "summary.json").write_text(json.dumps({"source_sample": source["sample_id"], "target_sample": target["sample_id"], **causal, **closure}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
