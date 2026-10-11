"""Condition-specific T1 GRN provenance without source-data mutation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grn_io import GRNGraph, write_grn
from .validation import validate_graph


@dataclass(frozen=True)
class MaterializedCondition:
    inputs: dict[str, Path]
    perturbed_grn: Path


def materialize_source_grn(*, baseline: GRNGraph, perturbed: GRNGraph, baseline_inputs: dict[str, Path], output_dir: Path, pair_id: str) -> MaterializedCondition:
    validate_graph(perturbed, baseline)
    required = {"h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "index_t", "index_tp", "grn_t", "grn_tp"}
    missing = required - set(baseline_inputs)
    if missing:
        raise KeyError(f"Baseline inputs missing keys: {sorted(missing)}")
    topology = Path(output_dir) / "topology"
    grn_path = write_grn(perturbed, topology / "perturbed_grn_edges.csv")
    inputs = {key: Path(value) for key, value in baseline_inputs.items()}
    inputs["grn_t"] = grn_path
    return MaterializedCondition(inputs, grn_path)
