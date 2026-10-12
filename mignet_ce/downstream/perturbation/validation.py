"""Preflight validation for perturbed graphs and condition artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grn_io import GRNGraph
from .maturity_resolver import MATURITY_ID_COLUMN, MATURITY_VALUE_COLUMN


def validate_maturity(path: Path, spot_ids: list[str], *, id_column: str = MATURITY_ID_COLUMN,
                      maturity_column: str = MATURITY_VALUE_COLUMN) -> None:
    table = pd.read_csv(path)
    if not {id_column, maturity_column}.issubset(table.columns):
        raise ValueError(
            f"Maturity CSV must contain {id_column!r} and {maturity_column!r}: {path}"
        )
    ids = table[id_column].astype(str)
    if ids.duplicated().any() or set(ids) != set(map(str, spot_ids)):
        raise ValueError(f"Maturity IDs must exactly cover the CCI index: {path}")
    if table[maturity_column].isna().any():
        raise ValueError(f"Maturity values must not be missing: {path}")


def validate_training_run(*, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path,
                          spot_ids_t: list[str], spot_ids_tp: list[str], k: int,
                          out_dir: Path, device: str) -> None:
    missing = {name: str(path) for name, path in inputs.items() if not Path(path).is_file()}
    if missing:
        raise FileNotFoundError(f"Perturbation training inputs are missing: {missing}")
    if k <= 1:
        raise ValueError("Optimal CG K must be greater than one.")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device: {device}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Runner output directory must be empty: {out_dir}")
    validate_maturity(maturity_t, spot_ids_t)
    validate_maturity(maturity_tp, spot_ids_tp)


def validate_graph(graph: GRNGraph, baseline: GRNGraph | None = None, *, preserve_degrees: bool = False, preserve_indegree: bool = False) -> None:
    adjacency = graph.adjacency
    if not np.isfinite(adjacency).all():
        raise ValueError("GRN contains non-finite weights.")
    if np.any(np.diag(adjacency) != 0.0):
        raise ValueError("Perturbation created a self-loop.")
    if baseline is None:
        return
    if graph.nodes != baseline.nodes:
        raise ValueError("Perturbation changed the node set.")
    if preserve_indegree and not np.array_equal(np.count_nonzero(graph.adjacency, axis=0), np.count_nonzero(baseline.adjacency, axis=0)):
        raise ValueError("Perturbation violated the in-degree constraint.")
    if preserve_degrees and (not np.array_equal(np.count_nonzero(graph.adjacency, axis=0), np.count_nonzero(baseline.adjacency, axis=0)) or not np.array_equal(np.count_nonzero(graph.adjacency, axis=1), np.count_nonzero(baseline.adjacency, axis=1))):
        raise ValueError("Perturbation violated the degree-preservation constraint.")
