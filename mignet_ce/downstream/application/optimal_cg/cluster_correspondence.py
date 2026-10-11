"""Macro-cluster correspondence derived from an Optimal CG transition matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MacroCorrespondence:
    source_cluster: int
    target_cluster: int
    transition_probability: float
    source_rank: int = 1
    target_in_degree: int = 0


def row_normalize_macro_pij(pij_macro: np.ndarray) -> np.ndarray:
    """Validate and row-normalize a non-negative macro transition matrix."""
    matrix = np.asarray(pij_macro, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"PIJ_macro must be a non-empty 2D matrix, got {matrix.shape}.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("PIJ_macro must contain finite non-negative values.")
    sums = matrix.sum(axis=1)
    if np.any(sums <= 0.0):
        raise ValueError("Every PIJ_macro row must have positive mass.")
    return matrix / sums[:, None]


def build_macro_correspondence(pij_macro: np.ndarray) -> tuple[np.ndarray, list[MacroCorrespondence]]:
    """Return row-normalized PIJ and one argmax target for every source state."""
    matrix = row_normalize_macro_pij(pij_macro)
    targets = matrix.argmax(axis=1)
    in_degree = np.bincount(targets, minlength=matrix.shape[1])
    records = [
        MacroCorrespondence(
            source_cluster=source,
            target_cluster=int(target),
            transition_probability=float(matrix[source, target]),
            target_in_degree=int(in_degree[target]),
        )
        for source, target in enumerate(targets)
    ]
    return matrix, records


def correspondence_rows(records: list[MacroCorrespondence]) -> list[dict[str, object]]:
    return [
        {
            "source_cluster": item.source_cluster,
            "target_cluster": item.target_cluster,
            "transition_probability": item.transition_probability,
            "source_rank": item.source_rank,
            "target_in_degree": item.target_in_degree,
        }
        for item in records
    ]
