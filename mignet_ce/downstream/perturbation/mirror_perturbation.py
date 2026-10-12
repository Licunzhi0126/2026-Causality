"""Application and provenance for exact mirrored edge-weight perturbations."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .grn_io import GRNGraph


@dataclass(frozen=True)
class MirrorOperation:
    selected_edges: tuple[tuple[int, int], ...]
    baseline_weights: tuple[float, ...]
    positive_weights: tuple[float, ...]
    negative_weights: tuple[float, ...]
    mirror_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_mirrored_weight_perturbation(graph: GRNGraph, *, selected_edges: tuple[tuple[int, int], ...], mirror_fraction: float) -> tuple[GRNGraph, GRNGraph, MirrorOperation]:
    if not 0.0 < mirror_fraction <= 1.0:
        raise ValueError("mirror_fraction must be strictly between zero and one.")
    positive, negative = graph.adjacency.copy(), graph.adjacency.copy()
    baseline_weights: list[float] = []
    for i, j in selected_edges:
        if i == j or i < 0 or j < 0 or i >= len(graph.nodes) or j >= len(graph.nodes):
            raise ValueError("selected edge is outside the graph or is a self-loop.")
        weight = float(graph.adjacency[i, j])
        if weight <= 0.0:
            raise ValueError("selected edge must exist in the baseline graph.")
        baseline_weights.append(weight)
        delta = mirror_fraction * weight
        positive[i, j] = weight + delta
        negative[i, j] = weight - delta
    negative[np.abs(negative) < np.finfo(float).eps] = 0.0
    operation = MirrorOperation(
        selected_edges=selected_edges,
        baseline_weights=tuple(baseline_weights),
        positive_weights=tuple(float(positive[i, j]) for i, j in selected_edges),
        negative_weights=tuple(float(negative[i, j]) for i, j in selected_edges),
        mirror_fraction=mirror_fraction,
    )
    return GRNGraph(graph.nodes, positive), GRNGraph(graph.nodes, negative), operation
