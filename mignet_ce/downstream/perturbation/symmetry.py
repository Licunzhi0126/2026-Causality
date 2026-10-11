"""Quality control for direct mirrored weight perturbations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .grn_io import GRNGraph
from .selection import PerturbationSelection


@dataclass(frozen=True)
class SymmetryQC:
    same_node_selection: bool
    same_edge_selection: bool
    mirror_error_max: float
    untouched_error_max: float
    nonnegative_weights: bool
    finite_weights: bool
    no_self_loops: bool
    selected_node_count: int
    region_node_count: int
    selected_edge_count: int
    region_edge_count: int
    node_region_coverage: float
    edge_region_coverage: float
    overall_pass: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_mirror_pair(*, baseline: GRNGraph, positive: GRNGraph, negative: GRNGraph, selection: PerturbationSelection, mirror_tolerance: float) -> SymmetryQC:
    selected = np.zeros_like(baseline.adjacency, dtype=bool)
    for i, j in selection.selected_edges:
        selected[i, j] = True
    mirror_error = np.abs(positive.adjacency[selected] + negative.adjacency[selected] - 2.0 * baseline.adjacency[selected])
    untouched = ~selected
    untouched_error = np.concatenate((np.abs(positive.adjacency[untouched] - baseline.adjacency[untouched]), np.abs(negative.adjacency[untouched] - baseline.adjacency[untouched])))
    same_nodes = baseline.nodes == positive.nodes == negative.nodes
    same_edges = all(positive.adjacency[i, j] != baseline.adjacency[i, j] and negative.adjacency[i, j] != baseline.adjacency[i, j] for i, j in selection.selected_edges)
    finite = bool(np.isfinite(positive.adjacency).all() and np.isfinite(negative.adjacency).all())
    nonnegative = bool((positive.adjacency >= 0.0).all() and (negative.adjacency >= 0.0).all())
    no_self_loops = bool(np.all(np.diag(positive.adjacency) == 0.0) and np.all(np.diag(negative.adjacency) == 0.0))
    mirror_error_max = float(mirror_error.max()) if mirror_error.size else 0.0
    untouched_error_max = float(untouched_error.max()) if untouched_error.size else 0.0
    node_coverage = len(selection.selected_nodes) / max(len(selection.region_nodes), 1)
    edge_coverage = len(selection.selected_edges) / max(len(selection.region_edges), 1)
    overall = same_nodes and same_edges and finite and nonnegative and no_self_loops and mirror_error_max <= mirror_tolerance and untouched_error_max <= mirror_tolerance
    return SymmetryQC(same_nodes, same_edges, mirror_error_max, untouched_error_max, nonnegative, finite, no_self_loops, len(selection.selected_nodes), len(selection.region_nodes), len(selection.selected_edges), len(selection.region_edges), node_coverage, edge_coverage, overall)


