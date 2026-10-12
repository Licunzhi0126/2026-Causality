from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.mirror_perturbation import apply_mirrored_weight_perturbation
from mignet_ce.downstream.perturbation.selection import PerturbationSelection
from mignet_ce.downstream.perturbation.symmetry import evaluate_mirror_pair


def test_mirror_weights_and_alpha_one() -> None:
    graph = GRNGraph(("a", "b"), np.array([[0.0, 0.8], [0.0, 0.0]]))
    positive, negative, operation = apply_mirrored_weight_perturbation(graph, selected_edges=((0, 1),), mirror_fraction=0.5)
    assert np.isclose(positive.adjacency[0, 1], 1.2)
    assert np.isclose(negative.adjacency[0, 1], 0.4)
    assert np.isclose(positive.adjacency[0, 1] + negative.adjacency[0, 1], 1.6)
    positive, negative, _ = apply_mirrored_weight_perturbation(graph, selected_edges=((0, 1),), mirror_fraction=1.0)
    assert np.isclose(positive.adjacency[0, 1], 1.6)
    assert np.isclose(negative.adjacency[0, 1], 0.0)
    assert operation.selected_edges == ((0, 1),)


@pytest.mark.parametrize("pair_id", ("P6_triangle_intensity", "P7_feedforward_loop", "P8_feedback_cycle"))
def test_motif_pairs_use_exact_same_default_mirror_formula(pair_id: str) -> None:
    baseline = GRNGraph(("a", "b", "c"), np.array([[0.0, 2.0, 7.0], [0.0, 0.0, 3.0], [5.0, 0.0, 0.0]]))
    selection = PerturbationSelection(pair_id, 0.5, (0, 1, 2), ((0, 1), (1, 2), (2, 0)), (0, 1, 2), ((0, 1), (1, 2)))
    positive, negative, _ = apply_mirrored_weight_perturbation(baseline, selected_edges=selection.selected_edges, mirror_fraction=1.0)
    selected = np.zeros_like(baseline.adjacency, dtype=bool)
    for edge in selection.selected_edges:
        selected[edge] = True
    assert np.allclose(positive.adjacency[selected], 2.0 * baseline.adjacency[selected])
    assert np.allclose(negative.adjacency[selected], 0.0)
    assert np.allclose(positive.adjacency[~selected], baseline.adjacency[~selected])
    assert np.allclose(negative.adjacency[~selected], baseline.adjacency[~selected])
    assert np.allclose(positive.adjacency - baseline.adjacency, -(negative.adjacency - baseline.adjacency))
    assert evaluate_mirror_pair(baseline=baseline, positive=positive, negative=negative, selection=selection, mirror_tolerance=1e-12).overall_pass
