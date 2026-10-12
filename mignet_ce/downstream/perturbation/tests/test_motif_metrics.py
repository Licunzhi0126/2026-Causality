from __future__ import annotations

import numpy as np

from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.motif_index import build_motif_index
from mignet_ce.downstream.perturbation.topology_metrics import compute_motif_metrics


def test_weighted_motifs_have_expected_directional_counts() -> None:
    ffl = GRNGraph(tuple("abc"), np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    cycle = GRNGraph(tuple("abc"), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    ffl_metrics = compute_motif_metrics(ffl, baseline_scale=1.0, index=build_motif_index(ffl))
    cycle_metrics = compute_motif_metrics(cycle, baseline_scale=1.0, index=build_motif_index(cycle))
    assert ffl_metrics["triangle_intensity"] == 1.0
    assert ffl_metrics["feedforward_loop"] == 1.0
    assert ffl_metrics["feedback_cycle"] == 0.0
    assert cycle_metrics["feedback_cycle"] == 1.0
    assert cycle_metrics["feedforward_loop"] == 0.0


def test_motif_values_are_label_invariant_and_weight_homogeneous() -> None:
    graph = GRNGraph(tuple("abc"), np.array([[0.0, 2.0, 3.0], [0.0, 0.0, 4.0], [1.0, 0.0, 0.0]]))
    original = compute_motif_metrics(graph, baseline_scale=1.0, index=build_motif_index(graph))
    scaled = compute_motif_metrics(GRNGraph(graph.nodes, graph.adjacency * 5.0), baseline_scale=1.0, index=build_motif_index(graph))
    assert all(np.isclose(scaled[key], 5.0 * original[key]) for key in ("triangle_intensity", "feedforward_loop", "feedback_cycle"))
