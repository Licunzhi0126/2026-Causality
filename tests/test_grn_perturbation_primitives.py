from __future__ import annotations

import numpy as np

from mignet_ce.downstream.analysis.grn_perturbation.operators import (
    METHODS,
    apply_operator,
    fixed_targets,
)
from mignet_ce.downstream.analysis.grn_perturbation.propagation import horizontal, path_metrics


def test_all_twelve_operators_have_bounded_topology_effects() -> None:
    graph = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]]
    )
    targets = fixed_targets(graph)
    assert len(METHODS) == 12
    for method in METHODS:
        result, metadata = apply_operator(graph, method, 0.5, 7, targets)
        assert result.shape == graph.shape
        assert np.isfinite(result).all()
        assert metadata["operated_edge_count"] >= 0


def test_horizontal_path_and_metrics_are_well_defined() -> None:
    delta = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    pij = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    assert np.allclose(horizontal(delta, pij), pij.T @ delta)
    assert path_metrics(delta, delta)["path_delta_l1"] == 0.0


def test_strength_zero_is_an_identity_operator() -> None:
    graph = np.asarray([[0.0, 1.0], [2.0, 0.0]])
    result, metadata = apply_operator(
        graph,
        "random_grn_edge_deletion",
        0.0,
        21,
        fixed_targets(graph),
    )
    assert np.array_equal(result, graph)
    assert metadata["operated_edge_count"] == 0
