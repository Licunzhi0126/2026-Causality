from __future__ import annotations

import numpy as np

from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.topology_metrics import compute_topology_metrics


def test_hub_degree_gini_exceeds_balanced_graph() -> None:
    hub = GRNGraph(tuple("abcd"), np.array([[0., 1., 1., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]))
    balanced = GRNGraph(tuple("abcd"), np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.], [1., 0., 0., 0.]]))
    assert compute_topology_metrics(hub).degree_gini > compute_topology_metrics(balanced).degree_gini


def test_reciprocity_detects_bidirectional_edges() -> None:
    graph = GRNGraph(("a", "b"), np.array([[0., 1.], [1., 0.]]))
    assert compute_topology_metrics(graph).reciprocity == 1.0
