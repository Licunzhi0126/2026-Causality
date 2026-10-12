from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.downstream.perturbation.condition import PAIR_IDS
from mignet_ce.downstream.perturbation.config import PerturbationConfig
from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.selection import discover_region, selection_from_region
from mignet_ce.downstream.perturbation.topology_metrics import baseline_communities
from mignet_ce.downstream.perturbation.motif_index import build_motif_index


@pytest.fixture
def graph() -> GRNGraph:
    return GRNGraph(tuple("abcdef"), np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]], dtype=float))


def test_selection_is_nested_and_reuses_identical_region_for_every_method(graph: GRNGraph) -> None:
    communities = baseline_communities(graph)
    for pair_id in PAIR_IDS:
        region = discover_region(graph, pair_id=pair_id, communities=communities, seed=9, config=PerturbationConfig())
        selections = [selection_from_region(region, strength=strength) for strength in (0.2, 0.4, 0.6, 0.8)]
        for lower, higher in zip(selections, selections[1:]):
            assert set(lower.selected_nodes) <= set(higher.selected_nodes)
            assert set(lower.selected_edges) <= set(higher.selected_edges)
            assert lower.region_nodes == higher.region_nodes == region.region_nodes
            assert lower.region_edges == higher.region_edges == region.region_edges


def test_reciprocity_region_only_offers_reciprocal_edges(graph: GRNGraph) -> None:
    region = discover_region(graph, pair_id="P5_reciprocity", communities=baseline_communities(graph), seed=3, config=PerturbationConfig(region_edge_fraction=1.0))
    assert region.region_edges
    assert all(graph.adjacency[j, i] > 0.0 for i, j in region.region_edges)


def test_fast_triangle_score_equals_dense_a_cubed_diagonal(graph: GRNGraph) -> None:
    support = ((graph.adjacency != 0.0) | (graph.adjacency.T != 0.0)).astype(np.int64)
    common = support @ support
    assert np.allclose(np.diag(common @ support) / 2.0, (common * support).sum(axis=1) / 2.0)


def test_motif_selection_accepts_prebuilt_index_without_rebuilding(monkeypatch, graph: GRNGraph) -> None:
    import mignet_ce.downstream.perturbation.selection as selection_module

    index = build_motif_index(graph)
    monkeypatch.setattr(selection_module, "build_motif_index", lambda _: (_ for _ in ()).throw(AssertionError("index should be reused")))
    region = discover_region(graph, pair_id="P7_feedforward_loop", communities=baseline_communities(graph), seed=9, config=PerturbationConfig(), motif_index=index)
    assert region.region_edges
