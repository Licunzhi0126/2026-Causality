import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from mignet_ce.downstream.post_perbutation.cci import discover_regions, mirror_condition, topology_metrics, without_loops
from mignet_ce.downstream.post_perbutation.config import Settings, Task


def test_mirror_only_touches_nested_non_loop_edges():
    rng = np.random.default_rng(5)
    matrix = sparse.csr_matrix(rng.uniform(.1, 2, (12, 12)))
    settings = Settings(backbone_fraction=.5)
    regions, _, audit = discover_regions(matrix, settings)
    assert audit["backbone_edges"] == 66
    assert all(len(region) for region in regions.values())
    for topology, region in regions.items():
        positive, small = mirror_condition(matrix, region, Task("cci", "positive", 10, topology), settings)
        negative, matching = mirror_condition(matrix, region, Task("cci", "negative", 10, topology), settings)
        large, wide = mirror_condition(matrix, region, Task("cci", "positive", 90, topology), settings)
        np.testing.assert_allclose((positive + negative).toarray(), 2 * matrix.toarray())
        np.testing.assert_array_equal(negative.diagonal(), matrix.diagonal())
        assert small["selected_edges"] == matching["selected_edges"] <= wide["selected_edges"]
        assert set(zip(*(positive - matrix).nonzero())) <= set(zip(*(large - matrix).nonzero()))


def test_weighted_clustering_matches_directed_networkx_definition():
    matrix = sparse.csr_matrix(np.array([[8, 2, 4, 1], [0, 8, 3, 1], [6, 2, 8, 2], [1, 4, 0, 8.]]))
    clean = without_loops(matrix)
    graph = nx.from_scipy_sparse_array(clean, create_using=nx.DiGraph)
    metrics = topology_metrics(matrix, np.array([0, 0, 1, 1]), 6.)
    assert metrics["clustering"] == pytest.approx(nx.average_clustering(graph, weight="weight"), abs=1e-6)
    assert metrics["modularity"] == pytest.approx(nx.community.modularity(graph, [{0, 1}, {2, 3}], weight="weight"))
    assert metrics["reciprocity"] == nx.reciprocity(graph)
