from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.downstream.perturbation.condition import PAIR_IDS
from mignet_ce.downstream.perturbation.grn_io import GRNGraph, read_grn, write_grn
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


@pytest.mark.parametrize("pair_id", PAIR_IDS)
def test_all_pairs_use_exact_same_default_mirror_formula(pair_id: str) -> None:
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


@pytest.mark.parametrize("weight", ["bad", "", "NaN", "inf", "-inf"])
def test_fixed_grn_read_rejects_invalid_weights(tmp_path, weight):
    nodes = tmp_path / "grn_nodes.json"
    path = tmp_path / "edges.csv"
    nodes.write_text('["b", "a", "isolated"]')
    path.write_text(f"regulator,target,weight\na,b,{weight}\n")
    with pytest.raises(ValueError, match="weights"):
        read_grn(path, nodes_path=nodes)


def test_fixed_grn_round_trip_preserves_order_and_floats(tmp_path):
    nodes_path = tmp_path / "grn_nodes.json"
    graph = GRNGraph(("B", "A", "C", "D"), np.zeros((4, 4)))
    graph.adjacency[1, 0] = np.nextafter(0.1, 1.)
    path = write_grn(graph, tmp_path / "edges.csv", nodes_path=nodes_path)
    actual = read_grn(path, nodes_path=nodes_path)
    assert actual.nodes == graph.nodes
    np.testing.assert_array_equal(actual.adjacency, graph.adjacency)
    path.write_text("regulator,target,weight\nA,B,1\nA,B,0\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_grn(path, nodes_path=nodes_path)
    path.write_text("regulator,target,weight\nA,unknown,0\n")
    with pytest.raises(ValueError, match="unknown genes"):
        read_grn(path, nodes_path=nodes_path)
