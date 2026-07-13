from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare import cost_fusion
from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase, _build_fused_pre_cost
from mignet_ce.pij.compare.distances import (
    pairwise_scalar_absolute_distance,
    pairwise_vector_distance,
    robust_normalize_cost,
)
from mignet_ce.pij.compare.features import CompareFeatureSet
from mignet_ce.pij.compare.sparse_ot import _topk_candidates


def _normalized(cost: np.ndarray) -> np.ndarray:
    return robust_normalize_cost(cost.copy())[0]


def test_fused_cost_matches_independent_normalization_and_weighting() -> None:
    components = {
        "L": (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 1.0]])),
        "E": (np.array([[0.0], [2.0]]), np.array([[1.0], [4.0]])),
        "Sr": (np.array([[1.0], [5.0]]), np.array([[2.0], [8.0]])),
    }
    weights = {"L": 1.0, "E": 2.0, "Sr": 3.0}
    fused, metadata = _build_fused_pre_cost(components, "euclidean", weights)
    expected = (
        _normalized(pairwise_vector_distance(*components["L"], "euclidean"))
        + 2.0 * _normalized(pairwise_vector_distance(*components["E"], "euclidean"))
        + 3.0 * _normalized(pairwise_scalar_absolute_distance(*components["Sr"]))
    ) / 6.0
    np.testing.assert_allclose(fused, expected)
    assert metadata["component_normalization"] == "robust_5_95_before_fusion"
    assert metadata["candidate_cost_rescaling"] == "existing_candidate_minmax"
    assert metadata["components"]["Sr"]["distance_rule"] == "scalar_absolute_difference"


def test_zero_weight_components_degenerate_to_l_cost() -> None:
    components = {
        "L": (np.array([[0.0], [2.0]]), np.array([[1.0], [4.0]])),
        "E": (np.array([[10.0], [20.0]]), np.array([[30.0], [40.0]])),
        "Sr": (np.array([[7.0], [9.0]]), np.array([[1.0], [5.0]])),
    }
    l_only, _ = _build_fused_pre_cost({"L": components["L"]}, "euclidean", {"L": 1.0})
    fused, _ = _build_fused_pre_cost(
        components,
        "euclidean",
        {"L": 1.0, "E": 0.0, "Sr": 0.0},
    )
    np.testing.assert_allclose(fused, l_only)


def test_final_candidate_edges_come_from_fused_cost() -> None:
    components = {
        "L": (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ),
        "Sr": (np.array([[0.0], [10.0]]), np.array([[10.0], [0.0], [5.0]])),
    }
    l_cost, _ = _build_fused_pre_cost({"L": components["L"]}, "cosine", {"L": 1.0})
    fused, _ = _build_fused_pre_cost(components, "cosine", {"L": 1.0, "Sr": 1.0})
    l_edges = set(zip(*_topk_candidates(l_cost, source_k=1, target_k=1)))
    fused_edges = set(zip(*_topk_candidates(fused, source_k=1, target_k=1)))
    assert l_edges != fused_edges
    assert int(np.argmin(l_cost[0])) != int(np.argmin(fused[0]))


class _DummyPair:
    lower_layer = "spot"
    upper_layer = "domain"

    def label(self) -> str:
        return "spot__domain"


def _context():
    return SimpleNamespace(
        organ="heart",
        pair=_DummyPair(),
        time_points=["11.5", "12.5"],
        feature_alignment_space="native_units",
        lower_coords_by_time=[np.zeros((2, 2)), np.zeros((2, 2))],
        upper_coords_by_time=[np.zeros((2, 2)), np.zeros((2, 2))],
    )


def _feature_set(key: str, *, mismatched: bool = False) -> CompareFeatureSet:
    if key == "L":
        values = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 0.0]])]
    else:
        first = np.array([[0.0], [2.0], [4.0]]) if mismatched else np.array([[0.0], [2.0]])
        values = [first, np.array([[1.0], [4.0]])]
    return CompareFeatureSet(
        lower_features=values,
        upper_features=[value.copy() for value in values],
        feature_names=[f"{key}:1"],
        metadata={"key": key},
    )


class _TestLEMethod(CompareCostFusionSotBase):
    name = "test_L_E_costmix"
    component_keys = ("L", "E")
    vector_metric = "cosine"


def test_method_builds_each_component_once_without_multikey_concat_and_uses_costmix_for_p(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def fake_build(context, cfg, keys, *, apply_feature_weights=True):
        calls.append((tuple(keys), apply_feature_weights))
        return _feature_set(keys[0])

    captured: list[np.ndarray] = []
    original_ot = cost_fusion.run_sparse_semi_relaxed_ot_from_cost

    def capture_ot(dense_cost, **kwargs):
        captured.append(np.asarray(dense_cost).copy())
        return original_ot(dense_cost, **kwargs)

    monkeypatch.setattr(cost_fusion, "build_compare_feature_set", fake_build)
    monkeypatch.setattr(cost_fusion, "run_sparse_semi_relaxed_ot_from_cost", capture_ot)
    cfg = TemporalRunConfig(
        network_method="light_cci",
        pij_method="compare_L_E_costmix_cos_sot",
        ot_dist_k=1,
        ot_sim_k=1,
    )
    result, kernels = _TestLEMethod().run(_context(), cfg, [(0, 1)])
    assert calls == [(("L",), False), (("E",), False)]
    assert len(captured) == 2
    expected, _ = _build_fused_pre_cost(
        {
            "L": (_feature_set("L").lower_features[0], _feature_set("L").lower_features[1]),
            "E": (_feature_set("E").lower_features[0], _feature_set("E").lower_features[1]),
        },
        "cosine",
        {"L": 1.0, "E": 1.0},
    )
    np.testing.assert_allclose(captured[0], expected)
    assert result.method_metadata["method_result_features_used_for_P"] is False
    assert kernels.kernel_metadata["fusion_mode"] == "cost_mix"


def test_component_row_mismatch_has_context_rich_error(monkeypatch) -> None:
    def fake_build(context, cfg, keys, *, apply_feature_weights=True):
        return _feature_set(keys[0], mismatched=keys[0] == "E")

    monkeypatch.setattr(cost_fusion, "build_compare_feature_set", fake_build)
    cfg = TemporalRunConfig(network_method="light_cci", pij_method="compare_L_E_costmix_cos_sot")
    with pytest.raises(ValueError, match=r"test_L_E_costmix.*organ=heart.*time_pair=11.5->12.5.*side=lower"):
        _TestLEMethod().run(_context(), cfg, [(0, 1)])
