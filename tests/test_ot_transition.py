from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.mapping import OverlapMapping
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.registry import get_pij_method
from mignet_ce.transition.cost_components import combine_costs, pairwise_feature_cost, pairwise_spatial_cost
from mignet_ce.transition.ot import build_entropic_ot_kernel


def assert_row_stochastic(matrix: np.ndarray) -> None:
    assert matrix.ndim == 2
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_entropic_ot_kernel_fallback_is_row_stochastic() -> None:
    cost = np.array([[0.0, 1.0, 2.0], [2.0, 0.5, 0.0]])
    kernel = build_entropic_ot_kernel(cost, epsilon=0.1, use_pot=False, max_iter=50)
    assert kernel.shape == (2, 3)
    assert_row_stochastic(kernel)


def test_entropic_ot_kernel_unbalanced_fallback_is_row_stochastic() -> None:
    cost = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    kernel = build_entropic_ot_kernel(
        cost,
        epsilon=0.1,
        use_pot=False,
        unbalanced=True,
        mass_reg=1.0,
        max_iter=20,
    )
    assert kernel.shape == (3, 2)
    assert_row_stochastic(kernel)


def test_cost_components_combine_weighted_costs() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0, 0.0], [1.0, 1.0]])
    components = {
        "expression": pairwise_feature_cost(source, target),
        "spatial": pairwise_spatial_cost(np.zeros((2, 2)), np.ones((2, 2))),
    }
    cost, summary = combine_costs(components, {"expression": 1.0, "spatial": 0.2})
    assert cost.shape == (2, 2)
    assert np.all(cost >= 0)
    assert np.all(cost <= 1)
    assert summary["total_weight"] == pytest.approx(1.2)


def _synthetic_context() -> NetworkContext:
    stable_units = ["u1", "u2", "u3"]
    overlaps = [
        OverlapMapping(
            lower_units=["l1", "l2", "l3"],
            upper_units=stable_units,
            counts=np.eye(3),
            weights=np.eye(3),
        ),
        OverlapMapping(
            lower_units=["l1", "l2", "l3"],
            upper_units=stable_units,
            counts=np.eye(3),
            weights=np.eye(3),
        ),
    ]
    lower_mats = [
        np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.2, 0.1, 1.0]]),
        np.array([[0.9, 0.1, 0.3], [0.1, 0.9, 0.2], [0.3, 0.2, 0.9]]),
    ]
    upper_mats = [
        np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.2], [0.3, 0.2, 1.0]]),
        np.array([[0.8, 0.2, 0.4], [0.2, 0.8, 0.3], [0.4, 0.3, 0.8]]),
    ]
    return NetworkContext(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        time_points=["11.5", "12.5"],
        network_method="synthetic",
        stable_upper_units=stable_units,
        shared_genes=["g1", "g2", "g3"],
        lower_mats=lower_mats,
        upper_mats=upper_mats,
        overlaps=overlaps,
        lower_units_by_time=[["l1", "l2", "l3"], ["l1", "l2", "l3"]],
        upper_units_by_time=[stable_units, stable_units],
        upper_coords_by_time=[
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.1, 0.0], [1.1, 0.0], [0.0, 1.1]]),
        ],
        feature_names=["f1", "f2", "f3"],
        feature_blocks={"synthetic": ["f1", "f2", "f3"]},
        graph_summaries=[],
        coverage_tables=[pd.DataFrame(), pd.DataFrame()],
    )


@pytest.mark.parametrize("method_name", ["expr_ot", "energy_entropy_ot"])
def test_ot_pij_methods_build_precomputed_kernels(method_name: str) -> None:
    cfg = TemporalRunConfig(
        organs=["heart"],
        time_points=["11.5", "12.5"],
        pij_method=method_name,
        pij_feature_components=None,
        ot_max_iter=20,
    )
    result, kernels = get_pij_method(method_name).run(_synthetic_context(), cfg, [(0, 1)])
    assert kernels is not None
    assert result.method_metadata["pij_method"] == method_name
    assert kernels.kernel_metadata["pij_method"] == method_name
    assert (0, 1) in kernels.p_lower
    assert (0, 1) in kernels.p_upper
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    assert_row_stochastic(kernels.p_upper[(0, 1)])
