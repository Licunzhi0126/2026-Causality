from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare._shared.distances import robust_normalize_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    N_CORRECTION_WEIGHT,
    CompareNGKlGRNAnchorV5PijMethod,
    build_grnanchored_kl_cost,
)
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_source = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    n_target = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    g_source = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]])
    g_target = np.array([[12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 12.0, 0.0]])
    return n_source, n_target, g_source, g_target


def test_grnanchored_cost_matches_frozen_formula_and_is_not_unit_bounded() -> None:
    n_source, n_target, g_source, g_target = _features()
    cost, metadata = build_grnanchored_kl_cost(n_source, n_target, g_source, g_target)

    n_cost = pairwise_feature_kl(n_source, n_target, beta=FIXED_FEATURE_BETA)
    g_cost = pairwise_feature_kl(g_source, g_target, beta=FIXED_FEATURE_BETA)
    normalized_n, _ = robust_normalize_cost(n_cost, copy=True)
    np.testing.assert_allclose(
        cost,
        g_cost + N_CORRECTION_WEIGHT * normalized_n,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(cost.max()) > 1.0
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0.0)
    assert metadata["final_cost_clipped_to_unit_interval"] is False
    assert metadata["removes_unit_interval_gibbs_ei_bound"] is True


def test_grnanchor_v5_is_registered_without_rebinding_frozen_baseline() -> None:
    method = get_pij_method("compare_NG_kl_grnanchor_v5")
    assert isinstance(method, CompareNGKlGRNAnchorV5PijMethod)
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod


def test_grnanchor_v5_config_requires_light_cci_grn() -> None:
    TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method="compare_NG_kl_grnanchor_v5",
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    ).validate()
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(
            network_method="light_cci",
            pij_method="compare_NG_kl_grnanchor_v5",
        ).validate()


def test_grnanchor_v5_rejects_version_parameter_drift() -> None:
    method = CompareNGKlGRNAnchorV5PijMethod()
    n_source, n_target, g_source, g_target = _features()
    with pytest.raises(ValueError, match="fixes pij_entropy_epsilon"):
        method.build_kl_cost(
            n_source,
            n_target,
            beta=0.5,
            weight_n=0.5,
            weight_g=0.5,
            grn_source=g_source,
            grn_target=g_target,
        )
    with pytest.raises(ValueError, match="fixes pij_temperature"):
        method._build_pair_kernel(
            source=n_source,
            target=n_target,
            cfg=TemporalRunConfig(pij_temperature=0.5),
            grn_source=g_source,
            grn_target=g_target,
        )
