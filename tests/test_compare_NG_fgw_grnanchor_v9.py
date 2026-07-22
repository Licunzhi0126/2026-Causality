from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_fgw_grnanchor_v9 import (
    FIXED_FEATURE_BETA,
    CompareNGFGWGRNAnchorV9PijMethod,
)
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import build_grnanchored_kl_cost
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_source = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    n_target = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    g_source = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]])
    g_target = np.array([[12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 12.0, 0.0]])
    return n_source, n_target, g_source, g_target


def test_v9_node_cost_is_exactly_the_frozen_v5_cost() -> None:
    n_source, n_target, g_source, g_target = _features()
    expected, _ = build_grnanchored_kl_cost(n_source, n_target, g_source, g_target)
    actual, metadata = CompareNGFGWGRNAnchorV9PijMethod().build_kl_cost(
        n_source,
        n_target,
        beta=FIXED_FEATURE_BETA,
        weight_n=0.5,
        weight_g=0.5,
        grn_source=g_source,
        grn_target=g_target,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert metadata is not None
    assert metadata["node_cost_is_exact_frozen_v5_formula"] is True


def test_fgw_v9_is_registered_without_rebinding_frozen_baseline() -> None:
    method = get_pij_method("compare_NG_fgw_grnanchor_v9")
    assert isinstance(method, CompareNGFGWGRNAnchorV9PijMethod)
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod


def test_fgw_v9_config_requires_light_cci_grn() -> None:
    TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method="compare_NG_fgw_grnanchor_v9",
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    ).validate()
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(
            network_method="light_cci",
            pij_method="compare_NG_fgw_grnanchor_v9",
        ).validate()


def test_fgw_v9_builds_balanced_kernel_without_leakage_metadata() -> None:
    method = CompareNGFGWGRNAnchorV9PijMethod()
    n_source, n_target, g_source, g_target = _features()
    source_adjacency = sp.csr_matrix(np.array([[0.0, 1.0], [0.5, 0.0]]))
    target_adjacency = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.2],
                [0.3, 0.0, 1.0],
                [0.8, 0.4, 0.0],
            ]
        )
    )
    cfg = TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method=method.name,
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    )

    joint_sparse, pij_sparse, pij, diagnostics = method._build_pair_kernel(
        source=n_source,
        target=n_target,
        cfg=cfg,
        source_adjacency=source_adjacency,
        target_adjacency=target_adjacency,
        grn_source=g_source,
        grn_target=g_target,
    )

    joint = joint_sparse.toarray()
    np.testing.assert_allclose(joint.sum(axis=1), np.full(2, 0.5), atol=1.0e-9)
    np.testing.assert_allclose(joint.sum(axis=0), np.full(3, 1.0 / 3.0), atol=1.0e-9)
    np.testing.assert_allclose(pij_sparse.toarray(), pij, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pij.sum(axis=1), np.ones(2), atol=1.0e-12)
    assert diagnostics["fgw"]["outer_iterations"] == 10
    assert diagnostics["fgw"]["uses_ei_for_fitting"] is False
    assert diagnostics["fgw"]["uses_layer_identity"] is False
    assert diagnostics["fgw"]["uses_third_timepoint"] is False


def test_fgw_v9_rejects_parameter_drift() -> None:
    method = CompareNGFGWGRNAnchorV9PijMethod()
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
            source_adjacency=sp.eye(2, format="csr"),
            target_adjacency=sp.eye(3, format="csr"),
            grn_source=g_source,
            grn_target=g_target,
        )
