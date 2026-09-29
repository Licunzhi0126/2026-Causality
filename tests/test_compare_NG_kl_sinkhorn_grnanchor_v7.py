from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import build_grnanchored_kl_cost
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import (
    FIXED_FEATURE_BETA,
    SINKHORN_TOLERANCE,
    CompareNGKlSinkhornGRNAnchorV7PijMethod,
    balance_kernel_sinkhorn,
)
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_source = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    n_target = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    g_source = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]])
    g_target = np.array([[12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 12.0, 0.0]])
    return n_source, n_target, g_source, g_target


def test_balanced_sinkhorn_rectangular_kernel_has_uniform_joint_marginals() -> None:
    kernel = np.array(
        [
            [5.0, 1.0, 0.25],
            [0.5, 2.0, 4.0],
        ],
        dtype=float,
    )

    joint, conditional, metadata = balance_kernel_sinkhorn(kernel)

    np.testing.assert_allclose(joint.sum(axis=1), np.full(2, 0.5), rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(joint.sum(axis=0), np.full(3, 1.0 / 3.0), rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(conditional.sum(axis=1), np.ones(2), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(conditional.mean(axis=0), np.full(3, 1.0 / 3.0), rtol=0.0, atol=2.0e-9)
    assert metadata["converged"] is True
    assert metadata["max_absolute_marginal_residual"] <= 2.0e-9
    assert metadata["uses_ei_for_fitting"] is False
    assert metadata["uses_layer_identity"] is False


def test_balanced_sinkhorn_is_invariant_to_positive_row_and_column_scaling() -> None:
    kernel = np.array(
        [
            [3.0, 0.7, 1.4],
            [0.4, 4.0, 0.9],
            [1.1, 0.6, 2.5],
        ],
        dtype=float,
    )
    _, reference, _ = balance_kernel_sinkhorn(kernel)
    rescaled = np.array([0.2, 7.0, 3.0])[:, None] * kernel * np.array([5.0, 0.1, 2.0])[None, :]
    _, actual, _ = balance_kernel_sinkhorn(rescaled)

    np.testing.assert_allclose(actual, reference, rtol=1.0e-8, atol=1.0e-9)


def test_v7_cost_is_exactly_the_frozen_v5_cost() -> None:
    n_source, n_target, g_source, g_target = _features()
    expected, _ = build_grnanchored_kl_cost(n_source, n_target, g_source, g_target)
    actual, metadata = CompareNGKlSinkhornGRNAnchorV7PijMethod().build_kl_cost(
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
    assert metadata["cost_is_exact_frozen_v5_formula"] is True


def test_sinkhorn_v7_is_registered_without_rebinding_frozen_baseline() -> None:
    method = get_pij_method("compare_NG_kl_sinkhorn_grnanchor_v7")
    assert isinstance(method, CompareNGKlSinkhornGRNAnchorV7PijMethod)
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod


def test_sinkhorn_v7_config_requires_light_cci_grn() -> None:
    TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method="compare_NG_kl_sinkhorn_grnanchor_v7",
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    ).validate()
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(
            network_method="light_cci",
            pij_method="compare_NG_kl_sinkhorn_grnanchor_v7",
        ).validate()


def test_sinkhorn_v7_rejects_parameter_drift_and_unsupported_kernel() -> None:
    method = CompareNGKlSinkhornGRNAnchorV7PijMethod()
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
    with pytest.raises(ValueError, match="positive support"):
        balance_kernel_sinkhorn(np.array([[1.0, 0.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        balance_kernel_sinkhorn(
            np.array([[5.0, 1.0, 0.25], [0.5, 2.0, 4.0]]),
            max_iterations=0,
            tolerance=SINKHORN_TOLERANCE,
        )
