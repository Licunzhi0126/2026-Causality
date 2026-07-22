from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare._shared.quadratic_balanced_ot import (
    solve_state_normalized_quadratic_balanced_ot,
)
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import build_grnanchored_kl_cost
from mignet_ce.pij.compare.compare_NG_kl_sparseot_grnanchor_v8 import (
    FIXED_FEATURE_BETA,
    CompareNGKlSparseOTGRNAnchorV8PijMethod,
)
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_source = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    n_target = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    g_source = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]])
    g_target = np.array([[12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 12.0, 0.0]])
    return n_source, n_target, g_source, g_target


def test_quadratic_balanced_ot_produces_sparse_deterministic_diagonal_plan() -> None:
    cost = np.array([[0.0, 10.0], [10.0, 0.0]])

    joint, conditional, metadata = solve_state_normalized_quadratic_balanced_ot(cost)

    np.testing.assert_allclose(joint, np.diag([0.5, 0.5]), rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(conditional, np.eye(2), rtol=0.0, atol=1.0e-10)
    assert metadata["regularization"] == 2.0
    assert metadata["sparsity"] == pytest.approx(0.5)
    assert metadata["uses_ei_for_fitting"] is False
    assert metadata["uses_layer_identity"] is False


def test_quadratic_balanced_ot_rectangular_plan_has_uniform_marginals() -> None:
    cost = np.array([[0.0, 1.0, 3.0], [2.0, 0.2, 0.0]])

    joint, conditional, metadata = solve_state_normalized_quadratic_balanced_ot(cost)

    np.testing.assert_allclose(joint.sum(axis=1), np.full(2, 0.5), rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(joint.sum(axis=0), np.full(3, 1.0 / 3.0), rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(conditional.sum(axis=1), np.ones(2), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(conditional.mean(axis=0), np.full(3, 1.0 / 3.0), atol=2.0e-9)
    assert metadata["sparsity"] > 0.0
    assert metadata["final_target_marginal_residual"] <= 2.0e-9


def test_quadratic_balanced_ot_is_invariant_to_row_cost_constants() -> None:
    cost = np.array([[0.1, 0.5, 2.0], [1.5, 0.2, 0.4]])
    _, reference, _ = solve_state_normalized_quadratic_balanced_ot(cost)
    _, actual, _ = solve_state_normalized_quadratic_balanced_ot(
        cost + np.array([7.0, 21.0])[:, None]
    )

    np.testing.assert_allclose(actual, reference, rtol=1.0e-8, atol=1.0e-9)


def test_v8_cost_is_exactly_the_frozen_v5_cost() -> None:
    n_source, n_target, g_source, g_target = _features()
    expected, _ = build_grnanchored_kl_cost(n_source, n_target, g_source, g_target)
    actual, metadata = CompareNGKlSparseOTGRNAnchorV8PijMethod().build_kl_cost(
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


def test_sparseot_v8_is_registered_without_rebinding_frozen_baseline() -> None:
    method = get_pij_method("compare_NG_kl_sparseot_grnanchor_v8")
    assert isinstance(method, CompareNGKlSparseOTGRNAnchorV8PijMethod)
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod


def test_sparseot_v8_config_requires_light_cci_grn() -> None:
    TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method="compare_NG_kl_sparseot_grnanchor_v8",
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    ).validate()
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(
            network_method="light_cci",
            pij_method="compare_NG_kl_sparseot_grnanchor_v8",
        ).validate()


def test_sparseot_v8_rejects_parameter_drift_and_invalid_solver_input() -> None:
    method = CompareNGKlSparseOTGRNAnchorV8PijMethod()
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
    with pytest.raises(ValueError, match="finite and nonnegative"):
        solve_state_normalized_quadratic_balanced_ot(np.array([[0.0, np.nan]]))
