import numpy as np
import pytest

from mignet_ce.pij.compare._shared.log_balanced_ot import balance_cost_log_sinkhorn


def test_log_balanced_ot_handles_extreme_dynamic_range() -> None:
    cost = np.array([[0.0, 1_000.0], [1_000.0, 0.0]], dtype=float)

    joint, conditional, metadata = balance_cost_log_sinkhorn(cost)

    np.testing.assert_allclose(joint.sum(axis=1), np.full(2, 0.5), atol=1.0e-10)
    np.testing.assert_allclose(joint.sum(axis=0), np.full(2, 0.5), atol=1.0e-10)
    assert float(np.max(np.abs(conditional - np.eye(2)))) <= 1.0e-9
    assert metadata["converged"] is True
    assert metadata["max_absolute_marginal_residual"] <= 1.0e-9


def test_log_balanced_ot_supports_rectangular_costs() -> None:
    cost = np.array([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]], dtype=float)

    joint, conditional, _ = balance_cost_log_sinkhorn(cost)

    np.testing.assert_allclose(joint.sum(axis=1), np.full(2, 0.5), atol=1.0e-9)
    np.testing.assert_allclose(joint.sum(axis=0), np.full(3, 1.0 / 3.0), atol=1.0e-9)
    np.testing.assert_allclose(conditional.sum(axis=1), np.ones(2), atol=1.0e-12)


def test_log_balanced_ot_is_invariant_to_row_constants() -> None:
    cost = np.array([[0.2, 1.0, 0.7], [2.0, 0.1, 0.8]], dtype=float)
    shifted = cost + np.array([[17.0], [3.0]])

    _, baseline, _ = balance_cost_log_sinkhorn(cost)
    _, shifted_result, _ = balance_cost_log_sinkhorn(shifted)

    np.testing.assert_allclose(shifted_result, baseline, rtol=1.0e-9, atol=1.0e-10)


@pytest.mark.parametrize(
    "cost",
    [
        np.array([1.0, 2.0]),
        np.array([[0.0, np.nan]]),
        np.array([[0.0, -1.0]]),
    ],
)
def test_log_balanced_ot_rejects_invalid_costs(cost: np.ndarray) -> None:
    with pytest.raises(ValueError):
        balance_cost_log_sinkhorn(cost)
