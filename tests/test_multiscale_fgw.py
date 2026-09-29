import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.pij.compare._shared.lowrank_fgw import _balance_cost_with_log_fallback
from mignet_ce.pij.compare._shared.multiscale_fgw import (
    MULTISCALE_TEMPERATURE_SCHEDULE,
    solve_multiscale_directed_fgw,
)


def test_multiscale_fgw_is_deterministic_balanced_and_uses_fixed_schedule() -> None:
    node_cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.5, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    source = sp.csr_matrix(
        [[0.0, 2.0, 0.5], [0.2, 0.0, 1.5], [1.0, 0.3, 0.0]]
    )
    target = sp.csr_matrix(
        [[0.0, 1.5, 0.1], [0.5, 0.0, 2.0], [1.2, 0.4, 0.0]]
    )

    joint_a, conditional_a, metadata_a = solve_multiscale_directed_fgw(
        node_cost,
        source,
        target,
        structure_rank=2,
    )
    joint_b, conditional_b, metadata_b = solve_multiscale_directed_fgw(
        node_cost,
        source,
        target,
        structure_rank=2,
    )

    np.testing.assert_allclose(joint_a, joint_b, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(conditional_a, conditional_b, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(joint_a.sum(axis=1), np.full(3, 1.0 / 3.0), atol=1.0e-9)
    np.testing.assert_allclose(joint_a.sum(axis=0), np.full(3, 1.0 / 3.0), atol=1.0e-9)
    np.testing.assert_allclose(conditional_a.sum(axis=1), np.ones(3), atol=1.0e-12)
    assert metadata_a["diffusion_steps"] == [1, 2, 4]
    assert metadata_a["temperature_schedule"] == list(MULTISCALE_TEMPERATURE_SCHEDULE)
    assert metadata_a["outer_iterations"] == len(MULTISCALE_TEMPERATURE_SCHEDULE)
    assert metadata_a["uses_ei_for_fitting"] is False
    assert metadata_a["iterations"] == metadata_b["iterations"]


def test_multiscale_fgw_zero_graphs_reduce_to_final_annealed_node_transport() -> None:
    node_cost = np.array([[0.0, 2.0, 1.0], [1.5, 0.0, 0.5]], dtype=float)
    source = sp.csr_matrix((2, 2), dtype=float)
    target = sp.csr_matrix((3, 3), dtype=float)

    joint, conditional, metadata = solve_multiscale_directed_fgw(
        node_cost,
        source,
        target,
    )
    expected_joint, expected_conditional, _ = _balance_cost_with_log_fallback(
        node_cost / MULTISCALE_TEMPERATURE_SCHEDULE[-1]
    )

    np.testing.assert_allclose(joint, expected_joint, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(conditional, expected_conditional, rtol=1.0e-10, atol=1.0e-11)
    assert metadata["source_factorization"]["base_factorization"]["effective_rank"] == 0
    assert metadata["target_factorization"]["base_factorization"]["effective_rank"] == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diffusion_steps": ()}, "Diffusion steps"),
        ({"diffusion_steps": (1, 1)}, "Diffusion steps"),
        ({"temperature_schedule": (1.0, 2.0)}, "non-increasing"),
        ({"temperature_schedule": (1.0, 0.0)}, "finite positive"),
    ],
)
def test_multiscale_fgw_rejects_invalid_fixed_controls(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_multiscale_directed_fgw(
            np.ones((2, 3)),
            np.eye(2),
            np.eye(3),
            **kwargs,
        )
