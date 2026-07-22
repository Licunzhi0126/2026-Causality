import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.lowrank_fgw import solve_lowrank_directed_fgw
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn


def test_lowrank_fgw_is_deterministic_and_balanced() -> None:
    node_cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.5, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    source = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.5],
                [0.2, 0.0, 1.5],
                [1.0, 0.3, 0.0],
            ]
        )
    )
    target = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.5, 0.1],
                [0.5, 0.0, 2.0],
                [1.2, 0.4, 0.0],
            ]
        )
    )

    joint_a, conditional_a, metadata_a = solve_lowrank_directed_fgw(
        node_cost,
        source,
        target,
        structure_rank=2,
    )
    joint_b, conditional_b, metadata_b = solve_lowrank_directed_fgw(
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
    assert metadata_a["outer_iterations"] == 10
    assert metadata_a["uses_ei_for_fitting"] is False
    assert metadata_a["iterations"] == metadata_b["iterations"]


def test_lowrank_fgw_reduces_to_node_sinkhorn_for_zero_graphs() -> None:
    node_cost = np.array([[0.0, 2.0, 1.0], [1.5, 0.0, 0.5]], dtype=float)
    source = sp.csr_matrix((2, 2), dtype=float)
    target = sp.csr_matrix((3, 3), dtype=float)

    joint, conditional, metadata = solve_lowrank_directed_fgw(
        node_cost,
        source,
        target,
    )
    kernel, _ = row_normalized_kernel_from_cost(node_cost, tau=1.0)
    expected_joint, expected_conditional, _ = balance_kernel_sinkhorn(kernel)

    np.testing.assert_allclose(joint, expected_joint, rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(conditional, expected_conditional, rtol=1.0e-11, atol=1.0e-12)
    assert metadata["source_factorization"]["effective_rank"] == 0
    assert metadata["target_factorization"]["effective_rank"] == 0


@pytest.mark.parametrize(
    ("node_cost", "source", "target"),
    [
        (np.ones((2, 3)), np.eye(3), np.eye(3)),
        (np.ones((2, 3)), np.eye(2), np.eye(2)),
        (np.ones((2, 3)), np.array([[0.0, -1.0], [0.0, 0.0]]), np.eye(3)),
    ],
)
def test_lowrank_fgw_rejects_invalid_adjacencies(
    node_cost: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        solve_lowrank_directed_fgw(node_cost, source, target)
