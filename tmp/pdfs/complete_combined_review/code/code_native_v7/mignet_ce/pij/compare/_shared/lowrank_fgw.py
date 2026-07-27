from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.distances import robust_normalize_cost
from mignet_ce.pij.compare._shared.log_balanced_ot import balance_cost_log_sinkhorn
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn


FGW_OUTER_ITERATIONS = 10
FGW_STRUCTURE_RANK = 16
FGW_STRUCTURE_WEIGHT = 1.0
FGW_FACTORIZATION_SEED = 20260722


def _balance_cost_with_log_fallback(
    cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    kernel, _ = row_normalized_kernel_from_cost(cost, tau=1.0)
    try:
        joint, conditional, details = balance_kernel_sinkhorn(kernel)
        return joint, conditional, {
            "solver": "frozen_v7_primal_sinkhorn",
            "log_domain_fallback_used": False,
            "iterations": int(details["iterations"]),
            "max_absolute_marginal_residual": float(
                details["max_absolute_marginal_residual"]
            ),
            "details": details,
        }
    except RuntimeError as error:
        joint, conditional, details = balance_cost_log_sinkhorn(cost)
        iterations = int(details["initial_scaling"]["iterations"]) + int(
            details["post_scaling"]["iterations"]
        )
        return joint, conditional, {
            "solver": "log_domain_balanced_ot_fallback",
            "log_domain_fallback_used": True,
            "fallback_reason": str(error),
            "iterations": iterations,
            "max_absolute_marginal_residual": float(
                details["max_absolute_marginal_residual"]
            ),
            "details": details,
        }


def _validate_adjacency(matrix: sp.spmatrix | np.ndarray, *, name: str) -> sp.csr_matrix:
    values = matrix.tocsr(copy=True).astype(float) if sp.issparse(matrix) else sp.csr_matrix(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} adjacency must be a non-empty square matrix; got {values.shape}.")
    if values.nnz:
        if not np.isfinite(values.data).all() or np.any(values.data < 0.0):
            raise ValueError(f"{name} adjacency must be finite and nonnegative.")
        values.eliminate_zeros()
    return values


def _row_stochastic_adjacency(matrix: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0.0)
    return (sp.diags(inverse, format="csr") @ matrix).tocsr()


def _directed_svd_factors(
    matrix: sp.csr_matrix,
    *,
    requested_rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    state_count = matrix.shape[0]
    maximum_rank = min(int(requested_rank), state_count)
    frobenius_squared = float(np.sum(matrix.data * matrix.data)) if matrix.nnz else 0.0
    if maximum_rank < 1 or frobenius_squared <= 0.0:
        return (
            np.zeros((state_count, 0), dtype=float),
            np.zeros((state_count, 0), dtype=float),
            {
                "requested_rank": int(requested_rank),
                "effective_rank": 0,
                "state_count": int(state_count),
                "frobenius_norm": float(np.sqrt(frobenius_squared)),
                "relative_reconstruction_error": 0.0,
                "solver": "zero_matrix",
                "seed": int(seed),
            },
        )

    if maximum_rank >= state_count:
        dense = matrix.toarray()
        left_vectors, singular_values, right_transpose = np.linalg.svd(dense, full_matrices=False)
        solver = "dense_deterministic_svd"
    else:
        left_vectors, singular_values, right_transpose = svds(
            matrix,
            k=maximum_rank,
            which="LM",
            random_state=int(seed),
            return_singular_vectors=True,
        )
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        left_vectors = left_vectors[:, order]
        right_transpose = right_transpose[order]
        solver = "scipy_sparse_svds"

    singular_values = np.maximum(np.asarray(singular_values, dtype=float), 0.0)
    square_root = np.sqrt(singular_values)
    left_factor = np.asarray(left_vectors, dtype=float) * square_root[None, :]
    right_factor = np.asarray(right_transpose.T, dtype=float) * square_root[None, :]
    retained_squared = float(np.sum(singular_values * singular_values))
    residual_squared = max(frobenius_squared - retained_squared, 0.0)
    relative_error = float(np.sqrt(residual_squared / frobenius_squared))
    return left_factor, right_factor, {
        "requested_rank": int(requested_rank),
        "effective_rank": int(singular_values.size),
        "state_count": int(state_count),
        "frobenius_norm": float(np.sqrt(frobenius_squared)),
        "retained_singular_value_squared_fraction": float(
            min(retained_squared / frobenius_squared, 1.0)
        ),
        "relative_reconstruction_error": relative_error,
        "solver": solver,
        "seed": int(seed),
    }


def _directed_structural_cost(
    source_adjacency: sp.csr_matrix,
    target_adjacency: sp.csr_matrix,
    source_left: np.ndarray,
    source_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    source_count, target_count = coupling.shape
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)

    source_out_constant = np.asarray(source_adjacency.power(2) @ source_marginal).ravel()
    target_out_constant = np.asarray(target_adjacency.power(2) @ target_marginal).ravel()
    source_in_constant = np.asarray(source_adjacency.T.power(2) @ source_marginal).ravel()
    target_in_constant = np.asarray(target_adjacency.T.power(2) @ target_marginal).ravel()

    outgoing_middle = source_right.T @ coupling @ target_right
    incoming_middle = source_left.T @ coupling @ target_left
    outgoing_cross = source_left @ outgoing_middle @ target_left.T
    incoming_cross = source_right @ incoming_middle @ target_right.T
    outgoing = (
        source_out_constant[:, None]
        + target_out_constant[None, :]
        - 2.0 * outgoing_cross
    )
    incoming = (
        source_in_constant[:, None]
        + target_in_constant[None, :]
        - 2.0 * incoming_cross
    )
    structural = 0.5 * (outgoing + incoming)
    return np.maximum(np.nan_to_num(structural, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def solve_lowrank_directed_fgw(
    node_cost: np.ndarray,
    source_adjacency: sp.spmatrix | np.ndarray,
    target_adjacency: sp.spmatrix | np.ndarray,
    *,
    outer_iterations: int = FGW_OUTER_ITERATIONS,
    structure_rank: int = FGW_STRUCTURE_RANK,
    structure_weight: float = FGW_STRUCTURE_WEIGHT,
    factorization_seed: int = FGW_FACTORIZATION_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Run fixed-iteration directed low-rank FGW from the frozen V5 node cost."""
    started = perf_counter()
    cost = np.asarray(node_cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError(f"FGW node cost must be a non-empty 2D matrix; got {cost.shape}.")
    if not np.isfinite(cost).all() or np.any(cost < 0.0):
        raise ValueError("FGW node cost must be finite and nonnegative.")
    if int(outer_iterations) < 1:
        raise ValueError("outer_iterations must be positive.")
    if int(structure_rank) < 1:
        raise ValueError("structure_rank must be positive.")
    if float(structure_weight) < 0.0:
        raise ValueError("structure_weight must be nonnegative.")

    source = _row_stochastic_adjacency(
        _validate_adjacency(source_adjacency, name="source")
    )
    target = _row_stochastic_adjacency(
        _validate_adjacency(target_adjacency, name="target")
    )
    if source.shape[0] != cost.shape[0] or target.shape[0] != cost.shape[1]:
        raise ValueError(
            "FGW adjacency sizes must match the node cost; "
            f"cost={cost.shape}, source={source.shape}, target={target.shape}."
        )

    source_left, source_right, source_factorization = _directed_svd_factors(
        source,
        requested_rank=int(structure_rank),
        seed=int(factorization_seed),
    )
    target_left, target_right, target_factorization = _directed_svd_factors(
        target,
        requested_rank=int(structure_rank),
        seed=int(factorization_seed),
    )

    joint, conditional, initial_sinkhorn = _balance_cost_with_log_fallback(cost)
    iteration_metadata: list[dict[str, object]] = []
    final_structural_cost = np.zeros_like(cost)
    final_structural_normalization: dict[str, object] = {}
    for iteration in range(1, int(outer_iterations) + 1):
        previous = joint
        structural_cost = _directed_structural_cost(
            source,
            target,
            source_left,
            source_right,
            target_left,
            target_right,
            previous,
        )
        normalized_structural, structural_normalization = robust_normalize_cost(
            structural_cost,
            copy=True,
        )
        combined_cost = cost + float(structure_weight) * normalized_structural
        joint, conditional, sinkhorn_metadata = _balance_cost_with_log_fallback(combined_cost)
        total_variation = float(0.5 * np.sum(np.abs(joint - previous)))
        iteration_metadata.append(
            {
                "iteration": int(iteration),
                "coupling_total_variation": total_variation,
                "node_transport_cost": float(np.sum(joint * cost)),
                "structural_linearized_cost": float(np.sum(joint * normalized_structural)),
                "sinkhorn_solver": sinkhorn_metadata["solver"],
                "sinkhorn_log_domain_fallback_used": bool(
                    sinkhorn_metadata["log_domain_fallback_used"]
                ),
                "sinkhorn_iterations": int(sinkhorn_metadata["iterations"]),
                "sinkhorn_marginal_residual": float(
                    sinkhorn_metadata["max_absolute_marginal_residual"]
                ),
            }
        )
        final_structural_cost = structural_cost
        final_structural_normalization = structural_normalization

    return joint, conditional, {
        "mode": "fixed_iteration_lowrank_directed_fused_gromov_wasserstein",
        "objective": "frozen_V5_node_cost + normalized_directed_graph_relational_cost",
        "outer_iterations": int(outer_iterations),
        "structure_rank": int(structure_rank),
        "structure_weight": float(structure_weight),
        "factorization_seed": int(factorization_seed),
        "source_adjacency_shape": list(source.shape),
        "target_adjacency_shape": list(target.shape),
        "source_adjacency_nnz": int(source.nnz),
        "target_adjacency_nnz": int(target.nnz),
        "source_factorization": source_factorization,
        "target_factorization": target_factorization,
        "initial_sinkhorn": initial_sinkhorn,
        "iterations": iteration_metadata,
        "final_structural_cost_min": float(np.min(final_structural_cost)),
        "final_structural_cost_max": float(np.max(final_structural_cost)),
        "final_structural_normalization": final_structural_normalization,
        "final_marginal_residual": float(
            iteration_metadata[-1]["sinkhorn_marginal_residual"]
        ),
        "elapsed_seconds": float(perf_counter() - started),
        "uses_ei_for_fitting": False,
        "uses_layer_identity": False,
        "uses_labels": False,
        "uses_third_timepoint": False,
    }
