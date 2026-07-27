from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from mignet_ce.utils.matrix import safe_row_normalize


DEFAULT_MAX_ITERATIONS = 1_200
DEFAULT_MARGINAL_TOLERANCE = 1.0e-9
DEFAULT_BLOCK_SIZE = 256
POST_BALANCE_MAX_ITERATIONS = 5_000
OPTIMIZER_RESIDUAL_LIMIT = 1.0e-4


def _initial_source_potential(
    centered_cost: np.ndarray,
    regularization: float,
    source_marginal: np.ndarray,
) -> np.ndarray:
    """Satisfy every row marginal once while the target potential is zero."""
    source_potential = np.empty(centered_cost.shape[0], dtype=float)
    for row_index, row_cost in enumerate(centered_cost):
        values = -np.asarray(row_cost, dtype=float) / regularization
        ordered = np.sort(values)[::-1]
        cumulative = np.cumsum(ordered) - float(source_marginal[row_index])
        positions = np.arange(1, ordered.size + 1, dtype=float)
        active = ordered - cumulative / positions > 0.0
        if not np.any(active):
            raise RuntimeError("Quadratic OT simplex initialization found no active target.")
        rho = int(np.flatnonzero(active)[-1])
        threshold = float(cumulative[rho] / positions[rho])
        source_potential[row_index] = -regularization * threshold
    return source_potential


def _dual_objective_factory(
    centered_cost: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    regularization: float,
    block_size: int,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    source_count, target_count = centered_cost.shape

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        source_potential = parameters[:source_count]
        target_potential = np.concatenate([parameters[source_count:], np.zeros(1, dtype=float)])
        source_gradient = np.empty(source_count, dtype=float)
        target_mass = np.zeros(target_count, dtype=float)
        squared_positive_sum = 0.0

        for start in range(0, source_count, block_size):
            stop = min(start + block_size, source_count)
            positive = (
                source_potential[start:stop, None]
                + target_potential[None, :]
                - centered_cost[start:stop]
            )
            np.maximum(positive, 0.0, out=positive)
            squared_positive_sum += float(np.sum(positive * positive))
            block_mass = positive / regularization
            source_gradient[start:stop] = block_mass.sum(axis=1) - source_marginal[start:stop]
            target_mass += block_mass.sum(axis=0)

        objective = (
            0.5 * squared_positive_sum / regularization
            - float(source_marginal @ source_potential)
            - float(target_marginal @ target_potential)
        )
        gradient = np.concatenate(
            [source_gradient, (target_mass - target_marginal)[:-1]]
        )
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            return float(np.finfo(float).max), np.full_like(parameters, 1.0e100)
        return objective, gradient

    return objective_and_gradient


def _joint_from_potentials(
    centered_cost: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    regularization: float,
    block_size: int,
) -> np.ndarray:
    joint = np.empty_like(centered_cost, dtype=float)
    for start in range(0, centered_cost.shape[0], block_size):
        stop = min(start + block_size, centered_cost.shape[0])
        block = (
            source_potential[start:stop, None]
            + target_potential[None, :]
            - centered_cost[start:stop]
        )
        np.maximum(block, 0.0, out=block)
        joint[start:stop] = block / regularization
    return joint


def _balance_fixed_support(
    joint: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if np.any(joint.sum(axis=1) <= 0.0) or np.any(joint.sum(axis=0) <= 0.0):
        raise RuntimeError("Quadratic OT support does not cover every source and target state.")

    source_scale = np.ones(joint.shape[0], dtype=float)
    target_scale = np.ones(joint.shape[1], dtype=float)
    residual = float("inf")
    converged = False
    iterations = 0
    for iterations in range(1, POST_BALANCE_MAX_ITERATIONS + 1):
        source_scale = source_marginal / np.maximum(joint @ target_scale, 1.0e-300)
        target_scale = target_marginal / np.maximum(joint.T @ source_scale, 1.0e-300)
        if (iterations - 1) % 10 == 0 or iterations == POST_BALANCE_MAX_ITERATIONS:
            current_source = source_scale * (joint @ target_scale)
            current_target = target_scale * (joint.T @ source_scale)
            residual = float(
                max(
                    np.max(np.abs(current_source - source_marginal)),
                    np.max(np.abs(current_target - target_marginal)),
                )
            )
            if residual <= tolerance:
                converged = True
                break
    if not converged:
        raise RuntimeError(
            "Quadratic OT fixed-support balancing did not converge; "
            f"iterations={POST_BALANCE_MAX_ITERATIONS}, residual={residual:.6g}."
        )
    balanced = (source_scale[:, None] * joint) * target_scale[None, :]
    return balanced, {
        "converged": True,
        "iterations": int(iterations),
        "max_iterations": POST_BALANCE_MAX_ITERATIONS,
        "tolerance": float(tolerance),
        "max_absolute_marginal_residual": residual,
    }


def solve_state_normalized_quadratic_balanced_ot(
    cost: np.ndarray,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    marginal_tolerance: float = DEFAULT_MARGINAL_TOLERANCE,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve balanced squared-L2 OT with a state-count-normalized coefficient.

    The optimization is

        min_{Pi in U(a,b)} <Pi, C> + gamma/2 ||Pi||_F^2,

    with uniform marginals and gamma=min(n_source, n_target). The coefficient
    makes the quadratic penalty of a maximally concentrated uniform-marginal
    plan comparable across state-space sizes without using layer identities.
    """
    started = perf_counter()
    values = np.asarray(cost, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Quadratic OT cost must be a non-empty 2D matrix; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Quadratic OT cost must be finite and nonnegative.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")
    if float(marginal_tolerance) <= 0.0:
        raise ValueError("marginal_tolerance must be positive.")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive.")

    source_count, target_count = values.shape
    effective_state_count = min(source_count, target_count)
    regularization = float(effective_state_count)
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)

    # Row constants do not affect a transport optimum with fixed row marginals.
    centered_cost = values - values.min(axis=1, keepdims=True)
    source_initial = _initial_source_potential(
        centered_cost,
        regularization,
        source_marginal,
    )
    initial = np.concatenate([source_initial, np.zeros(target_count - 1, dtype=float)])
    objective = _dual_objective_factory(
        centered_cost,
        source_marginal,
        target_marginal,
        regularization,
        int(block_size),
    )
    optimized = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "maxcor": 20,
            "maxls": 50,
            "ftol": 1.0e-15,
            "gtol": 1.0e-10,
        },
    )
    source_potential = optimized.x[:source_count]
    target_potential = np.concatenate([optimized.x[source_count:], np.zeros(1, dtype=float)])
    joint_unbalanced = _joint_from_potentials(
        centered_cost,
        source_potential,
        target_potential,
        regularization,
        int(block_size),
    )
    optimizer_source_residual = float(
        np.max(np.abs(joint_unbalanced.sum(axis=1) - source_marginal))
    )
    optimizer_target_residual = float(
        np.max(np.abs(joint_unbalanced.sum(axis=0) - target_marginal))
    )
    optimizer_residual = max(optimizer_source_residual, optimizer_target_residual)
    if optimizer_residual > OPTIMIZER_RESIDUAL_LIMIT:
        raise RuntimeError(
            "Quadratic OT dual optimizer stopped too far from the transport polytope; "
            f"residual={optimizer_residual:.6g}."
        )

    joint, balance_metadata = _balance_fixed_support(
        joint_unbalanced,
        source_marginal,
        target_marginal,
        tolerance=float(marginal_tolerance),
    )
    conditional = safe_row_normalize(joint)
    final_target_marginal = conditional.mean(axis=0)
    final_residual = float(np.max(np.abs(final_target_marginal - target_marginal)))
    support = joint > 0.0
    nonzero_count = int(np.count_nonzero(support))
    elapsed = float(perf_counter() - started)

    metadata: dict[str, object] = {
        "mode": "state_count_normalized_squared_l2_balanced_ot",
        "objective": "<Pi,C> + min(n_source,n_target)/2 * ||Pi||_F^2",
        "solver": "blockwise_L-BFGS-B_dual_then_fixed_support_balance",
        "regularization": regularization,
        "effective_state_count": int(effective_state_count),
        "source_marginal_policy": "uniform",
        "target_marginal_policy": "uniform",
        "cost_row_centered": True,
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "optimizer_iterations": int(optimized.nit),
        "optimizer_function_evaluations": int(optimized.nfev),
        "optimizer_source_marginal_residual": optimizer_source_residual,
        "optimizer_target_marginal_residual": optimizer_target_residual,
        "optimizer_max_marginal_residual": optimizer_residual,
        "post_balance": balance_metadata,
        "final_target_marginal_residual": final_residual,
        "nonzero_count": nonzero_count,
        "total_entries": int(joint.size),
        "density": float(nonzero_count / joint.size),
        "sparsity": float(1.0 - nonzero_count / joint.size),
        "transport_cost": float(np.sum(joint * values)),
        "quadratic_penalty": float(0.5 * regularization * np.sum(joint * joint)),
        "elapsed_seconds": elapsed,
        "uses_ei_for_fitting": False,
        "uses_layer_identity": False,
        "uses_labels": False,
        "uses_third_timepoint": False,
    }
    return joint, conditional, metadata
