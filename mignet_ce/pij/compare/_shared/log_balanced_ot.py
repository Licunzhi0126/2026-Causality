from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from mignet_ce.utils.matrix import safe_row_normalize


LOG_SCALING_MAX_ITERATIONS = 500
LOG_SCALING_POST_MAX_ITERATIONS = 5_000
LOG_DUAL_MAX_ITERATIONS = 1_000
LOG_BALANCE_TOLERANCE = 1.0e-9
LOG_BALANCE_CHECK_EVERY = 10
LOG_BALANCE_BLOCK_SIZE = 256


def _marginal_residuals(
    log_kernel: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
) -> tuple[float, float]:
    source_mass = np.exp(
        source_potential + logsumexp(log_kernel + target_potential[None, :], axis=1)
    )
    target_mass = np.exp(
        target_potential + logsumexp(log_kernel + source_potential[:, None], axis=0)
    )
    return (
        float(np.max(np.abs(source_mass - source_marginal))),
        float(np.max(np.abs(target_mass - target_marginal))),
    )


def _log_scaling(
    log_kernel: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    log_source_marginal = np.log(source_marginal)
    log_target_marginal = np.log(target_marginal)
    source = np.asarray(source_potential, dtype=float).copy()
    target = np.asarray(target_potential, dtype=float).copy()
    source_residual = float("inf")
    target_residual = float("inf")
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        source = log_source_marginal - logsumexp(log_kernel + target[None, :], axis=1)
        target = log_target_marginal - logsumexp(log_kernel + source[:, None], axis=0)
        if (iterations - 1) % LOG_BALANCE_CHECK_EVERY == 0 or iterations == int(max_iterations):
            source_residual, target_residual = _marginal_residuals(
                log_kernel,
                source,
                target,
                source_marginal,
                target_marginal,
            )
            if max(source_residual, target_residual) <= float(tolerance):
                converged = True
                break
    return source, target, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "source_marginal_residual": source_residual,
        "target_marginal_residual": target_residual,
        "max_absolute_marginal_residual": max(source_residual, target_residual),
    }


def _dual_objective_factory(
    cost: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    block_size: int,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    source_count, target_count = cost.shape

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        source_potential = parameters[:source_count]
        target_potential = np.concatenate([parameters[source_count:], np.zeros(1, dtype=float)])
        source_gradient = np.empty(source_count, dtype=float)
        target_mass = np.zeros(target_count, dtype=float)
        joint_mass = 0.0

        for start in range(0, source_count, block_size):
            stop = min(start + block_size, source_count)
            log_joint = (
                source_potential[start:stop, None]
                + target_potential[None, :]
                - cost[start:stop]
            )
            if float(np.max(log_joint)) > 700.0:
                return float(np.finfo(float).max), np.full_like(parameters, 1.0e100)
            joint = np.exp(log_joint)
            source_mass = joint.sum(axis=1)
            source_gradient[start:stop] = source_mass - source_marginal[start:stop]
            target_mass += joint.sum(axis=0)
            joint_mass += float(np.sum(source_mass))

        objective = (
            joint_mass
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


def _refine_in_dual(
    cost: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    *,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source_count = cost.shape[0]
    gauge = float(target_potential[-1])
    source_initial = np.asarray(source_potential, dtype=float) + gauge
    target_initial = np.asarray(target_potential, dtype=float) - gauge
    initial = np.concatenate([source_initial, target_initial[:-1]])
    optimized = minimize(
        _dual_objective_factory(
            cost,
            source_marginal,
            target_marginal,
            int(block_size),
        ),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 1.0e-30,
            "gtol": 1.0e-12,
            "maxiter": LOG_DUAL_MAX_ITERATIONS,
            "maxls": 100,
            "maxcor": 20,
        },
    )
    source = optimized.x[:source_count]
    target = np.concatenate([optimized.x[source_count:], np.zeros(1, dtype=float)])
    return source, target, {
        "used": True,
        "optimizer": "L-BFGS-B_on_log_entropic_OT_dual",
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "iterations": int(optimized.nit),
        "function_evaluations": int(optimized.nfev),
    }


def _joint_from_potentials(
    cost: np.ndarray,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    *,
    block_size: int,
) -> np.ndarray:
    joint = np.empty_like(cost, dtype=float)
    for start in range(0, cost.shape[0], int(block_size)):
        stop = min(start + int(block_size), cost.shape[0])
        joint[start:stop] = np.exp(
            source_potential[start:stop, None]
            + target_potential[None, :]
            - cost[start:stop]
        )
    return joint


def balance_cost_log_sinkhorn(
    cost: np.ndarray,
    *,
    tolerance: float = LOG_BALANCE_TOLERANCE,
    scaling_max_iterations: int = LOG_SCALING_MAX_ITERATIONS,
    post_scaling_max_iterations: int = LOG_SCALING_POST_MAX_ITERATIONS,
    block_size: int = LOG_BALANCE_BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Balance ``exp(-cost)`` with uniform marginals without materializing that kernel."""
    started = perf_counter()
    values = np.asarray(cost, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Log-balanced OT cost must be a non-empty 2D matrix; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Log-balanced OT cost must be finite and nonnegative.")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")
    if int(scaling_max_iterations) < 1 or int(post_scaling_max_iterations) < 1:
        raise ValueError("scaling iteration limits must be positive.")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive.")

    source_count, target_count = values.shape
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)
    row_centered = values - values.min(axis=1, keepdims=True)
    maximum_resolved_cost = float(
        -np.log(float(tolerance)) + np.log(float(max(source_count, target_count)))
    )
    solver_cost = np.minimum(row_centered, maximum_resolved_cost)
    clipped_entries = int(np.count_nonzero(row_centered > maximum_resolved_cost))
    log_kernel = -solver_cost
    source_potential = np.zeros(source_count, dtype=float)
    target_potential = np.zeros(target_count, dtype=float)
    source_potential, target_potential, initial_scaling = _log_scaling(
        log_kernel,
        source_marginal,
        target_marginal,
        source_potential,
        target_potential,
        max_iterations=int(scaling_max_iterations),
        tolerance=float(tolerance),
    )

    dual_metadata: dict[str, object] = {"used": False}
    post_scaling: dict[str, object] = {
        "converged": bool(initial_scaling["converged"]),
        "iterations": 0,
        "max_iterations": int(post_scaling_max_iterations),
        "tolerance": float(tolerance),
        "source_marginal_residual": initial_scaling["source_marginal_residual"],
        "target_marginal_residual": initial_scaling["target_marginal_residual"],
        "max_absolute_marginal_residual": initial_scaling["max_absolute_marginal_residual"],
    }
    if not bool(initial_scaling["converged"]):
        source_potential, target_potential, dual_metadata = _refine_in_dual(
            solver_cost,
            source_marginal,
            target_marginal,
            source_potential,
            target_potential,
            block_size=int(block_size),
        )
        source_potential, target_potential, post_scaling = _log_scaling(
            log_kernel,
            source_marginal,
            target_marginal,
            source_potential,
            target_potential,
            max_iterations=int(post_scaling_max_iterations),
            tolerance=float(tolerance),
        )
    if not bool(post_scaling["converged"]):
        raise RuntimeError(
            "Log-balanced OT did not reach the requested marginal tolerance; "
            f"residual={float(post_scaling['max_absolute_marginal_residual']):.6g}."
        )

    joint = _joint_from_potentials(
        solver_cost,
        source_potential,
        target_potential,
        block_size=int(block_size),
    )
    conditional = safe_row_normalize(joint)
    final_target = conditional.mean(axis=0)
    source_residual = float(np.max(np.abs(joint.sum(axis=1) - source_marginal)))
    target_residual = float(np.max(np.abs(final_target - target_marginal)))
    final_residual = max(source_residual, target_residual)
    if not np.isfinite(joint).all() or not np.isfinite(conditional).all():
        raise RuntimeError("Log-balanced OT produced non-finite output.")
    if final_residual > float(tolerance):
        raise RuntimeError(
            "Log-balanced OT materialization exceeded the marginal tolerance; "
            f"residual={final_residual:.6g}."
        )

    return joint, conditional, {
        "mode": "log_domain_balanced_entropic_ot_uniform_marginals",
        "solver": "log_scaling_then_L-BFGS-B_dual_then_log_scaling",
        "converged": True,
        "tolerance": float(tolerance),
        "source_marginal_policy": "uniform",
        "target_marginal_policy": "uniform",
        "cost_row_centered": True,
        "maximum_resolved_cost": maximum_resolved_cost,
        "cost_clip_rule": "-log(tolerance)+log(max(n_source,n_target))",
        "clipped_entries": clipped_entries,
        "clipped_fraction": float(clipped_entries / values.size),
        "initial_scaling": initial_scaling,
        "dual_refinement": dual_metadata,
        "post_scaling": post_scaling,
        "source_marginal_residual": source_residual,
        "target_marginal_residual": target_residual,
        "max_absolute_marginal_residual": final_residual,
        "elapsed_seconds": float(perf_counter() - started),
        "uses_ei_for_fitting": False,
        "uses_layer_identity": False,
        "uses_labels": False,
        "uses_third_timepoint": False,
    }
