from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp

from mignet_ce.pij.compare._shared.distances import robust_normalize_cost
from mignet_ce.pij.compare._shared.lowrank_fgw import (
    FGW_FACTORIZATION_SEED,
    FGW_STRUCTURE_RANK,
    _balance_cost_with_log_fallback,
    _directed_svd_factors,
    _row_stochastic_adjacency,
    _validate_adjacency,
)


MULTISCALE_DIFFUSION_STEPS = (1, 2, 4)
MULTISCALE_TEMPERATURE_SCHEDULE = (
    1.0,
    0.8,
    0.64,
    0.512,
    0.4096,
    0.32768,
    0.262144,
    0.2097152,
    0.16777216,
    0.134217728,
    0.1073741824,
    0.1,
)
MULTISCALE_STRUCTURE_WEIGHT = 1.0
ROUNDING_FALLBACK_SCALING_ITERATIONS = 200
ROUNDING_FALLBACK_COST_CAP = 30.0


def _round_to_uniform_marginals(joint: np.ndarray) -> np.ndarray:
    """Apply the standard nonnegative transport rounding correction."""
    values = np.maximum(np.asarray(joint, dtype=float), 0.0).copy()
    source_count, target_count = values.shape
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)

    row_mass = values.sum(axis=1)
    row_scale = np.minimum(
        np.divide(
            source_marginal,
            row_mass,
            out=np.ones_like(source_marginal),
            where=row_mass > 0.0,
        ),
        1.0,
    )
    values *= row_scale[:, None]
    column_mass = values.sum(axis=0)
    column_scale = np.minimum(
        np.divide(
            target_marginal,
            column_mass,
            out=np.ones_like(target_marginal),
            where=column_mass > 0.0,
        ),
        1.0,
    )
    values *= column_scale[None, :]

    source_residual = np.maximum(source_marginal - values.sum(axis=1), 0.0)
    target_residual = np.maximum(target_marginal - values.sum(axis=0), 0.0)
    missing_mass = float(source_residual.sum())
    if missing_mass > 0.0:
        values += np.outer(source_residual, target_residual) / missing_mass
    return np.maximum(values, 0.0)


def _balance_annealed_cost(
    cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    try:
        return _balance_cost_with_log_fallback(cost)
    except RuntimeError as error:
        values = np.asarray(cost, dtype=float)
        source_count, target_count = values.shape
        source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
        target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)
        row_centered = values - values.min(axis=1, keepdims=True)
        solver_cost = np.minimum(row_centered, ROUNDING_FALLBACK_COST_CAP)
        clipped_entries = int(np.count_nonzero(row_centered > ROUNDING_FALLBACK_COST_CAP))
        kernel = np.exp(-solver_cost)
        target_scale = np.ones(target_count, dtype=float)
        source_scale = np.ones(source_count, dtype=float)
        for _ in range(ROUNDING_FALLBACK_SCALING_ITERATIONS):
            source_scale = source_marginal / (kernel @ target_scale)
            target_scale = target_marginal / (kernel.T @ source_scale)
        approximate = source_scale[:, None] * kernel * target_scale[None, :]
        pre_rounding_residual = float(
            max(
                np.max(np.abs(approximate.sum(axis=1) - source_marginal)),
                np.max(np.abs(approximate.sum(axis=0) - target_marginal)),
            )
        )
        joint = _round_to_uniform_marginals(approximate)
        conditional = joint / source_marginal[:, None]
        post_rounding_residual = float(
            max(
                np.max(np.abs(joint.sum(axis=1) - source_marginal)),
                np.max(np.abs(joint.sum(axis=0) - target_marginal)),
            )
        )
        if not np.isfinite(joint).all() or post_rounding_residual > 1.0e-12:
            raise RuntimeError(
                "Multi-scale FGW transport rounding failed to recover uniform marginals; "
                f"residual={post_rounding_residual:.6g}."
            ) from error
        return joint, conditional, {
            "solver": "capped_primal_scaling_with_exact_transport_rounding",
            "log_domain_fallback_used": True,
            "rounding_fallback_used": True,
            "fallback_reason": str(error),
            "iterations": ROUNDING_FALLBACK_SCALING_ITERATIONS,
            "cost_cap": ROUNDING_FALLBACK_COST_CAP,
            "clipped_entries": clipped_entries,
            "clipped_fraction": float(clipped_entries / values.size),
            "pre_rounding_marginal_residual": pre_rounding_residual,
            "max_absolute_marginal_residual": post_rounding_residual,
        }


def _matrix_power_factors(
    left: np.ndarray,
    right: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Approximate a directed diffusion power without forming a dense square matrix."""
    if int(step) < 1:
        raise ValueError("Diffusion step must be positive.")
    if left.shape != right.shape:
        raise ValueError(f"Directed factors must have equal shapes; got {left.shape} and {right.shape}.")
    if left.shape[1] == 0:
        return left.copy(), right.copy(), {
            "step": int(step),
            "effective_rank": 0,
            "approximate_frobenius_norm": 0.0,
        }

    bridge = right.T @ left
    powered_left = left if int(step) == 1 else left @ np.linalg.matrix_power(bridge, int(step) - 1)
    powered_right = right.copy()
    left_gram = powered_left.T @ powered_left
    right_gram = powered_right.T @ powered_right
    frobenius_squared = max(float(np.trace(left_gram @ right_gram)), 0.0)
    frobenius_norm = float(np.sqrt(frobenius_squared))
    if frobenius_norm > 0.0:
        # Each diffusion scale contributes only through a separately robust-normalized
        # relational cost. Unit Frobenius scaling improves numerical conditioning and
        # cannot alter that scale's ordering.
        factor_scale = float(1.0 / np.sqrt(frobenius_norm))
        powered_left = powered_left * factor_scale
        powered_right = powered_right * factor_scale
    return powered_left, powered_right, {
        "step": int(step),
        "effective_rank": int(powered_left.shape[1]),
        "approximate_frobenius_norm_before_unit_scaling": frobenius_norm,
        "unit_frobenius_scaled": bool(frobenius_norm > 0.0),
    }


def _lowrank_directed_relational_cost(
    source_left: np.ndarray,
    source_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    source_count, target_count = coupling.shape
    if source_left.shape[0] != source_count or target_left.shape[0] != target_count:
        raise ValueError("Low-rank relation factors do not match the coupling shape.")
    if source_left.shape != source_right.shape or target_left.shape != target_right.shape:
        raise ValueError("Each directed relation requires equal left/right factor shapes.")
    if source_left.shape[1] == 0 or target_left.shape[1] == 0:
        return np.zeros((source_count, target_count), dtype=float)

    source_right_gram = source_right.T @ source_right
    target_right_gram = target_right.T @ target_right
    source_left_gram = source_left.T @ source_left
    target_left_gram = target_left.T @ target_left
    source_out_constant = np.einsum(
        "ir,rs,is->i", source_left, source_right_gram, source_left, optimize=True
    ) / float(source_count)
    target_out_constant = np.einsum(
        "ir,rs,is->i", target_left, target_right_gram, target_left, optimize=True
    ) / float(target_count)
    source_in_constant = np.einsum(
        "ir,rs,is->i", source_right, source_left_gram, source_right, optimize=True
    ) / float(source_count)
    target_in_constant = np.einsum(
        "ir,rs,is->i", target_right, target_left_gram, target_right, optimize=True
    ) / float(target_count)

    outgoing_middle = source_right.T @ coupling @ target_right
    incoming_middle = source_left.T @ coupling @ target_left
    outgoing_cross = source_left @ outgoing_middle @ target_left.T
    incoming_cross = source_right @ incoming_middle @ target_right.T
    outgoing = source_out_constant[:, None] + target_out_constant[None, :] - 2.0 * outgoing_cross
    incoming = source_in_constant[:, None] + target_in_constant[None, :] - 2.0 * incoming_cross
    structural = 0.5 * (outgoing + incoming)
    return np.maximum(
        np.nan_to_num(structural, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )


def _build_diffusion_factors(
    adjacency: sp.csr_matrix,
    *,
    steps: tuple[int, ...],
    rank: int,
    seed: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    left, right, base_metadata = _directed_svd_factors(
        adjacency,
        requested_rank=int(rank),
        seed=int(seed),
    )
    factors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scale_metadata: list[dict[str, object]] = []
    for step in steps:
        scale_left, scale_right, metadata = _matrix_power_factors(left, right, int(step))
        factors[int(step)] = (scale_left, scale_right)
        scale_metadata.append(metadata)
    return factors, {
        "base_factorization": base_metadata,
        "diffusion_scales": scale_metadata,
    }


def solve_multiscale_directed_fgw(
    node_cost: np.ndarray,
    source_adjacency: sp.spmatrix | np.ndarray,
    target_adjacency: sp.spmatrix | np.ndarray,
    *,
    diffusion_steps: tuple[int, ...] = MULTISCALE_DIFFUSION_STEPS,
    temperature_schedule: tuple[float, ...] = MULTISCALE_TEMPERATURE_SCHEDULE,
    structure_rank: int = FGW_STRUCTURE_RANK,
    structure_weight: float = MULTISCALE_STRUCTURE_WEIGHT,
    factorization_seed: int = FGW_FACTORIZATION_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve deterministic multi-scale directed FGW with fixed entropy continuation."""
    started = perf_counter()
    cost = np.asarray(node_cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError(f"Multi-scale FGW node cost must be non-empty and 2D; got {cost.shape}.")
    if not np.isfinite(cost).all() or np.any(cost < 0.0):
        raise ValueError("Multi-scale FGW node cost must be finite and nonnegative.")
    steps = tuple(int(step) for step in diffusion_steps)
    temperatures = tuple(float(value) for value in temperature_schedule)
    if not steps or any(step < 1 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("Diffusion steps must be unique positive integers.")
    if not temperatures or any((not np.isfinite(value)) or value <= 0.0 for value in temperatures):
        raise ValueError("Temperature schedule must contain finite positive values.")
    if any(temperatures[index] > temperatures[index - 1] for index in range(1, len(temperatures))):
        raise ValueError("Temperature schedule must be non-increasing.")
    if int(structure_rank) < 1:
        raise ValueError("structure_rank must be positive.")
    if float(structure_weight) < 0.0:
        raise ValueError("structure_weight must be nonnegative.")

    source = _row_stochastic_adjacency(_validate_adjacency(source_adjacency, name="source"))
    target = _row_stochastic_adjacency(_validate_adjacency(target_adjacency, name="target"))
    if source.shape[0] != cost.shape[0] or target.shape[0] != cost.shape[1]:
        raise ValueError(
            "Multi-scale FGW adjacency sizes must match the node cost; "
            f"cost={cost.shape}, source={source.shape}, target={target.shape}."
        )

    source_factors, source_factorization = _build_diffusion_factors(
        source,
        steps=steps,
        rank=int(structure_rank),
        seed=int(factorization_seed),
    )
    target_factors, target_factorization = _build_diffusion_factors(
        target,
        steps=steps,
        rank=int(structure_rank),
        seed=int(factorization_seed),
    )

    joint, conditional, initial_balance = _balance_annealed_cost(cost)
    iteration_metadata: list[dict[str, object]] = []
    final_scale_summaries: list[dict[str, object]] = []
    for iteration, temperature in enumerate(temperatures, start=1):
        previous = joint
        normalized_scale_costs: list[np.ndarray] = []
        scale_summaries: list[dict[str, object]] = []
        for step in steps:
            structural = _lowrank_directed_relational_cost(
                *source_factors[step],
                *target_factors[step],
                previous,
            )
            normalized, normalization = robust_normalize_cost(structural, copy=True)
            normalized_scale_costs.append(normalized)
            scale_summaries.append(
                {
                    "step": int(step),
                    "raw_min": float(np.min(structural)),
                    "raw_max": float(np.max(structural)),
                    "normalization": normalization,
                }
            )
        multiscale_structural = np.mean(np.stack(normalized_scale_costs, axis=0), axis=0)
        combined_cost = cost + float(structure_weight) * multiscale_structural
        annealed_cost = combined_cost / float(temperature)
        joint, conditional, balance_metadata = _balance_annealed_cost(annealed_cost)
        total_variation = float(0.5 * np.sum(np.abs(joint - previous)))
        iteration_metadata.append(
            {
                "iteration": int(iteration),
                "temperature": float(temperature),
                "coupling_total_variation": total_variation,
                "node_transport_cost": float(np.sum(joint * cost)),
                "multiscale_structural_transport_cost": float(
                    np.sum(joint * multiscale_structural)
                ),
                "balance_solver": balance_metadata["solver"],
                "balance_log_domain_fallback_used": bool(
                    balance_metadata["log_domain_fallback_used"]
                ),
                "balance_iterations": int(balance_metadata["iterations"]),
                "balance_marginal_residual": float(
                    balance_metadata["max_absolute_marginal_residual"]
                ),
            }
        )
        final_scale_summaries = scale_summaries

    return joint, conditional, {
        "mode": "fixed_schedule_multiscale_lowrank_directed_fgw",
        "objective": "frozen_V5_node_cost + mean_normalized_directed_diffusion_relational_cost",
        "diffusion_steps": list(steps),
        "temperature_schedule": list(temperatures),
        "outer_iterations": len(temperatures),
        "structure_rank": int(structure_rank),
        "structure_weight": float(structure_weight),
        "factorization_seed": int(factorization_seed),
        "source_adjacency_shape": list(source.shape),
        "target_adjacency_shape": list(target.shape),
        "source_adjacency_nnz": int(source.nnz),
        "target_adjacency_nnz": int(target.nnz),
        "source_factorization": source_factorization,
        "target_factorization": target_factorization,
        "initial_balance": initial_balance,
        "iterations": iteration_metadata,
        "final_scale_summaries": final_scale_summaries,
        "final_marginal_residual": float(
            iteration_metadata[-1]["balance_marginal_residual"]
        ),
        "elapsed_seconds": float(perf_counter() - started),
        "uses_ei_for_fitting": False,
        "uses_layer_identity": False,
        "uses_labels": False,
        "uses_third_timepoint": False,
        "uses_developmental_features": False,
    }
