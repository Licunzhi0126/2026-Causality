from __future__ import annotations

"""Shared NumPy/Torch N/G KL-cost and balanced-OT equations."""

import numpy as np
import torch

from mignet_ce.metrics import effective_information
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.distances import (
    robust_normalize_cost,
    summarize_dense_cost,
)
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare._shared.log_balanced_ot import balance_cost_log_sinkhorn
from mignet_ce.utils.matrix import safe_row_normalize

EPS = 1e-12
TORCH_SINKHORN_ITERATIONS = 96
# DEPRECATION-CANDIDATE(user-removal): Native-V7 compatibility constants.
# Canonical production code does not consume this block.
NATIVE_V7_FEATURE_BETA = 0.05
NATIVE_V7_G_SCALE = 1.0
NATIVE_V7_N_WEIGHT = 0.25

# Canonical production N/G transition contract. Quantiles are fractions here;
# NumPy percentile APIs receive these values multiplied by 100, while Torch
# quantile APIs consume the fractions directly.
CANONICAL_TRANSITION_PROTOCOL = "canonical_ng_v1"
CANONICAL_FEATURE_BETA_N = 0.05
CANONICAL_FEATURE_BETA_G = 0.05
CANONICAL_ALPHA_CCI = 0.10
CANONICAL_TEMPERATURE = 0.10
CANONICAL_ROBUST_LOWER_QUANTILE = 0.05
CANONICAL_ROBUST_UPPER_QUANTILE = 0.95
CANONICAL_SINKHORN_MAX_ITERATIONS = 2_000
CANONICAL_SINKHORN_TOLERANCE = 1.0e-9
CANONICAL_SINKHORN_CHECK_EVERY = 10
CANONICAL_TORCH_SINKHORN_ITERATIONS = 512


def canonical_transition_contract() -> dict[str, object]:
    """Return the manifest-safe canonical transition contract."""

    return {
        "transition_protocol": CANONICAL_TRANSITION_PROTOCOL,
        "alpha_cci": CANONICAL_ALPHA_CCI,
        "tau": CANONICAL_TEMPERATURE,
        "beta_n": CANONICAL_FEATURE_BETA_N,
        "beta_g": CANONICAL_FEATURE_BETA_G,
        "component_normalization": "independent_robust_5_95",
        "combined_scale_control": "mixed_q95_minus_q05_no_clipping",
    }


def _canonical_robust_span_numpy(
    values: np.ndarray,
    *,
    lower_quantile: float = CANONICAL_ROBUST_LOWER_QUANTILE,
    upper_quantile: float = CANONICAL_ROBUST_UPPER_QUANTILE,
) -> tuple[float, float, float, str]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot estimate a robust span from an empty/non-finite matrix.")
    q_low = float(np.quantile(finite, float(lower_quantile)))
    q_high = float(np.quantile(finite, float(upper_quantile)))
    span = q_high - q_low
    mode = "quantile_5_95"
    if not np.isfinite(span) or span <= EPS:
        span = float(finite.max() - finite.min())
        mode = "full_range_fallback"
    if not np.isfinite(span) or span <= EPS:
        span = 1.0
        mode = "unit_fallback"
    return q_low, q_high, span, mode


def build_ng_component_costs_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta_n: float = CANONICAL_FEATURE_BETA_N,
    beta_g: float = CANONICAL_FEATURE_BETA_G,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build independently Robust5-95-normalized CCI/N and GRN/G costs."""

    beta_n = float(beta_n)
    beta_g = float(beta_g)
    if beta_n <= 0.0 or beta_g <= 0.0:
        raise ValueError("beta_n and beta_g must be positive.")
    d_n_raw = pairwise_feature_kl(n_source, n_target, beta=beta_n)
    d_g_raw = pairwise_feature_kl(g_source, g_target, beta=beta_g)
    if d_n_raw.shape != d_g_raw.shape:
        raise ValueError(
            f"N and G KL cost shapes differ: {d_n_raw.shape} vs {d_g_raw.shape}."
        )
    percentile_low = 100.0 * CANONICAL_ROBUST_LOWER_QUANTILE
    percentile_high = 100.0 * CANONICAL_ROBUST_UPPER_QUANTILE
    d_n, n_normalization = robust_normalize_cost(
        d_n_raw,
        lower_percentile=percentile_low,
        upper_percentile=percentile_high,
        copy=True,
    )
    d_g, g_normalization = robust_normalize_cost(
        d_g_raw,
        lower_percentile=percentile_low,
        upper_percentile=percentile_high,
        copy=True,
    )
    return d_n, d_g, {
        "transition_protocol": CANONICAL_TRANSITION_PROTOCOL,
        "beta_n": beta_n,
        "beta_g": beta_g,
        "component_normalization": "independent_robust_5_95",
        "D_N_raw": summarize_dense_cost(d_n_raw),
        "D_G_raw": summarize_dense_cost(d_g_raw),
        "D_N_normalization": n_normalization,
        "D_G_normalization": g_normalization,
        "D_N_normalized": summarize_dense_cost(d_n),
        "D_G_normalized": summarize_dense_cost(d_g),
        "component_costs_clipped_to_unit_interval": True,
    }


def mix_ng_cost_numpy(
    d_n: np.ndarray,
    d_g: np.ndarray,
    *,
    alpha_cci: float = CANONICAL_ALPHA_CCI,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mix normalized components and control global scale without clipping."""

    alpha = float(alpha_cci)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_cci must lie in [0, 1].")
    n_values = np.asarray(d_n, dtype=float)
    g_values = np.asarray(d_g, dtype=float)
    if n_values.shape != g_values.shape:
        raise ValueError(f"N/G normalized cost shapes differ: {n_values.shape} vs {g_values.shape}.")
    if n_values.ndim != 2 or n_values.size == 0:
        raise ValueError(f"N/G normalized costs must be non-empty 2D matrices; got {n_values.shape}.")
    if not np.isfinite(n_values).all() or not np.isfinite(g_values).all():
        raise ValueError("N/G normalized costs must be finite.")
    g_term = (1.0 - alpha) * g_values
    n_term = alpha * n_values
    mixed = g_term + n_term
    q05, q95, span, span_mode = _canonical_robust_span_numpy(mixed)
    controlled = mixed / span
    denominator = float(np.mean(mixed))
    actual_grn_share = (
        float(np.mean(g_term) / denominator) if denominator > EPS else 1.0 - alpha
    )
    return controlled, {
        "alpha_cci": alpha,
        "nominal_grn_weight": 1.0 - alpha,
        "nominal_cci_weight": alpha,
        "actual_grn_mean_cost_share": actual_grn_share,
        "mixed_q05": q05,
        "mixed_q95": q95,
        "mixed_robust_span": span,
        "mixed_span_mode": span_mode,
        "combined_scale_control": "mixed_q95_minus_q05_no_clipping",
        "combined_cost_clipped": False,
        "combined_cost": summarize_dense_cost(controlled),
    }


def build_canonical_ng_cost_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta_n: float = CANONICAL_FEATURE_BETA_N,
    beta_g: float = CANONICAL_FEATURE_BETA_G,
    alpha_cci: float = CANONICAL_ALPHA_CCI,
) -> tuple[np.ndarray, dict[str, object]]:
    d_n, d_g, component_metadata = build_ng_component_costs_numpy(
        n_source,
        n_target,
        g_source,
        g_target,
        beta_n=beta_n,
        beta_g=beta_g,
    )
    cost, mix_metadata = mix_ng_cost_numpy(d_n, d_g, alpha_cci=alpha_cci)
    return cost, {
        **component_metadata,
        **mix_metadata,
        "formula": (
            "C=((1-alpha_cci)*Robust5_95(KL(G))+alpha_cci*Robust5_95(KL(N)))"
            "/RobustSpan5_95(C_pre)"
        ),
    }


def _balance_kernel_uniform_numpy(
    kernel: np.ndarray,
    *,
    max_iterations: int = CANONICAL_SINKHORN_MAX_ITERATIONS,
    tolerance: float = CANONICAL_SINKHORN_TOLERANCE,
    check_every: int = CANONICAL_SINKHORN_CHECK_EVERY,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.asarray(kernel, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Sinkhorn kernel must be a non-empty 2D matrix; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Sinkhorn kernel must be finite and nonnegative.")
    if np.any(values.sum(axis=1) <= 0.0) or np.any(values.sum(axis=0) <= 0.0):
        raise RuntimeError("Sinkhorn kernel lost positive row/column support.")
    source_count, target_count = values.shape
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)
    target_scale = np.ones(target_count, dtype=float)
    source_scale = np.ones(source_count, dtype=float)
    residual = float("inf")
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        kernel_times_target = values @ target_scale
        if not np.isfinite(kernel_times_target).all() or np.any(kernel_times_target <= 0.0):
            raise RuntimeError("Sinkhorn source scaling became unsupported.")
        source_scale = source_marginal / kernel_times_target
        kernel_transpose_times_source = values.T @ source_scale
        if not np.isfinite(kernel_transpose_times_source).all() or np.any(
            kernel_transpose_times_source <= 0.0
        ):
            raise RuntimeError("Sinkhorn target scaling became unsupported.")
        target_scale = target_marginal / kernel_transpose_times_source
        if (iterations - 1) % int(check_every) == 0 or iterations == int(max_iterations):
            current_source = source_scale * (values @ target_scale)
            current_target = target_scale * (values.T @ source_scale)
            residual = float(
                max(
                    np.max(np.abs(current_source - source_marginal)),
                    np.max(np.abs(current_target - target_marginal)),
                )
            )
            if residual <= float(tolerance):
                converged = True
                break
    if not converged:
        raise RuntimeError(
            "Canonical balanced Sinkhorn did not converge; "
            f"iterations={iterations}, residual={residual:.6g}."
        )
    joint = (source_scale[:, None] * values) * target_scale[None, :]
    conditional = safe_row_normalize(joint)
    source_residual = float(np.max(np.abs(joint.sum(axis=1) - source_marginal)))
    target_residual = float(
        np.max(np.abs(conditional.mean(axis=0) - target_marginal))
    )
    return joint, conditional, {
        "mode": "balanced_entropic_ot_uniform_marginals",
        "converged": True,
        "iterations": int(iterations),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "source_marginal_policy": "uniform",
        "target_marginal_policy": "uniform",
        "source_marginal_residual": source_residual,
        "target_marginal_residual": target_residual,
        "max_absolute_marginal_residual": max(source_residual, target_residual),
    }


def canonical_ng_pij_from_cost_numpy(
    cost: np.ndarray,
    *,
    temperature: float = CANONICAL_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.asarray(cost, dtype=float)
    temperature = float(temperature)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Canonical N/G cost must be a non-empty 2D matrix; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Canonical N/G cost must be finite and nonnegative.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    kernel, prebalanced = row_normalized_kernel_from_cost(values, tau=temperature)
    try:
        joint, pij, sinkhorn = _balance_kernel_uniform_numpy(kernel)
        sinkhorn = {**sinkhorn, "log_domain_fallback_used": False}
    except RuntimeError as error:
        joint, pij, sinkhorn = balance_cost_log_sinkhorn(values / temperature)
        sinkhorn = {
            **sinkhorn,
            "log_domain_fallback_used": True,
            "standard_sinkhorn_error_type": type(error).__name__,
            "standard_sinkhorn_error": str(error),
        }
    return joint, pij, {
        "transition_protocol": CANONICAL_TRANSITION_PROTOCOL,
        "temperature": temperature,
        "tau": temperature,
        "kernel": summarize_dense_cost(kernel),
        "prebalanced_ei": float(effective_information(prebalanced.copy())),
        "sinkhorn": sinkhorn,
    }


def canonical_ng_pij_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta_n: float = CANONICAL_FEATURE_BETA_N,
    beta_g: float = CANONICAL_FEATURE_BETA_G,
    alpha_cci: float = CANONICAL_ALPHA_CCI,
    temperature: float = CANONICAL_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cost, cost_metadata = build_canonical_ng_cost_numpy(
        n_source,
        n_target,
        g_source,
        g_target,
        beta_n=beta_n,
        beta_g=beta_g,
        alpha_cci=alpha_cci,
    )
    joint, pij, transition_metadata = canonical_ng_pij_from_cost_numpy(
        cost,
        temperature=temperature,
    )
    return joint, pij, {
        **cost_metadata,
        **transition_metadata,
        "cost": cost_metadata,
    }


def build_ng_kl_cost_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta_n: float,
    beta_g: float,
    g_scale: float,
    n_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """DEPRECATION-CANDIDATE(user-removal): build the old asymmetric cost."""
    beta_n = float(beta_n)
    beta_g = float(beta_g)
    g_scale = float(g_scale)
    n_weight = float(n_weight)
    if beta_n <= 0.0 or beta_g <= 0.0:
        raise ValueError("beta_n and beta_g must be positive.")
    if g_scale < 0.0 or n_weight < 0.0:
        raise ValueError("g_scale and n_weight must be nonnegative.")

    n_cost = pairwise_feature_kl(n_source, n_target, beta=beta_n)
    g_cost = pairwise_feature_kl(g_source, g_target, beta=beta_g)
    if n_cost.shape != g_cost.shape:
        raise ValueError(f"N and G KL cost shapes differ: {n_cost.shape} vs {g_cost.shape}.")
    normalized_n, n_normalization = robust_normalize_cost(n_cost, copy=True)
    combined = g_scale * g_cost + n_weight * normalized_n
    if not np.isfinite(combined).all() or np.any(combined < 0.0):
        raise ValueError("N/G KL cost must be finite and nonnegative.")

    return combined, {
        "mode": "scaled_raw_grn_kl_plus_bounded_n_correction",
        "formula": "g_scale*KL(G,beta_g)+n_weight*Robust5_95(KL(N,beta_n))",
        "beta_n": beta_n,
        "beta_g": beta_g,
        "g_scale": g_scale,
        "n_weight": n_weight,
        "n_cost": summarize_dense_cost(n_cost),
        "g_cost": summarize_dense_cost(g_cost),
        "n_normalization": n_normalization,
        "combined_cost": summarize_dense_cost(combined),
        "grn_cost_scale": f"{g_scale:g}_times_raw_kl_nats",
        "n_correction_scale": f"robust_5_95_times_{n_weight:g}",
        "final_cost_clipped_to_unit_interval": False,
    }


def pairwise_kl_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    source_prob = torch.softmax(source / float(beta), dim=1).clamp_min(EPS)
    target_prob = torch.softmax(target / float(beta), dim=1).clamp_min(EPS)
    source_entropy = (source_prob * torch.log(source_prob)).sum(dim=1, keepdim=True)
    return (source_entropy - source_prob @ torch.log(target_prob).T).clamp_min(0.0)


def robust_normalize_torch(cost: torch.Tensor) -> torch.Tensor:
    """DEPRECATION-CANDIDATE(user-removal): Native-V7-only normalizer."""

    flat = cost.reshape(-1)
    lower = torch.quantile(flat, 0.05)
    upper = torch.quantile(flat, 0.95)
    scale = (upper - lower).clamp_min(EPS)
    return ((cost - lower) / scale).clamp(0.0, 1.0)


def balanced_sinkhorn_torch_from_cost(
    cost: torch.Tensor,
    *,
    temperature: float = 1.0,
    iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError(f"Torch Sinkhorn cost must be non-empty 2D; got {tuple(cost.shape)}.")
    if int(iterations) <= 0:
        raise ValueError("Torch Sinkhorn iterations must be positive.")
    if float(temperature) <= 0.0:
        raise ValueError("Torch Sinkhorn temperature must be positive.")
    source_count, target_count = cost.shape
    log_kernel = -cost / float(temperature)
    log_source = torch.full(
        (source_count,),
        -float(np.log(source_count)),
        dtype=cost.dtype,
        device=cost.device,
    )
    log_target = torch.full(
        (target_count,),
        -float(np.log(target_count)),
        dtype=cost.dtype,
        device=cost.device,
    )
    source_potential = torch.zeros_like(log_source)
    target_potential = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        source_potential = log_source - torch.logsumexp(
            log_kernel + target_potential[None, :],
            dim=1,
        )
        target_potential = log_target - torch.logsumexp(
            log_kernel + source_potential[:, None],
            dim=0,
        )
    joint = torch.exp(
        log_kernel + source_potential[:, None] + target_potential[None, :]
    )
    return joint / joint.sum(dim=1, keepdim=True).clamp_min(EPS)


def ng_kl_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    g_source: torch.Tensor,
    g_target: torch.Tensor,
    *,
    beta_n: float,
    beta_g: float,
    g_scale: float,
    n_weight: float,
    temperature: float = 1.0,
    sinkhorn_iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    """DEPRECATION-CANDIDATE(user-removal): old asymmetric Torch transition."""
    if float(g_scale) < 0.0 or float(n_weight) < 0.0:
        raise ValueError("g_scale and n_weight must be nonnegative.")
    n_cost = pairwise_kl_torch(n_source, n_target, beta=beta_n)
    g_cost = pairwise_kl_torch(g_source, g_target, beta=beta_g)
    if n_cost.shape != g_cost.shape:
        raise ValueError(
            f"N and G KL cost shapes differ: {tuple(n_cost.shape)} vs {tuple(g_cost.shape)}."
        )
    cost = float(g_scale) * g_cost + float(n_weight) * robust_normalize_torch(n_cost)
    return balanced_sinkhorn_torch_from_cost(
        cost,
        temperature=temperature,
        iterations=sinkhorn_iterations,
    )


def native_v7_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    g_source: torch.Tensor,
    g_target: torch.Tensor,
    *,
    beta: float = NATIVE_V7_FEATURE_BETA,
    n_correction_weight: float = NATIVE_V7_N_WEIGHT,
    sinkhorn_iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    """DEPRECATION-CANDIDATE(user-removal): Native-V7 compatibility wrapper."""
    return ng_kl_pij_torch(
        n_source,
        n_target,
        g_source,
        g_target,
        beta_n=beta,
        beta_g=beta,
        g_scale=NATIVE_V7_G_SCALE,
        n_weight=n_correction_weight,
        temperature=1.0,
        sinkhorn_iterations=sinkhorn_iterations,
    )


def canonical_robust_normalize_torch(cost: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of the project's Robust5-95 normalization semantics."""

    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError(f"Torch cost must be a non-empty 2D tensor; got {tuple(cost.shape)}.")
    if not bool(torch.isfinite(cost).all().detach().cpu()):
        raise ValueError("Torch cost must be finite.")
    flat = cost.reshape(-1)
    lower = torch.quantile(flat, CANONICAL_ROBUST_LOWER_QUANTILE)
    upper = torch.quantile(flat, CANONICAL_ROBUST_UPPER_QUANTILE)
    robust_span = upper - lower
    if bool((torch.isfinite(robust_span) & (robust_span > EPS)).detach().cpu()):
        low = lower
        span = robust_span
    else:
        low = torch.min(flat)
        full_span = torch.max(flat) - low
        if bool((torch.isfinite(full_span) & (full_span > EPS)).detach().cpu()):
            span = full_span
        else:
            span = torch.ones((), dtype=cost.dtype, device=cost.device)
    return ((cost - low) / span).clamp(0.0, 1.0)


def _canonical_robust_span_torch(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.numel() == 0:
        raise ValueError(
            f"Torch mixed cost must be a non-empty 2D tensor; got {tuple(values.shape)}."
        )
    if not bool(torch.isfinite(values).all().detach().cpu()):
        raise ValueError("Torch mixed cost must be finite.")
    flat = values.reshape(-1)
    lower = torch.quantile(flat, CANONICAL_ROBUST_LOWER_QUANTILE)
    upper = torch.quantile(flat, CANONICAL_ROBUST_UPPER_QUANTILE)
    robust_span = upper - lower
    if bool((torch.isfinite(robust_span) & (robust_span > EPS)).detach().cpu()):
        return robust_span
    full_span = torch.max(flat) - torch.min(flat)
    if bool((torch.isfinite(full_span) & (full_span > EPS)).detach().cpu()):
        return full_span
    return torch.ones((), dtype=values.dtype, device=values.device)


def build_canonical_ng_cost_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    g_source: torch.Tensor,
    g_target: torch.Tensor,
    *,
    beta_n: float = CANONICAL_FEATURE_BETA_N,
    beta_g: float = CANONICAL_FEATURE_BETA_G,
    alpha_cci: float = CANONICAL_ALPHA_CCI,
) -> torch.Tensor:
    """Differentiable canonical N/G cost with the same equation as NumPy."""

    alpha = float(alpha_cci)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_cci must lie in [0, 1].")
    d_n_raw = pairwise_kl_torch(n_source, n_target, beta=beta_n)
    d_g_raw = pairwise_kl_torch(g_source, g_target, beta=beta_g)
    if d_n_raw.shape != d_g_raw.shape:
        raise ValueError(
            f"N and G KL cost shapes differ: {tuple(d_n_raw.shape)} vs {tuple(d_g_raw.shape)}."
        )
    d_n = canonical_robust_normalize_torch(d_n_raw)
    d_g = canonical_robust_normalize_torch(d_g_raw)
    mixed = (1.0 - alpha) * d_g + alpha * d_n
    return mixed / _canonical_robust_span_torch(mixed)


def canonical_ng_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    g_source: torch.Tensor,
    g_target: torch.Tensor,
    *,
    beta_n: float = CANONICAL_FEATURE_BETA_N,
    beta_g: float = CANONICAL_FEATURE_BETA_G,
    alpha_cci: float = CANONICAL_ALPHA_CCI,
    temperature: float = CANONICAL_TEMPERATURE,
    sinkhorn_iterations: int = CANONICAL_TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    cost = build_canonical_ng_cost_torch(
        n_source,
        n_target,
        g_source,
        g_target,
        beta_n=beta_n,
        beta_g=beta_g,
        alpha_cci=alpha_cci,
    )
    return balanced_sinkhorn_torch_from_cost(
        cost,
        temperature=temperature,
        iterations=sinkhorn_iterations,
    )
