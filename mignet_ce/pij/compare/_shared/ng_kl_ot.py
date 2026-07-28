from __future__ import annotations

"""Shared NumPy/Torch N/G KL-cost and balanced-OT equations."""

import numpy as np
import torch

from mignet_ce.pij.compare._shared.distances import (
    robust_normalize_cost,
    summarize_dense_cost,
)
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl

EPS = 1e-12
TORCH_SINKHORN_ITERATIONS = 96
NATIVE_V7_FEATURE_BETA = 0.05
NATIVE_V7_G_SCALE = 1.0
NATIVE_V7_N_WEIGHT = 0.25


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
    """Build ``g_scale * KL(G) + n_weight * Robust5_95(KL(N))``."""
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
    """Differentiable N/G KL cost followed by balanced Sinkhorn."""
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
    """Compatibility wrapper for the complete-combined Native-V7 objective."""
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
