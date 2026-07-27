from __future__ import annotations

"""Differentiable Native-V7 PIJ equations for complete coarse graining."""

import numpy as np
import torch

from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    N_CORRECTION_WEIGHT,
)

EPS = 1e-12
TORCH_SINKHORN_ITERATIONS = 96


def pairwise_kl(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    source_prob = torch.softmax(source / float(beta), dim=1).clamp_min(EPS)
    target_prob = torch.softmax(target / float(beta), dim=1).clamp_min(EPS)
    source_entropy = (source_prob * torch.log(source_prob)).sum(dim=1, keepdim=True)
    return (source_entropy - source_prob @ torch.log(target_prob).T).clamp_min(0.0)


def robust_normalize(cost: torch.Tensor) -> torch.Tensor:
    flat = cost.reshape(-1)
    lower = torch.quantile(flat, 0.05)
    upper = torch.quantile(flat, 0.95)
    scale = (upper - lower).clamp_min(EPS)
    return ((cost - lower) / scale).clamp(0.0, 1.0)


def balanced_sinkhorn_from_cost(
    cost: torch.Tensor,
    *,
    iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError(f"Torch Sinkhorn cost must be non-empty 2D; got {tuple(cost.shape)}.")
    if int(iterations) <= 0:
        raise ValueError("Torch Sinkhorn iterations must be positive.")
    source_count, target_count = cost.shape
    log_kernel = -cost
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


def native_v7_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    g_source: torch.Tensor,
    g_target: torch.Tensor,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
    sinkhorn_iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    """Raw true-G KL plus bounded N correction, followed by balanced Sinkhorn."""
    n_cost = pairwise_kl(n_source, n_target, beta=beta)
    g_cost = pairwise_kl(g_source, g_target, beta=beta)
    cost = g_cost + float(n_correction_weight) * robust_normalize(n_cost)
    return balanced_sinkhorn_from_cost(cost, iterations=sinkhorn_iterations)
