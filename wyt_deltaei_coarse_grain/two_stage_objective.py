from __future__ import annotations

"""Losses migrated from WYT train_feature_align_deltaei_v40.py.

The FeatureAlign, DeltaEI, variance, locality, sharpness, prototype and usage
equations retain the v40 defaults and semantics.
"""

import torch
import torch.nn.functional as functional

from wyt_deltaei_coarse_grain.assignment import assignment_entropy
from wyt_deltaei_coarse_grain.macro_builder import adjacency_matmul, row_normalize_torch


EPS = 1e-8


def effective_information(matrix: torch.Tensor) -> torch.Tensor:
    probabilities = row_normalize_torch(matrix)
    average = torch.clamp(probabilities.mean(dim=0), min=EPS)
    output_entropy = -(average * torch.log2(average)).sum()
    safe = torch.clamp(probabilities, min=EPS)
    conditional_entropy = -(safe * torch.log2(safe)).sum(dim=1).mean()
    return output_entropy - conditional_entropy


def sym_kl_feature(
    source: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    source_prob = functional.softmax(source / float(temperature), dim=1).clamp_min(EPS)
    target_prob = functional.softmax(target / float(temperature), dim=1).clamp_min(EPS)
    source_target = (
        source_prob * (torch.log(source_prob) - torch.log(target_prob))
    ).sum(dim=1).mean()
    target_source = (
        target_prob * (torch.log(target_prob) - torch.log(source_prob))
    ).sum(dim=1).mean()
    return 0.5 * (source_target + target_source)


def variance_loss(hidden: torch.Tensor, target_std: float) -> torch.Tensor:
    std = torch.sqrt(hidden.var(dim=0) + EPS)
    return functional.relu(float(target_std) - std).mean()


def local_smoothness(assignment: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    return (
        assignment - adjacency_matmul(adjacency, assignment)
    ).pow(2).sum(dim=1).mean()


def prototype_repulsion(prototypes: torch.Tensor, max_cosine: float) -> torch.Tensor:
    normalized = functional.normalize(prototypes, dim=1, eps=EPS)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(
        similarity.shape[0],
        dtype=torch.bool,
        device=similarity.device,
    )
    return functional.relu(similarity[mask] - float(max_cosine)).pow(2).mean()


def usage_loss(
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    min_usage: float,
) -> torch.Tensor:
    """Prevent dead prototypes without imposing uniform cluster sizes."""
    k = assignment_t.shape[1]
    if assignment_tp.shape[1] != k:
        raise ValueError("Assignments must have the same prototype count.")
    usage_t = assignment_t.mean(dim=0)
    usage_tp = assignment_tp.mean(dim=0)
    floor = assignment_t.new_tensor(float(min_usage))
    loss_t = torch.relu(floor - usage_t).pow(2).mean()
    loss_tp = torch.relu(floor - usage_tp).pow(2).mean()
    return 0.5 * (loss_t + loss_tp)


def effective_prototype_count(assignment: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Entropy effective number of occupied prototypes (Keff)."""
    usage = assignment.mean(dim=0).clamp_min(eps)
    usage = usage / usage.sum().clamp_min(eps)
    entropy_bits = -(usage * torch.log2(usage)).sum()
    return torch.pow(assignment.new_tensor(2.0), entropy_bits)


def keff_floor_loss(
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    keff_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize collapsed effective diversity without uniformizing clusters."""
    if keff_min <= 0.0:
        zero = assignment_t.new_zeros(())
        return zero, effective_prototype_count(assignment_t), effective_prototype_count(assignment_tp)
    keff_t = effective_prototype_count(assignment_t)
    keff_tp = effective_prototype_count(assignment_tp)
    floor = assignment_t.new_tensor(float(keff_min))
    loss_t = torch.relu((floor - keff_t) / floor).pow(2)
    loss_tp = torch.relu((floor - keff_tp) / floor).pow(2)
    return 0.5 * (loss_t + loss_tp), keff_t, keff_tp


def row_js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence for corresponding probability rows."""
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("p and q must be two-dimensional tensors with matching shapes.")
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_p = (p * (torch.log(p) - torch.log(midpoint))).sum(dim=1)
    kl_q = (q * (torch.log(q) - torch.log(midpoint))).sum(dim=1)
    return 0.5 * (kl_p + kl_q)


def dynamical_closure_loss(
    micro_pij: torch.Tensor,
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    macro_pij: torch.Tensor,
    *,
    future: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match micro-propagated and macro-predicted future memberships."""
    r_micro = micro_pij @ assignment_tp if future is None else future
    r_macro = assignment_t @ macro_pij
    return row_js_divergence(r_micro, r_macro).mean()


def within_macro_dynamics_loss(
    micro_pij: torch.Tensor,
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    eps: float = EPS,
    *,
    future: torch.Tensor | None = None,
    future_centers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Make future macro dynamics homogeneous within each soft macro state."""
    if (future is None) != (future_centers is None):
        raise ValueError("future and future_centers must be supplied together.")
    if future is None:
        future, future_centers, _ = future_dynamics_centers(
            micro_pij, assignment_t, assignment_tp, eps=eps
        )
    pairwise_js = row_js_divergence(
        future[:, None, :].expand(-1, assignment_t.shape[1], -1).reshape(-1, future.shape[1]),
        future_centers[None, :, :].expand(future.shape[0], -1, -1).reshape(-1, future.shape[1]),
        eps=eps,
    ).reshape_as(assignment_t)
    return (assignment_t * pairwise_js).sum() / assignment_t.sum().clamp_min(eps)


def future_dynamics_centers(
    micro_pij: torch.Tensor,
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return micro future distributions, soft macro centers and source masses."""
    future = micro_pij @ assignment_tp
    future = future / future.sum(dim=1, keepdim=True).clamp_min(eps)
    mass = assignment_t.mean(dim=0).clamp_min(eps)
    centers = (assignment_t.transpose(0, 1) @ future) / (
        assignment_t.sum(dim=0).clamp_min(eps)[:, None]
    )
    centers = centers / centers.sum(dim=1, keepdim=True).clamp_min(eps)
    return future, centers, mass


def inter_macro_dynamics_loss(
    future_centers: torch.Tensor,
    source_usage: torch.Tensor,
    margin: float,
    min_usage_for_inter: float,
    eps: float = EPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Separate occupied macro future centers with a JS-margin objective."""
    active = source_usage >= float(min_usage_for_inter)
    active_centers = future_centers[active]
    zero = future_centers.new_zeros(())
    if active_centers.shape[0] < 2:
        return zero, {
            "mean": zero,
            "min": zero,
            "median": zero,
            "active_count": active.sum().to(dtype=future_centers.dtype),
        }
    pairwise = row_js_divergence(
        active_centers[:, None, :]
        .expand(-1, active_centers.shape[0], -1)
        .reshape(-1, active_centers.shape[1]),
        active_centers[None, :, :]
        .expand(active_centers.shape[0], -1, -1)
        .reshape(-1, active_centers.shape[1]),
        eps=eps,
    ).reshape(active_centers.shape[0], active_centers.shape[0])
    values = pairwise[torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)]
    return torch.relu(float(margin) - values).pow(2).mean(), {
        "mean": values.mean(),
        "min": values.min(),
        "median": values.median(),
        "active_count": active.sum().to(dtype=future_centers.dtype),
    }


def information_closure_metrics(
    micro_pij: torch.Tensor,
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    eps: float = EPS,
    *,
    future: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable counterpart of downstream's uniform-spot information budget."""
    if future is None:
        future, _, _ = future_dynamics_centers(
            micro_pij, assignment_t, assignment_tp, eps=eps
        )
    source = assignment_t / assignment_t.sum(dim=1, keepdim=True).clamp_min(eps)
    joint = (source.transpose(0, 1) @ future) / float(source.shape[0])
    joint = joint / joint.sum().clamp_min(eps)
    source_mass = joint.sum(dim=1)
    target_mass = joint.sum(dim=0)
    independent = source_mass[:, None] * target_mass[None, :]
    retained = torch.where(
        joint > 0,
        joint * torch.log2(joint.clamp_min(eps) / independent.clamp_min(eps)),
        torch.zeros_like(joint),
    ).sum()
    target_average = future.mean(dim=0, keepdim=True)
    available = (
        future
        * torch.log2(future.clamp_min(eps) / target_average.clamp_min(eps))
    ).sum(dim=1).mean()
    leakage = available - retained
    quality = retained / available.clamp_min(eps)
    return {
        "I_available": available,
        "I_retained": retained,
        "closure_leakage": leakage,
        "closure_quality": quality,
    }


def ei_floor_constraint(
    delta_ei: torch.Tensor,
    reference: float,
    retain_ratio: float,
) -> torch.Tensor:
    floor = delta_ei.new_tensor(float(reference) * float(retain_ratio))
    return torch.relu(floor - delta_ei).pow(2)


def normalized_retained_information_loss(
    retained: torch.Tensor,
    available_reference: float,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximize retained information against an immutable Stage-1 denominator."""
    reference = retained.new_tensor(float(available_reference)).clamp_min(eps)
    normalized = retained / reference
    return -normalized, normalized


def retained_information_floor_constraint(
    retained: torch.Tensor,
    retained_reference: float,
    retain_ratio: float,
    eps: float = EPS,
) -> torch.Tensor:
    """Keep a configured fraction of frozen Stage-1 retained information."""
    reference = retained.new_tensor(float(retained_reference)).clamp_min(eps)
    floor = reference * float(retain_ratio)
    return torch.relu((floor - retained) / reference).pow(2)


def sharpness_loss(
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        assignment_entropy(assignment_t) + assignment_entropy(assignment_tp)
    )

