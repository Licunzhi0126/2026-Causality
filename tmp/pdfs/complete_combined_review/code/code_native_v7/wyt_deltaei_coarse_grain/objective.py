from __future__ import annotations

"""Losses migrated from WYT train_feature_align_deltaei_v40.py.

The FeatureAlign, DeltaEI, variance, locality, sharpness, prototype and usage
equations retain the v40 defaults and semantics.
"""

import torch
import torch.nn.functional as functional

from wyt_deltaei_coarse_grain.assignment import assignment_entropy, usage_stats
from wyt_deltaei_coarse_grain.macro_builder import adjacency_matmul, row_normalize_torch


EPS = 1e-12


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


def usage_penalty(
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
    min_frac: float,
    max_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = assignment_t.shape[1]
    minimum = float(min_frac) / k
    maximum = float(max_frac) / k
    usage_t, _ = usage_stats(assignment_t)
    usage_tp, _ = usage_stats(assignment_tp)
    min_penalty = 0.5 * (
        functional.relu(minimum - usage_t).pow(2).sum()
        + functional.relu(minimum - usage_tp).pow(2).sum()
    )
    max_penalty = 0.5 * (
        functional.relu(usage_t - maximum).pow(2).sum()
        + functional.relu(usage_tp - maximum).pow(2).sum()
    )
    return min_penalty, max_penalty


def sharpness_loss(
    assignment_t: torch.Tensor,
    assignment_tp: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        assignment_entropy(assignment_t) + assignment_entropy(assignment_tp)
    )
