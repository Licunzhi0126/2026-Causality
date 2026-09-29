from __future__ import annotations

"""Differentiable pooling migrated from WYT train_feature_align_deltaei_v40.py.

Adaptation: supports sparse within-time networks without changing ``S.T P S``.
"""

from collections.abc import Mapping

import torch


EPS = 1e-12


def row_normalize_torch(matrix: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(matrix, min=0.0)
    row_sum = values.sum(dim=1, keepdim=True)
    zero = row_sum[:, 0] <= EPS
    if torch.any(zero):
        values = values.clone()
        values[zero] = 1.0 / values.shape[1]
        row_sum = values.sum(dim=1, keepdim=True)
    return values / (row_sum + EPS)


def adjacency_matmul(adjacency: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if adjacency.layout != torch.strided:
        return torch.sparse.mm(adjacency, values)
    return adjacency @ values


def macro_matrix(network: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    projected = adjacency_matmul(network, assignment)
    return row_normalize_torch(assignment.T @ projected)


def pool_to_macro(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    mass = assignment.sum(dim=0) + EPS
    return (assignment.T @ values) / mass.unsqueeze(1)


def pool_feature_blocks(
    blocks: Mapping[str, torch.Tensor],
    assignment: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: pool_to_macro(values, assignment)
        for name, values in blocks.items()
    }
