from __future__ import annotations

"""Neural modules migrated from WYT train_feature_align_deltaei_v40.py."""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from wyt_deltaei_coarse_grain.macro_builder import adjacency_matmul, row_normalize_torch


EPS = 1e-12


class GraphConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.lin(adjacency_matmul(adjacency, hidden))


class PrototypeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, k: int, layers: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConv(hidden_dim) for _ in range(layers)])
        self.prototypes = nn.Parameter(torch.randn(k, hidden_dim) * 0.02)

    def embed(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = functional.relu(self.in_proj(features))
        for layer in self.layers:
            hidden = functional.relu(hidden + layer(hidden, adjacency))
        return hidden

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        temperature: float,
        *,
        return_embed: bool = False,
    ):
        hidden = self.embed(features, adjacency)
        logits = (
            functional.normalize(hidden, dim=1, eps=EPS)
            @ functional.normalize(self.prototypes, dim=1, eps=EPS).T
        )
        assignment = functional.softmax(logits / float(temperature), dim=1)
        if return_embed:
            return assignment, hidden
        return assignment


class MacroFeatureNet(nn.Module):
    def __init__(self, k: int, hidden_dim: int, mid_dim: int, layers: int):
        super().__init__()
        self.in_proj = nn.Linear(k + 1, hidden_dim)
        self.layers = nn.ModuleList([GraphConv(hidden_dim) for _ in range(layers)])
        self.out = nn.Linear(hidden_dim, mid_dim)

    def forward(self, macro_network: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        features = torch.cat([macro_network, mass.unsqueeze(1)], dim=1)
        adjacency = row_normalize_torch(
            macro_network
            + torch.eye(
                macro_network.shape[0],
                dtype=macro_network.dtype,
                device=macro_network.device,
            )
        )
        hidden = functional.relu(self.in_proj(features))
        for layer in self.layers:
            hidden = functional.relu(hidden + layer(hidden, adjacency))
        return self.out(hidden)
