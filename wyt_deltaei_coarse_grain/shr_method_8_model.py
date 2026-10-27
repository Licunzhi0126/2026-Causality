from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as functional
from wyt_deltaei_coarse_grain.macro_builder import adjacency_matmul

class GraphConv(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.lin = nn.Linear(dim, dim)
    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor: return self.lin(adjacency_matmul(adjacency, hidden))

class PCAKNNLinearHeadEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, k: int, layers: int):
        super().__init__(); self.in_proj = nn.Linear(in_dim, hidden_dim); self.layers = nn.ModuleList([GraphConv(hidden_dim) for _ in range(layers)]); self.assignment_head = nn.Linear(hidden_dim, k)
    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, temperature: float, *, return_embed: bool = False):
        hidden = functional.relu(self.in_proj(features))
        for layer in self.layers: hidden = functional.relu(hidden + layer(hidden, adjacency))
        assignment = functional.softmax(self.assignment_head(hidden) / float(temperature), dim=1)
        return (assignment, hidden) if return_embed else assignment
