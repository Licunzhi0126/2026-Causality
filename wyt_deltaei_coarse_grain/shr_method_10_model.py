from __future__ import annotations

"""Method 10's independent CCI and cross-time attention modules."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


EPS = 1e-12


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains NaN or Inf.")


def stable_cross_time_bias(prior: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a row-stochastic prior to a bounded additive attention bias."""
    _finite("cross-time prior", prior)
    if prior.ndim != 2 or bool((prior < 0).any()):
        raise ValueError("cross-time prior must be a non-negative 2D tensor.")
    bias = prior.clamp_min(eps).log()
    bias = bias - bias.mean(dim=1, keepdim=True)
    scale = torch.quantile(bias.abs().flatten(), 0.95).clamp_min(1e-6)
    return (bias / scale).clamp(-5.0, 5.0)


def reverse_row_normalized_prior(forward: torch.Tensor) -> torch.Tensor:
    """Return row-normalized ``forward.T``, using uniform weights for zero rows."""
    reverse = forward.transpose(0, 1)
    totals = reverse.sum(dim=1, keepdim=True)
    uniform = torch.full_like(reverse, 1.0 / reverse.shape[1])
    return torch.where(totals > EPS, reverse / totals.clamp_min(EPS), uniform)


class Method10GlobalAttentionEncoder(nn.Module):
    """A standalone copy of method 9's two-layer CCI-biased self-attention."""

    def __init__(self, in_dim: int, hidden_dim: int, k: int, layers: int = 2, heads: int = 4,
                 ffn_dim: int | None = None, dropout: float = 0.0, beta_init: float = 0.1):
        super().__init__()
        if layers != 2 or heads <= 0 or hidden_dim % heads:
            raise ValueError("Method 10 requires two layers and hidden_dim divisible by heads.")
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.heads, self.head_dim = heads, hidden_dim // heads
        self.qkv = nn.ModuleList(nn.Linear(hidden_dim, 3 * hidden_dim) for _ in range(layers))
        self.out = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.norm1 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        width = ffn_dim or 4 * hidden_dim
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, hidden_dim)) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.full((layers, heads), float(beta_init)))
        self.assignment_head = nn.Linear(hidden_dim, k)
        self.last_attention_entropy: torch.Tensor | None = None

    @staticmethod
    def _bias(relation: torch.Tensor) -> torch.Tensor:
        relation = relation.to_dense() if relation.is_sparse else relation
        _finite("CCI relation", relation)
        compressed = torch.log1p(relation.clamp_min(0)) if bool((relation >= 0).all()) else relation.sign() * torch.log1p(relation.abs())
        return (compressed / torch.quantile(compressed.abs().flatten(), .95).clamp_min(1e-6)).clamp(-5., 5.)

    def embed(self, features: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        hidden, bias, entropies = functional.gelu(self.in_proj(features)), self._bias(relation), []
        for qkv, out, norm1, norm2, ffn, beta in zip(self.qkv, self.out, self.norm1, self.norm2, self.ffn, self.beta):
            query, key, value = qkv(hidden).chunk(3, dim=-1)
            query = query.view(-1, self.heads, self.head_dim).transpose(0, 1)
            key = key.view(-1, self.heads, self.head_dim).transpose(0, 1)
            value = value.view(-1, self.heads, self.head_dim).transpose(0, 1)
            logits = query @ key.transpose(-2, -1) / (self.head_dim ** .5) + beta[:, None, None] * bias[None]
            weights = functional.softmax(logits, dim=-1)
            entropies.append(-(weights * weights.clamp_min(EPS).log()).sum(dim=-1).mean())
            attended = (weights @ value).transpose(0, 1).reshape_as(hidden)
            hidden = norm1(hidden + self.dropout(out(attended)))
            hidden = norm2(hidden + self.dropout(ffn(hidden)))
        self.last_attention_entropy = torch.stack(entropies).mean()
        return hidden


class CrossTimeAttention(nn.Module):
    """Full multi-head cross-attention with an additive, learned-strength prior."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float, beta_init: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by cross-time attention heads.")
        self.heads, self.head_dim = heads, hidden_dim // heads
        self.query, self.key, self.value = (nn.Linear(hidden_dim, hidden_dim) for _ in range(3))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.full((heads,), float(beta_init)))
        self.last_attention_entropy: torch.Tensor | None = None

    def forward(self, query_hidden: torch.Tensor, key_value_hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n_query, n_key = query_hidden.shape[0], key_value_hidden.shape[0]
        if bias.shape != (n_query, n_key):
            raise ValueError(f"cross-time bias shape {tuple(bias.shape)} does not match {(n_query, n_key)}.")
        query = self.query(query_hidden).view(n_query, self.heads, self.head_dim).transpose(0, 1)
        key = self.key(key_value_hidden).view(n_key, self.heads, self.head_dim).transpose(0, 1)
        value = self.value(key_value_hidden).view(n_key, self.heads, self.head_dim).transpose(0, 1)
        logits = query @ key.transpose(-2, -1) / (self.head_dim ** .5) + self.beta[:, None, None] * bias[None]
        weights = functional.softmax(logits, dim=-1)
        self.last_attention_entropy = -(weights * weights.clamp_min(EPS).log()).sum(dim=-1).mean()
        return self.out((self.dropout(weights) @ value).transpose(0, 1).reshape(n_query, -1))


class GatedCrossTimeFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, gate_bias_init: float):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_gate_mean: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((hidden, message), dim=-1)))
        self.last_gate_mean = gate.mean()
        return self.norm(hidden + gate * self.dropout(self.message(message)))


class BidirectionalCrossTimeAttentionModel(nn.Module):
    """Method 10 joint model.  No parameters are shared between time points."""

    def __init__(self, in_dim_t: int, in_dim_tp: int, hidden_dim: int, k: int, *, layers: int = 2,
                 heads: int = 4, ffn_dim: int | None = None, dropout: float = 0., attention_beta_init: float = .1,
                 cross_time_beta_init: float = .1, cross_time_gate_bias_init: float = -2.):
        super().__init__()
        encoder_args = dict(hidden_dim=hidden_dim, k=k, layers=layers, heads=heads, ffn_dim=ffn_dim, dropout=dropout, beta_init=attention_beta_init)
        self.encoder_t = Method10GlobalAttentionEncoder(in_dim_t, **encoder_args)
        self.encoder_tp = Method10GlobalAttentionEncoder(in_dim_tp, **encoder_args)
        self.cross_t_from_tp = CrossTimeAttention(hidden_dim, heads, dropout, cross_time_beta_init)
        self.cross_tp_from_t = CrossTimeAttention(hidden_dim, heads, dropout, cross_time_beta_init)
        self.fusion_t = GatedCrossTimeFusion(hidden_dim, dropout, cross_time_gate_bias_init)
        self.fusion_tp = GatedCrossTimeFusion(hidden_dim, dropout, cross_time_gate_bias_init)

    def forward(self, features_t: torch.Tensor, relation_t: torch.Tensor, features_tp: torch.Tensor,
                relation_tp: torch.Tensor, micro_pij: torch.Tensor, temperature: float) -> dict[str, torch.Tensor]:
        hidden_t, hidden_tp = self.encoder_t.embed(features_t, relation_t), self.encoder_tp.embed(features_tp, relation_tp)
        forward_bias = stable_cross_time_bias(micro_pij)
        reverse_bias = stable_cross_time_bias(reverse_row_normalized_prior(micro_pij))
        cross_t = self.cross_t_from_tp(hidden_t, hidden_tp, forward_bias)
        cross_tp = self.cross_tp_from_t(hidden_tp, hidden_t, reverse_bias)
        hidden_t, hidden_tp = self.fusion_t(hidden_t, cross_t), self.fusion_tp(hidden_tp, cross_tp)
        assignment_t = functional.softmax(self.encoder_t.assignment_head(hidden_t) / float(temperature), dim=1)
        assignment_tp = functional.softmax(self.encoder_tp.assignment_head(hidden_tp) / float(temperature), dim=1)
        return {"assignment_t": assignment_t, "assignment_tp": assignment_tp, "hidden_t": hidden_t, "hidden_tp": hidden_tp,
                "cross_attention_entropy_t_from_tp": self.cross_t_from_tp.last_attention_entropy,
                "cross_attention_entropy_tp_from_t": self.cross_tp_from_t.last_attention_entropy,
                "cross_gate_mean_t": self.fusion_t.last_gate_mean, "cross_gate_mean_tp": self.fusion_tp.last_gate_mean,
                "self_attention_entropy_t": self.encoder_t.last_attention_entropy, "self_attention_entropy_tp": self.encoder_tp.last_attention_entropy}
