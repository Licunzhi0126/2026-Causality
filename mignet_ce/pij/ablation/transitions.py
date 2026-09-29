from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.distances import summarize_dense_cost
from mignet_ce.pij.compare._shared.log_balanced_ot import balance_cost_log_sinkhorn
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_from_cost_numpy
from mignet_ce.utils.matrix import safe_row_normalize

from .config import AblationConfig


@dataclass(frozen=True)
class AblationTransitionResult:
    raw_matrix: np.ndarray
    pij: np.ndarray
    metadata: dict[str, object]


def _balance_uniform_sinkhorn(
    kernel: np.ndarray,
    *,
    max_iter: int,
    tolerance: float,
    check_every: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """DEPRECATION-CANDIDATE(user-removal): superseded by production Sinkhorn."""
    values = np.asarray(kernel, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Sinkhorn kernel must be non-empty 2D; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Sinkhorn kernel must be finite and nonnegative.")
    if np.any(values.sum(axis=1) <= 0.0) or np.any(values.sum(axis=0) <= 0.0):
        raise RuntimeError("Sinkhorn kernel lost positive row/column support.")
    n, m = values.shape
    a = np.full(n, 1.0 / n, dtype=float)
    b = np.full(m, 1.0 / m, dtype=float)
    u = np.ones(n, dtype=float)
    v = np.ones(m, dtype=float)
    residual = float("inf")
    for iteration in range(1, int(max_iter) + 1):
        kv = values @ v
        if np.any(kv <= 0.0) or not np.isfinite(kv).all():
            raise RuntimeError("Sinkhorn source scaling lost support.")
        u = a / kv
        ktu = values.T @ u
        if np.any(ktu <= 0.0) or not np.isfinite(ktu).all():
            raise RuntimeError("Sinkhorn target scaling lost support.")
        v = b / ktu
        if iteration % int(check_every) == 0 or iteration == int(max_iter):
            source = u * (values @ v)
            target = v * (values.T @ u)
            residual = float(max(np.max(np.abs(source - a)), np.max(np.abs(target - b))))
            if residual <= float(tolerance):
                joint = (u[:, None] * values) * v[None, :]
                return joint, safe_row_normalize(joint), {
                    "converged": True,
                    "iterations": iteration,
                    "max_absolute_marginal_residual": residual,
                    "source_marginal_policy": "uniform",
                    "target_marginal_policy": "uniform",
                }
    raise RuntimeError(
        f"Controlled-ablation Sinkhorn did not converge after {max_iter} iterations; residual={residual}."
    )


def transition_from_cost(cost: np.ndarray, *, use_ot: bool, cfg: AblationConfig) -> AblationTransitionResult:
    cfg.validate()
    cost = np.asarray(cost, dtype=float)
    kernel, row_pij = row_normalized_kernel_from_cost(cost, tau=float(cfg.temperature))
    if not use_ot:
        return AblationTransitionResult(
            raw_matrix=kernel,
            pij=row_pij,
            metadata={
                "transition": "row_normalized_exp_kernel",
                "ot_enabled": False,
                "temperature": float(cfg.temperature),
                "kernel": summarize_dense_cost(kernel),
            },
        )
    joint, pij, production_metadata = canonical_ng_pij_from_cost_numpy(
        cost,
        temperature=float(cfg.temperature),
    )
    return AblationTransitionResult(
        raw_matrix=joint,
        pij=pij,
        metadata={
            "transition": "balanced_uniform_sinkhorn_on_same_exp_kernel",
            "ot_enabled": True,
            "temperature": float(cfg.temperature),
            "kernel_before_ot": summarize_dense_cost(kernel),
            "production_transition": production_metadata,
            "sinkhorn": production_metadata["sinkhorn"],
        },
    )
