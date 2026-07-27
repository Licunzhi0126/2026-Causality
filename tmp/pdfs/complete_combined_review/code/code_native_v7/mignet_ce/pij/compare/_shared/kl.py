from __future__ import annotations

import numpy as np

from mignet_ce.utils.matrix import safe_row_normalize


def row_softmax_features(features: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    scaled = arr / float(beta)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(scaled, -700.0, 700.0))
    return safe_row_normalize(exp_values)


def pairwise_feature_kl(
    source: np.ndarray,
    target: np.ndarray,
    *,
    beta: float = 1.0,
    eps: float = 1e-12,
    block_size: int = 512,
) -> np.ndarray:
    source_prob = np.clip(row_softmax_features(source, beta=beta), eps, 1.0)
    target_prob = np.clip(row_softmax_features(target, beta=beta), eps, 1.0)
    if source_prob.shape[1] != target_prob.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_prob.shape[1]} vs {target_prob.shape[1]}.")
    log_source = np.log(source_prob)
    log_target = np.log(target_prob)
    source_entropy_term = np.sum(source_prob * log_source, axis=1, keepdims=True)
    out = np.empty((source_prob.shape[0], target_prob.shape[0]), dtype=float)
    for start in range(0, source_prob.shape[0], max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), source_prob.shape[0])
        out[start:stop] = source_entropy_term[start:stop] - source_prob[start:stop] @ log_target.T
    return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
