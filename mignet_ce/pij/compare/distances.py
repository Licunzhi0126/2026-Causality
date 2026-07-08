from __future__ import annotations

from typing import Dict

import numpy as np

from mignet_ce.utils.matrix import safe_row_normalize


def pairwise_cosine_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-12,
    block_size: int = 1024,
) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {source_arr.shape} and {target_arr.shape}.")
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_arr.shape[1]} vs {target_arr.shape[1]}.")
    out = np.empty((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        return out

    target_norm = np.linalg.norm(target_arr, axis=1)
    target_safe = np.divide(target_arr, target_norm[:, None], out=np.zeros_like(target_arr), where=target_norm[:, None] > eps)
    for start in range(0, source_arr.shape[0], max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), source_arr.shape[0])
        block = source_arr[start:stop]
        block_norm = np.linalg.norm(block, axis=1)
        block_safe = np.divide(block, block_norm[:, None], out=np.zeros_like(block), where=block_norm[:, None] > eps)
        sim = block_safe @ target_safe.T
        out[start:stop] = 1.0 - np.clip(sim, -1.0, 1.0)
    return np.maximum(out, 0.0)


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


def kernel_from_cost(cost: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be positive.")
    arr = np.asarray(cost, dtype=float)
    return np.exp(-np.nan_to_num(arr, nan=np.inf, posinf=np.inf, neginf=0.0) / float(tau))


def row_normalized_kernel_from_cost(cost: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    kernel = kernel_from_cost(cost, tau=tau)
    return kernel, safe_row_normalize(kernel)


def matrix_summary(matrix: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary: Dict[str, object] = {
        "shape": list(arr.shape),
        "finite_values": int(finite.size),
    }
    if finite.size:
        summary.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
            }
        )
    return summary
