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
