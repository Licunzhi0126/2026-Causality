from __future__ import annotations

import numpy as np


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if values.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {values.shape}")
    if values.shape[1] == 0:
        raise ValueError("matrix must contain at least one target column")
    sums = values.sum(axis=1, keepdims=True)
    zero = sums[:, 0] <= eps
    if np.any(zero):
        values[zero, :] = 1.0 / values.shape[1]
        sums = values.sum(axis=1, keepdims=True)
    return values / np.maximum(sums, eps)


def entropy(probabilities: np.ndarray, axis: int | None = None, eps: float = 1e-12) -> np.ndarray:
    p = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    return -np.sum(p * np.log2(p + eps), axis=axis)
