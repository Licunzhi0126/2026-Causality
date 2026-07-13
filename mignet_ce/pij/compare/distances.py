from __future__ import annotations

from typing import Dict

import numpy as np

from mignet_ce.pij.compare.cosine import pairwise_cosine_distance


def summarize_dense_cost(cost: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(cost, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary: Dict[str, object] = {
        "shape": list(arr.shape),
        "values": int(arr.size),
        "finite_values": int(finite.size),
        "nonfinite_count": int(arr.size - finite.size),
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


def pairwise_euclidean_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
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

    target_sq = np.sum(target_arr * target_arr, axis=1)[None, :]
    step = max(1, int(block_size))
    for start in range(0, source_arr.shape[0], step):
        stop = min(start + step, source_arr.shape[0])
        block = source_arr[start:stop]
        block_sq = np.sum(block * block, axis=1)[:, None]
        squared = block_sq + target_sq - 2.0 * (block @ target_arr.T)
        np.maximum(squared, 0.0, out=squared)
        np.sqrt(squared, out=out[start:stop])
    return out


def pairwise_vector_distance(
    source: np.ndarray,
    target: np.ndarray,
    metric: str,
    *,
    eps: float = 1e-12,
    block_size: int = 1024,
) -> np.ndarray:
    if metric == "cosine":
        return pairwise_cosine_distance(source, target, eps=eps, block_size=block_size)
    if metric == "euclidean":
        return pairwise_euclidean_distance(source, target, block_size=block_size)
    raise ValueError("metric must be one of ['cosine', 'euclidean'].")


def pairwise_scalar_absolute_distance(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError(f"Expected 2D scalar feature matrices, got {source_arr.shape} and {target_arr.shape}.")
    if source_arr.shape[1] != 1 or target_arr.shape[1] != 1:
        raise ValueError(
            "Scalar absolute distance requires exactly one feature column; "
            f"got {source_arr.shape} and {target_arr.shape}."
        )
    return np.abs(source_arr[:, 0, None] - target_arr[None, :, 0])


def robust_normalize_cost(
    cost: np.ndarray,
    *,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    copy: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Expected 0 <= lower_percentile < upper_percentile <= 100.")
    source = np.asarray(cost, dtype=float)
    arr = source.copy() if copy else source
    raw_summary = summarize_dense_cost(arr)
    metadata: dict[str, object] = {
        "raw_summary": raw_summary,
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "q_low": None,
        "q_high": None,
        "fallback_min": None,
        "fallback_max": None,
        "normalization_mode": "empty" if arr.size == 0 else "all_zero_degenerate",
        "nonfinite_count": int(raw_summary["nonfinite_count"]),
    }
    if arr.size == 0:
        metadata["normalized_summary"] = summarize_dense_cost(arr)
        return arr, metadata

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        arr[...] = 0.0
        metadata["normalized_summary"] = summarize_dense_cost(arr)
        return arr, metadata

    q_low = float(np.percentile(finite, lower_percentile))
    q_high = float(np.percentile(finite, upper_percentile))
    metadata["q_low"] = q_low
    metadata["q_high"] = q_high
    low = q_low
    high = q_high
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
        metadata["fallback_min"] = low
        metadata["fallback_max"] = high
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            arr[...] = 0.0
            metadata["normalized_summary"] = summarize_dense_cost(arr)
            return arr, metadata
        metadata["normalization_mode"] = "minmax_fallback"
    else:
        metadata["normalization_mode"] = "quantile_5_95"

    arr -= low
    arr /= high - low
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(arr, 0.0, 1.0, out=arr)
    metadata["normalized_summary"] = summarize_dense_cost(arr)
    return arr, metadata
