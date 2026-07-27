from __future__ import annotations

from typing import Mapping

import numpy as np

from mignet_ce.pij.feature_versions.distances import matrix_summary


def robust_normalize_cost(
    cost: np.ndarray,
    *,
    quantiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[np.ndarray, dict[str, object]]:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Cost must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Cost contains non-finite values.")
    if np.any(arr < -1e-12):
        raise ValueError("Cost contains negative values.")
    arr = np.maximum(arr, 0.0)
    q_low, q_high = map(float, quantiles)
    if not 0.0 <= q_low < q_high <= 100.0:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 100.")
    if arr.size == 0:
        return arr.copy(), {
            "normalization": f"quantile_{q_low:g}_{q_high:g}",
            "constant_block": True,
            "q_low": None,
            "q_high": None,
        }
    low, high = np.percentile(arr, [q_low, q_high])
    constant = bool(high - low <= 1e-12)
    if constant:
        normalized = np.zeros_like(arr)
    else:
        normalized = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return normalized, {
        "normalization": f"quantile_{q_low:g}_{q_high:g}",
        "constant_block": constant,
        "q_low": float(low),
        "q_high": float(high),
        "raw": matrix_summary(arr),
        "normalized": matrix_summary(normalized),
    }


def fuse_cost_blocks(
    costs: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    quantiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    if set(costs) != set(weights):
        raise ValueError(f"Cost blocks and weights differ: {sorted(costs)} vs {sorted(weights)}.")
    numeric_weights = {name: float(value) for name, value in weights.items()}
    if any(value < 0.0 for value in numeric_weights.values()):
        raise ValueError("Fusion weights must be nonnegative.")
    if abs(sum(numeric_weights.values()) - 1.0) > 1e-9:
        raise ValueError("Fusion weights must sum to 1.")
    shapes = {name: np.asarray(cost).shape for name, cost in costs.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Cost block shapes differ: {shapes}.")
    if not costs:
        raise ValueError("At least one cost block is required.")

    normalized: dict[str, np.ndarray] = {}
    block_metadata: dict[str, object] = {}
    fused: np.ndarray | None = None
    for name in weights:
        normalized_cost, metadata = robust_normalize_cost(costs[name], quantiles=quantiles)
        normalized[name] = normalized_cost
        block_metadata[name] = {**metadata, "weight": numeric_weights[name]}
        contribution = numeric_weights[name] * normalized_cost
        fused = contribution.copy() if fused is None else fused + contribution
    assert fused is not None
    if not np.all(np.isfinite(fused)) or np.any(fused < -1e-12):
        raise ValueError("Fused cost is invalid.")
    fused = np.maximum(fused, 0.0)
    return fused, {
        "weights": numeric_weights,
        "weight_sum": float(sum(numeric_weights.values())),
        "enabled_blocks": list(weights),
        "block_diagnostics": block_metadata,
        "fused": matrix_summary(fused),
        "weight_redistribution": False,
    }, normalized
