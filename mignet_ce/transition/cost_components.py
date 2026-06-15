from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


def _l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values, dtype=float), where=norms > eps)


def normalize_cost(cost: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D cost matrix, got shape {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    finite_values = arr[finite]
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if max_value - min_value <= eps:
        return np.zeros_like(arr, dtype=float)
    filled = np.where(finite, arr, max_value)
    out = (filled - min_value) / (max_value - min_value)
    return np.clip(out, 0.0, 1.0)


def pairwise_feature_cost(source: np.ndarray, target: np.ndarray, metric: str = "cosine") -> np.ndarray:
    source_arr = _as_2d("source", source)
    target_arr = _as_2d("target", target)
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_arr.shape[1]} != {target_arr.shape[1]}.")
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        return np.zeros((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    if metric == "cosine":
        source_norm = _l2_normalize(source_arr)
        target_norm = _l2_normalize(target_arr)
        return normalize_cost(1.0 - source_norm @ target_norm.T)
    if metric == "euclidean":
        diff = source_arr[:, None, :] - target_arr[None, :, :]
        return normalize_cost(np.linalg.norm(diff, axis=2))
    raise ValueError("metric must be one of {'cosine', 'euclidean'}.")


def pairwise_spatial_cost(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    source_arr = _as_2d("source_coords", source_coords)
    target_arr = _as_2d("target_coords", target_coords)
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(f"Coordinate dimensions differ: {source_arr.shape[1]} != {target_arr.shape[1]}.")
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        return np.zeros((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    diff = source_arr[:, None, :] - target_arr[None, :, :]
    return normalize_cost(np.linalg.norm(diff, axis=2))


def pairwise_scalar_cost(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    source_arr = np.asarray(source_values, dtype=float).reshape(-1)
    target_arr = np.asarray(target_values, dtype=float).reshape(-1)
    return normalize_cost(np.abs(source_arr[:, None] - target_arr[None, :]))


def pairwise_velocity_direction_cost(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_velocity: np.ndarray,
) -> np.ndarray:
    source_arr = _as_2d("source_features", source_features)
    target_arr = _as_2d("target_features", target_features)
    velocity_arr = _as_2d("source_velocity", source_velocity)
    if source_arr.shape[1] != target_arr.shape[1] or source_arr.shape[1] != velocity_arr.shape[1]:
        raise ValueError("Feature and velocity dimensions must match.")
    if source_arr.shape[0] != velocity_arr.shape[0]:
        raise ValueError("source_velocity must have one row per source feature.")
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        return np.zeros((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    deltas = target_arr[None, :, :] - source_arr[:, None, :]
    delta_norms = np.linalg.norm(deltas, axis=2)
    velocity_norms = np.linalg.norm(velocity_arr, axis=1)
    dots = np.einsum("id,ijd->ij", velocity_arr, deltas)
    denom = velocity_norms[:, None] * delta_norms
    cosine = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom > 1e-12)
    return normalize_cost(1.0 - cosine)


def _summary(name: str, cost: np.ndarray, weight: float, enabled: bool) -> dict[str, object]:
    arr = np.asarray(cost, dtype=float)
    finite = np.isfinite(arr)
    warnings: list[str] = []
    if not finite.any():
        warnings.append("all_non_finite")
        min_value = max_value = mean_value = None
    else:
        finite_values = arr[finite]
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        mean_value = float(np.mean(finite_values))
        if max_value - min_value <= 1e-12:
            warnings.append("constant_cost")
    return {
        "component": name,
        "weight": float(weight),
        "enabled": bool(enabled),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "warnings": warnings,
    }


def combine_costs(
    components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    if not components:
        raise ValueError("At least one cost component is required.")
    shape = None
    summaries: dict[str, object] = {}
    combined = None
    total_weight = 0.0
    for name, cost in components.items():
        arr = np.asarray(cost, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Cost component {name!r} must be 2D, got shape {arr.shape}.")
        if shape is None:
            shape = arr.shape
            combined = np.zeros(shape, dtype=float)
        elif arr.shape != shape:
            raise ValueError(f"Cost component {name!r} has shape {arr.shape}; expected {shape}.")
        weight = float(weights.get(name, 0.0))
        if weight < 0:
            raise ValueError(f"Cost weight for {name!r} must be nonnegative.")
        enabled = weight > 0
        summaries[name] = _summary(name, arr, weight, enabled)
        if enabled:
            combined += weight * normalize_cost(arr)
            total_weight += weight
    if combined is None or shape is None:
        raise ValueError("At least one cost component is required.")
    if total_weight <= 0:
        raise ValueError("At least one cost component must have positive weight.")
    combined = combined / total_weight
    summary = {
        "components": summaries,
        "total_weight": float(total_weight),
        "combined": _summary("combined", combined, 1.0, True),
    }
    return combined, summary
