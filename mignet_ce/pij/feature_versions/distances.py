from __future__ import annotations

from typing import Callable

import numpy as np


def _as_finite_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_arr = _as_finite_2d(source, name="source")
    target_arr = _as_finite_2d(target, name="target")
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_arr.shape[1]} vs {target_arr.shape[1]}.")
    return source_arr, target_arr


def row_softmax(values: np.ndarray, *, beta: float) -> np.ndarray:
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    arr = _as_finite_2d(values, name="values")
    if arr.shape[1] == 0:
        return arr.copy()
    scaled = arr / float(beta)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(scaled, -700.0, 700.0))
    totals = exp_values.sum(axis=1, keepdims=True)
    return np.divide(exp_values, totals, out=np.zeros_like(exp_values), where=totals > 0.0)


def normalize_composition(values: np.ndarray, *, pseudocount: float = 0.0) -> np.ndarray:
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be nonnegative.")
    arr = _as_finite_2d(values, name="composition")
    if np.any(arr < 0.0):
        raise ValueError("Composition values must be nonnegative.")
    adjusted = arr + float(pseudocount)
    totals = adjusted.sum(axis=1, keepdims=True)
    if adjusted.shape[1] and np.any(totals <= 0.0):
        raise ValueError("Composition rows must have positive mass after adding the pseudocount.")
    return np.divide(adjusted, totals, out=np.zeros_like(adjusted), where=totals > 0.0)


def pairwise_kl(
    source: np.ndarray,
    target: np.ndarray,
    *,
    beta: float = 1.0,
    eps: float = 1e-12,
    block_size: int = 512,
) -> np.ndarray:
    source_arr, target_arr = _validate_pair(source, target)
    source_prob = np.clip(row_softmax(source_arr, beta=beta), eps, 1.0)
    target_prob = np.clip(row_softmax(target_arr, beta=beta), eps, 1.0)
    log_source = np.log(source_prob)
    log_target = np.log(target_prob)
    source_term = np.sum(source_prob * log_source, axis=1, keepdims=True)
    out = np.empty((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    step = max(1, int(block_size))
    for start in range(0, source_arr.shape[0], step):
        stop = min(start + step, source_arr.shape[0])
        out[start:stop] = source_term[start:stop] - source_prob[start:stop] @ log_target.T
    return _validated_cost(out, name="KL cost")


def pairwise_js(
    source: np.ndarray,
    target: np.ndarray,
    *,
    pseudocount: float = 0.0,
    eps: float = 1e-12,
    block_size: int = 512,
) -> np.ndarray:
    source_arr, target_arr = _validate_pair(source, target)
    source_prob = np.clip(normalize_composition(source_arr, pseudocount=pseudocount), eps, 1.0)
    target_prob = np.clip(normalize_composition(target_arr, pseudocount=pseudocount), eps, 1.0)
    out = np.empty((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    step = max(1, int(block_size))
    for start in range(0, source_arr.shape[0], step):
        stop = min(start + step, source_arr.shape[0])
        p = source_prob[start:stop, None, :]
        q = target_prob[None, :, :]
        midpoint = 0.5 * (p + q)
        out[start:stop] = 0.5 * np.sum(p * (np.log(p) - np.log(midpoint)), axis=2) + 0.5 * np.sum(
            q * (np.log(q) - np.log(midpoint)), axis=2
        )
    return _validated_cost(out, name="Jensen-Shannon cost")


def pairwise_hellinger(
    source: np.ndarray,
    target: np.ndarray,
    *,
    pseudocount: float = 0.0,
    block_size: int = 512,
) -> np.ndarray:
    source_arr, target_arr = _validate_pair(source, target)
    source_root = np.sqrt(normalize_composition(source_arr, pseudocount=pseudocount))
    target_root = np.sqrt(normalize_composition(target_arr, pseudocount=pseudocount))
    out = np.empty((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    step = max(1, int(block_size))
    for start in range(0, source_arr.shape[0], step):
        stop = min(start + step, source_arr.shape[0])
        diff = source_root[start:stop, None, :] - target_root[None, :, :]
        out[start:stop] = np.linalg.norm(diff, axis=2) / np.sqrt(2.0)
    return _validated_cost(out, name="Hellinger cost")


def pairwise_cosine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-12,
    block_size: int = 1024,
) -> np.ndarray:
    source_arr, target_arr = _validate_pair(source, target)
    out = np.empty((source_arr.shape[0], target_arr.shape[0]), dtype=float)
    target_norm = np.linalg.norm(target_arr, axis=1)
    target_unit = np.divide(
        target_arr,
        target_norm[:, None],
        out=np.zeros_like(target_arr),
        where=target_norm[:, None] > eps,
    )
    step = max(1, int(block_size))
    for start in range(0, source_arr.shape[0], step):
        stop = min(start + step, source_arr.shape[0])
        block = source_arr[start:stop]
        source_norm = np.linalg.norm(block, axis=1)
        source_unit = np.divide(block, source_norm[:, None], out=np.zeros_like(block), where=source_norm[:, None] > eps)
        distance = 1.0 - np.clip(source_unit @ target_unit.T, -1.0, 1.0)
        both_zero = (source_norm[:, None] <= eps) & (target_norm[None, :] <= eps)
        distance[both_zero] = 0.0
        out[start:stop] = distance
    return _validated_cost(out, name="cosine cost")


def pairwise_scalar_robust(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    source_arr, target_arr = _validate_pair(source, target)
    if source_arr.shape[1] != 1:
        raise ValueError("Scalar robust distance requires exactly one feature column.")
    combined = np.concatenate([source_arr[:, 0], target_arr[:, 0]])
    q25, q75 = np.percentile(combined, [25.0, 75.0]) if combined.size else (0.0, 0.0)
    scale = float(q75 - q25) + float(eps)
    out = np.abs(source_arr[:, 0, None] - target_arr[None, :, 0]) / scale
    return _validated_cost(out, name="scalar robust cost")


def _validated_cost(cost: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr < -1e-12):
        raise ValueError(f"{name} contains negative values.")
    return np.maximum(arr, 0.0)


def distance_function(name: str) -> Callable[..., np.ndarray]:
    functions: dict[str, Callable[..., np.ndarray]] = {
        "kl": pairwise_kl,
        "js": pairwise_js,
        "hellinger": pairwise_hellinger,
        "cosine": pairwise_cosine,
        "scalar_robust": pairwise_scalar_robust,
    }
    try:
        return functions[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported feature-version distance {name!r}; expected one of {sorted(functions)}.") from exc


def matrix_summary(matrix: np.ndarray) -> dict[str, object]:
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary: dict[str, object] = {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "nonfinite_count": int(arr.size - finite.size),
    }
    if finite.size:
        summary.update(
            {
                "min": float(finite.min()),
                "q05": float(np.percentile(finite, 5.0)),
                "median": float(np.median(finite)),
                "mean": float(finite.mean()),
                "q95": float(np.percentile(finite, 95.0)),
                "max": float(finite.max()),
            }
        )
    return summary
