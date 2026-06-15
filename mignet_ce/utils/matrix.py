from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def safe_row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    out = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > eps)
    zero_rows = np.squeeze(row_sums <= eps, axis=1)
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / arr.shape[1]
    return out


def row_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    arr = np.asarray(scores, dtype=float) / float(temperature)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    finite = np.isfinite(arr)
    row_has_finite = finite.any(axis=1)
    shifted = np.zeros_like(arr, dtype=float)
    shifted[row_has_finite] = arr[row_has_finite] - np.max(arr[row_has_finite], axis=1, keepdims=True)
    exp_scores = np.where(finite, np.exp(shifted), 0.0)
    return safe_row_normalize(exp_scores)


def save_transition_npz(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, sp.csr_matrix(np.asarray(matrix, dtype=float)))


def serialize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            out[key] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def transition_topk_table(
    matrix: np.ndarray,
    source_units: Sequence[str],
    target_units: Sequence[str],
    time_pair: str,
    space: str,
    top_k: int = 10,
    pij_method: str | None = None,
) -> pd.DataFrame:
    arr = np.asarray(matrix, dtype=float)
    source_units = list(map(str, source_units))
    target_units = list(map(str, target_units))
    if arr.shape != (len(source_units), len(target_units)):
        raise ValueError(
            f"Matrix shape {arr.shape} does not match {len(source_units)} source and {len(target_units)} target units."
        )
    k = max(1, min(int(top_k), arr.shape[1])) if arr.shape[1] else 0
    rows: list[dict[str, object]] = []
    for i, source in enumerate(source_units):
        if k == 0:
            continue
        values = arr[i]
        order = np.argsort(-values)[:k]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "time_pair": time_pair,
                    "space": space,
                    "pij_method": pij_method,
                    "source_unit": source,
                    "target_unit": target_units[j],
                    "probability": float(values[j]),
                    "pij": float(values[j]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)
