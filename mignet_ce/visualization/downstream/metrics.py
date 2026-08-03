from __future__ import annotations

import numpy as np
import pandas as pd


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


def ei_decomposition(matrix: np.ndarray) -> dict[str, float]:
    p = row_normalize(matrix)
    p_bar = p.mean(axis=0)
    h_effect = float(entropy(p_bar))
    row_h = entropy(p, axis=1)
    h_noise = float(np.mean(row_h))
    log_capacity = float(np.log2(max(p.shape[1], 1)))
    return {
        "n_source": int(p.shape[0]),
        "n_target": int(p.shape[1]),
        "H_effect": h_effect,
        "H_noise": h_noise,
        "EI": h_effect - h_noise,
        "determinism": log_capacity - h_noise,
        "degeneracy": log_capacity - h_effect,
        "row_entropy_mean": h_noise,
        "row_entropy_sd": float(np.std(row_h, ddof=1)) if len(row_h) > 1 else 0.0,
    }


def state_ei(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = row_normalize(matrix)
    p_bar = p.mean(axis=0)
    return np.sum(p * np.log2((p + eps) / (p_bar[None, :] + eps)), axis=1)


def effective_state_number(counts: np.ndarray, eps: float = 1e-12) -> float:
    values = np.maximum(np.asarray(counts, dtype=float), 0.0)
    if values.sum() <= eps:
        return 0.0
    return float(2.0 ** entropy(values / values.sum()))


def relative_frobenius(predicted: np.ndarray, observed: np.ndarray, eps: float = 1e-12) -> float:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {observed.shape}")
    return float(np.linalg.norm(predicted - observed) / max(np.linalg.norm(observed), eps))


def mean_row_js(predicted: np.ndarray, observed: np.ndarray, eps: float = 1e-12) -> float:
    a = row_normalize(predicted)
    b = row_normalize(observed)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    midpoint = 0.5 * (a + b)
    kl_a = np.sum(a * np.log2((a + eps) / (midpoint + eps)), axis=1)
    kl_b = np.sum(b * np.log2((b + eps) / (midpoint + eps)), axis=1)
    return float(np.mean(np.maximum(0.5 * (kl_a + kl_b), 0.0)))


def compose_transitions(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("matrices is empty")
    result = row_normalize(matrices[0])
    for matrix in matrices[1:]:
        next_matrix = row_normalize(matrix)
        if result.shape[1] != next_matrix.shape[0]:
            raise ValueError(f"transition shape mismatch: {result.shape} then {next_matrix.shape}")
        result = row_normalize(result @ next_matrix)
    return result


def hierarchy_counts(
    k150_map: pd.DataFrame,
    k40_map: pd.DataFrame,
    k150_units: list[str],
    k40_units: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = k150_map[["spot_id", "domain_id"]].rename(columns={"domain_id": "k150"}).merge(
        k40_map[["spot_id", "domain_id"]].rename(columns={"domain_id": "k40"}),
        on="spot_id",
        how="inner",
    )
    table = pd.crosstab(merged["k150"], merged["k40"]).reindex(
        index=k150_units,
        columns=k40_units,
        fill_value=0,
    )
    return table.to_numpy(dtype=float), merged


def aggregate_transition_by_overlap(
    p_lower: np.ndarray,
    counts_source: np.ndarray,
    counts_target: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    p = row_normalize(p_lower)
    c_source = np.maximum(np.asarray(counts_source, dtype=float), 0.0)
    c_target = np.maximum(np.asarray(counts_target, dtype=float), 0.0)
    if p.shape != (c_source.shape[0], c_target.shape[0]):
        raise ValueError(
            "Pij/overlap shape mismatch: "
            f"Pij={p.shape}, source overlap={c_source.shape}, target overlap={c_target.shape}"
        )
    source_macro_mass = c_source.sum(axis=0, keepdims=True)
    source_weights = (c_source / np.maximum(source_macro_mass, eps)).T
    target_micro_mass = c_target.sum(axis=1, keepdims=True)
    target_membership = c_target / np.maximum(target_micro_mass, eps)
    return row_normalize(source_weights @ p @ target_membership)


def purity_entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> pd.DataFrame:
    values = np.maximum(np.asarray(counts, dtype=float), 0.0)
    probs = values / np.maximum(values.sum(axis=1, keepdims=True), eps)
    h = entropy(probs, axis=1)
    return pd.DataFrame(
        {
            "purity": probs.max(axis=1),
            "mapping_entropy": h,
            "mapping_entropy_norm": h / np.log2(max(probs.shape[1], 2)),
        }
    )
