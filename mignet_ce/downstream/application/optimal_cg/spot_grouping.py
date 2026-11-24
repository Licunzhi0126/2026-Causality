"""Deterministic source groups selected by full-row spot argmax."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceGroup:
    source_cluster: int
    target_cluster: int
    source_indices: np.ndarray
    best_target_spot_indices: np.ndarray


def validate_global_outputs(pij_micro, s_t, s_tp):
    arrays = [np.asarray(x, dtype=np.float64) for x in (pij_micro, s_t, s_tp)]
    for name, arr in zip(("PIJ_micro", "S_t", "S_tp"), arrays):
        if arr.ndim != 2 or min(arr.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D matrix.")
        if not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError(f"{name} must contain finite non-negative values.")
        mass = arr.sum(axis=1)
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError(f"{name} rows must have finite positive mass.")
    p, source, target = arrays
    if p.shape != (source.shape[0], target.shape[0]):
        raise ValueError("PIJ_micro shape must equal (S_t rows, S_tp rows).")
    return arrays


def hard_assignments(s_t, s_tp):
    return np.asarray(s_t).argmax(axis=1), np.asarray(s_tp).argmax(axis=1)


def assign_best_target_spots(pij_micro, target_labels):
    best = np.asarray(pij_micro).argmax(axis=1)
    return best, np.asarray(target_labels)[best]


def build_source_groups(pij_micro, s_t, s_tp):
    p, source, target = validate_global_outputs(pij_micro, s_t, s_tp)
    source_labels, target_labels = hard_assignments(source, target)
    best, clusters = assign_best_target_spots(p, target_labels)
    # Sort once rather than scanning the entire PIJ for each group.
    order = np.lexsort((clusters, source_labels))
    keys = np.column_stack((source_labels[order], clusters[order]))
    boundaries = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(order)]
    groups = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = np.sort(order[start:stop])
        a, b = keys[start]
        groups.append(SourceGroup(int(a), int(b), indices, best[indices]))
    diagnostics = {
        "inactive_source_clusters": np.setdiff1d(np.arange(source.shape[1]), source_labels),
        "inactive_target_clusters": np.setdiff1d(np.arange(target.shape[1]), target_labels),
        "global_row_sum_max_abs_error": float(np.max(np.abs(p.sum(axis=1) - 1))),
        "global_nonstochastic_row_count": int(np.count_nonzero(~np.isclose(p.sum(axis=1), 1, rtol=1e-5, atol=1e-8))),
    }
    return groups, diagnostics
