"""Condition a source group on every spot of its selected target cluster."""
from dataclasses import dataclass

import numpy as np

from .spot_grouping import SourceGroup


@dataclass(frozen=True)
class GroupMicroData:
    source_indices_original: np.ndarray
    source_indices_used: np.ndarray
    target_indices: np.ndarray
    pij_raw: np.ndarray
    pij_conditional: np.ndarray
    retained_mass: np.ndarray
    dropped_mask: np.ndarray
    status: str
    warning: str


def validate_options(eps):
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative.")


def build_group_micro(*, pij_micro, target_labels, group: SourceGroup, eps=1e-12):
    validate_options(eps)
    targets = np.flatnonzero(np.asarray(target_labels) == group.target_cluster)
    raw = np.asarray(pij_micro, dtype=np.float64)[np.ix_(group.source_indices, targets)]
    mass = raw.sum(axis=1)
    if not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(mass).all():
        raise ValueError("Group matrix must be finite and non-negative.")
    dropped = mass <= eps
    used = group.source_indices[~dropped]
    local = raw[~dropped] / mass[~dropped, None]
    warnings = []
    if dropped.any():
        warnings.append("dropped source rows: target cluster mass <= eps")
    status = "ready"
    if used.size == 0 or targets.size == 0:
        status = "skipped"
        warnings.append("no valid source spots or empty target cluster")
    if used.size == 1:
        warnings.append("small_group")
    if targets.size == 1:
        warnings.append("single_target_spot")
    if local.size and (not np.isfinite(local).all() or np.any(local < 0) or
                       not np.allclose(local.sum(axis=1), 1, rtol=1e-10, atol=1e-12)):
        raise ValueError("Conditional Micro PIJ must have finite normalized positive rows.")
    return GroupMicroData(group.source_indices, used, targets, raw, local, mass, dropped, status, "; ".join(warnings))
