"""Construct validated conditional spot-level systems for macro cluster pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAIR_ROW_MASS_EPS = 1e-12
MIN_LOCAL_SPOTS = 10


@dataclass(frozen=True)
class LocalPairData:
    source_cluster: int
    target_cluster: int
    source_indices_original: np.ndarray
    source_indices_used: np.ndarray
    target_indices: np.ndarray
    pij_raw: np.ndarray
    pij_conditional: np.ndarray
    retained_mass: np.ndarray
    dropped_mask: np.ndarray
    status: str
    warning: str | None = None

    @property
    def n_source_original(self) -> int:
        return int(self.source_indices_original.size)

    @property
    def n_source_used(self) -> int:
        return int(self.source_indices_used.size)

    @property
    def n_source_dropped(self) -> int:
        return int(self.dropped_mask.sum())

    @property
    def n_target(self) -> int:
        return int(self.target_indices.size)


def validate_global_outputs(pij_micro: np.ndarray, pij_macro: np.ndarray, s_t: np.ndarray, s_tp: np.ndarray) -> None:
    micro, macro, source, target = map(np.asarray, (pij_micro, pij_macro, s_t, s_tp))
    if micro.ndim != 2 or macro.ndim != 2 or source.ndim != 2 or target.ndim != 2:
        raise ValueError("PIJ and assignment arrays must all be 2D.")
    if micro.shape != (source.shape[0], target.shape[0]):
        raise ValueError("PIJ_micro shape must equal (S_t rows, S_tp rows).")
    if macro.shape != (source.shape[1], target.shape[1]):
        raise ValueError("PIJ_macro shape must equal (S_t columns, S_tp columns).")
    for name, values in (("PIJ_micro", micro), ("PIJ_macro", macro), ("S_t", source), ("S_tp", target)):
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values.")


def build_local_pair(*, pij_micro: np.ndarray, s_t: np.ndarray, s_tp: np.ndarray, source_cluster: int, target_cluster: int, eps: float = PAIR_ROW_MASS_EPS, min_local_spots: int = MIN_LOCAL_SPOTS) -> LocalPairData:
    """Slice both PIJ axes, retain usable rows, and condition on the target cluster."""
    micro = np.asarray(pij_micro, dtype=np.float64)
    source_assignment, target_assignment = np.asarray(s_t), np.asarray(s_tp)
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")
    source_labels = source_assignment.argmax(axis=1)
    target_labels = target_assignment.argmax(axis=1)
    source_indices = np.flatnonzero(source_labels == source_cluster)
    target_indices = np.flatnonzero(target_labels == target_cluster)
    raw = micro[np.ix_(source_indices, target_indices)]
    retained_mass = raw.sum(axis=1)
    dropped_mask = retained_mass <= eps
    used_indices = source_indices[~dropped_mask]
    conditional = raw[~dropped_mask]
    if conditional.size:
        conditional = conditional / retained_mass[~dropped_mask, None]
    problems: list[str] = []
    if used_indices.size < min_local_spots:
        problems.append(f"source spots used {used_indices.size} < MIN_LOCAL_SPOTS {min_local_spots}")
    if target_indices.size < min_local_spots:
        problems.append(f"target spots {target_indices.size} < MIN_LOCAL_SPOTS {min_local_spots}")
    if conditional.size and (not np.isfinite(conditional).all() or np.any(conditional < 0.0) or not np.allclose(conditional.sum(axis=1), 1.0, atol=1e-6)):
        problems.append("conditional local PIJ is not finite, non-negative, and row-stochastic")
    dropped_fraction = float(dropped_mask.mean()) if dropped_mask.size else 1.0
    warning = (f"high low-mass row drop fraction: {dropped_fraction:.3f}" if dropped_fraction > 0.5 else None)
    return LocalPairData(
        source_cluster=int(source_cluster), target_cluster=int(target_cluster),
        source_indices_original=source_indices, source_indices_used=used_indices,
        target_indices=target_indices, pij_raw=raw, pij_conditional=conditional,
        retained_mass=retained_mass, dropped_mask=dropped_mask,
        status="skipped" if problems else "ready",
        warning="; ".join(problems) if problems else warning,
    )
