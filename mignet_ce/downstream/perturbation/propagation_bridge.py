"""Validated baseline-PIJ propagation used only for T1-to-T2 intervention transfer."""
from __future__ import annotations

import numpy as np

from mignet_ce.downstream.analysis.grn_perturbation.propagation import horizontal


def propagate_delta(delta_t: np.ndarray, spot_pij_propagation: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_t, dtype=float)
    pij = np.asarray(spot_pij_propagation, dtype=float)
    if pij.ndim != 2 or delta.ndim == 0 or pij.shape[0] != delta.shape[0]:
        raise ValueError("Propagation PIJ source dimension must match T1 state rows.")
    if not np.isfinite(pij).all() or not np.isfinite(delta).all():
        raise ValueError("Propagation inputs must be finite.")
    return horizontal(delta, pij)


def propagate_state(state_t: np.ndarray, baseline_t: np.ndarray, baseline_tp: np.ndarray, spot_pij_propagation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_t = np.asarray(state_t, dtype=float) - np.asarray(baseline_t, dtype=float)
    delta_tp = propagate_delta(delta_t, spot_pij_propagation)
    state_tp = np.asarray(baseline_tp, dtype=float) + delta_tp
    if not np.isfinite(state_tp).all():
        raise ValueError("Propagated T2 state is non-finite.")
    return state_tp, delta_tp
