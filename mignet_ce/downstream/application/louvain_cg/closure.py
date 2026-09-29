"""Thin Louvain adapters around existing closure and direct-vs-induced diagnostics."""

from __future__ import annotations

import numpy as np


def summarize_closure(spot_pij: np.ndarray, source_assignment: np.ndarray, target_assignment: np.ndarray, direct_macro_pij: np.ndarray) -> dict[str, object]:
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget, js_rows

    budget = information_closure_budget(spot_pij, source_assignment, target_assignment)
    direct = np.asarray(direct_macro_pij, dtype=float)
    induced = np.asarray(budget["q_induced"], dtype=float)
    if direct.shape != induced.shape:
        raise ValueError(f"Direct macro PIJ {direct.shape} and induced Q {induced.shape} differ.")
    return {key: value for key, value in budget.items() if key not in {"observed", "source_assignment", "target_assignment", "weights", "macro_mass", "predicted_induced"}} | {"direct_induced_q_js": float(np.mean(js_rows(direct, induced)))}
