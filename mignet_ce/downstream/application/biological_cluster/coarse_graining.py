"""Delegate induced dynamics, EI, and closure budgets to existing implementations."""

from __future__ import annotations

import numpy as np


def evaluate_fixed_partition(spot_pij: np.ndarray, source_assignment: np.ndarray, target_assignment: np.ndarray) -> dict[str, object]:
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import (
        effective_information,
        information_closure_budget,
        state_level_ei,
    )

    budget = information_closure_budget(spot_pij, source_assignment, target_assignment)
    cluster_pij = np.asarray(budget["q_induced"], dtype=np.float64)
    spot_ei = float(effective_information(np.asarray(spot_pij, dtype=float)))
    cluster_ei = float(effective_information(cluster_pij))
    return {"cluster_pij": cluster_pij, "spot_ei": spot_ei, "cluster_ei": cluster_ei, "delta_ei": cluster_ei - spot_ei, "cluster_ei_contributions": state_level_ei(cluster_pij), "budget": budget}

