from __future__ import annotations

"""Thin sensitivity adapter for the shared production N/G transition."""

import numpy as np

from mignet_ce.pij.compare._shared.ng_kl_ot import (
    build_ng_component_costs_numpy as build_component_costs,
    canonical_ng_pij_from_cost_numpy,
    mix_ng_cost_numpy as mix_cost,
)


def balanced_pij_from_cost(
    cost: np.ndarray,
    *,
    tau: float,
) -> tuple[np.ndarray, dict[str, object]]:
    _joint, pij, metadata = canonical_ng_pij_from_cost_numpy(
        cost,
        temperature=tau,
    )
    return pij, metadata
