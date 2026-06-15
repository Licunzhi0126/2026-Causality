from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.transition.cost_components import (
    pairwise_feature_cost,
    pairwise_scalar_cost,
    pairwise_spatial_cost,
)


class EnergyEntropyOTPijMethod:
    name = "energy_entropy_ot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        def component_builder(
            source_features: np.ndarray,
            target_features: np.ndarray,
            source_coords: np.ndarray | None,
            target_coords: np.ndarray | None,
            space: str,
        ):
            components = {
                "expression": pairwise_feature_cost(
                    source_features,
                    target_features,
                    metric=cfg.pij_cost_metric,
                ),
                "graph_energy": pairwise_scalar_cost(
                    np.linalg.norm(source_features, axis=1),
                    np.linalg.norm(target_features, axis=1),
                ),
            }
            if source_coords is not None and target_coords is not None:
                components["spatial"] = pairwise_spatial_cost(source_coords, target_coords)
            elif cfg.pij_spatial_weight > 0:
                raise ValueError("energy_entropy_ot requires coordinates when pij_spatial_weight > 0.")
            return (
                components,
                {
                    "expression": cfg.pij_expr_weight,
                    "spatial": cfg.pij_spatial_weight,
                    "graph_energy": cfg.pij_graph_energy_weight,
                },
            )

        return run_ot_pij_method(
            context=context,
            cfg=cfg,
            pairs=pairs,
            method_name=self.name,
            component_builder=component_builder,
        )
