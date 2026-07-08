from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.legacy._developmental_ot_components import (
    build_expression_cost,
    build_graph_energy_cost,
    build_pseudotime_cost,
    build_spatial_cost,
    build_sr_cost,
    developmental_metadata,
    make_developmental_table_loader,
)
from mignet_ce.pij.legacy._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class ExprPseudotimeSREnergySpatialOTPijMethod:
    name = "expr_pseudotime_sr_energy_spatial_ot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        table = make_developmental_table_loader(context, cfg)

        def component_builder(
            source_features: np.ndarray,
            target_features: np.ndarray,
            source_coords: np.ndarray | None,
            target_coords: np.ndarray | None,
            space: str,
            t0: int,
            t1: int,
        ):
            source_table = table(space, t0)
            target_table = table(space, t1)
            component_name, sr_cost, sr_column = build_sr_cost(source_table, target_table, self.name)
            components = {
                "expression": build_expression_cost(source_features, target_features, cfg),
                "pseudotime": build_pseudotime_cost(source_table, target_table, self.name),
                component_name: sr_cost,
                "graph_energy": build_graph_energy_cost(source_features, target_features),
                "spatial": build_spatial_cost(source_coords, target_coords, self.name),
            }
            weights = {
                "expression": cfg.pij_expr_weight,
                "pseudotime": cfg.pij_pseudotime_weight,
                component_name: cfg.pij_sr_weight,
                "graph_energy": cfg.pij_graph_energy_weight,
                "spatial": cfg.pij_spatial_weight,
            }
            metadata = developmental_metadata(source_table, target_table, ["pseudotime", sr_column], cfg)
            return components, weights, metadata

        return run_ot_pij_method(
            context=context,
            cfg=cfg,
            pairs=pairs,
            method_name=self.name,
            component_builder=component_builder,
        )
