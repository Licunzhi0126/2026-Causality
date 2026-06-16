from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.developmental_features import (
    DevelopmentalFeatureTable,
    load_developmental_features_for_pij,
    require_columns,
    select_first_available_column,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.transition.cost_components import pairwise_feature_cost, pairwise_scalar_cost


class SROTPijMethod:
    name = "sr_ot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        feature_cache: dict[tuple[str, int], DevelopmentalFeatureTable] = {}

        def table(space: str, time_index: int) -> DevelopmentalFeatureTable:
            key = (space, time_index)
            if key not in feature_cache:
                feature_cache[key] = load_developmental_features_for_pij(context, cfg, time_index, space)
            return feature_cache[key]

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
            column = select_first_available_column(source_table.values, ["sr", "potency_score"], self.name)
            require_columns(target_table.values, [column], self.name)
            source_values = source_table.values[column].to_numpy(dtype=float)
            target_values = target_table.values[column].to_numpy(dtype=float)
            component_name = "sr" if column == "sr" else "potency"
            components = {
                "expression": pairwise_feature_cost(
                    source_features,
                    target_features,
                    metric=cfg.pij_cost_metric,
                ),
                component_name: pairwise_scalar_cost(source_values, target_values),
            }
            if cfg.pij_reverse_potency_weight > 0:
                components[f"reverse_{component_name}"] = np.maximum(0.0, target_values[None, :] - source_values[:, None])
            scalar_weight = cfg.pij_sr_weight if column == "sr" else cfg.pij_potency_weight
            weights = {
                "expression": cfg.pij_expr_weight,
                component_name: scalar_weight,
                f"reverse_{component_name}": cfg.pij_reverse_potency_weight,
            }
            metadata = {
                "feature_columns_used": [column],
                "feature_aggregation": cfg.pij_feature_aggregation,
                "missing_feature_policy": cfg.pij_missing_feature_policy,
                "developmental_features": {
                    "source": source_table.metadata,
                    "target": target_table.metadata,
                },
            }
            return components, weights, metadata

        return run_ot_pij_method(
            context=context,
            cfg=cfg,
            pairs=pairs,
            method_name=self.name,
            component_builder=component_builder,
        )
