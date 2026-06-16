from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.developmental_features import DevelopmentalFeatureTable, load_developmental_features_for_pij, velocity_columns
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.transition.cost_components import pairwise_feature_cost, pairwise_velocity_direction_cost


class VelocityOTPijMethod:
    name = "velocity_ot"

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
            velocity_cols = velocity_columns(source_table.values)
            if not velocity_cols:
                raise ValueError("velocity_ot requires velocity_* columns in developmental feature CSV.")
            source_velocity = source_table.values.loc[:, velocity_cols].to_numpy(dtype=float)
            if source_velocity.shape[1] != source_features.shape[1]:
                raise ValueError("velocity_ot requires velocity_* dimension to match graph feature dimension.")
            components = {
                "expression": pairwise_feature_cost(
                    source_features,
                    target_features,
                    metric=cfg.pij_cost_metric,
                ),
                "velocity": pairwise_velocity_direction_cost(
                    source_features,
                    target_features,
                    source_velocity,
                ),
            }
            weights = {
                "expression": cfg.pij_expr_weight,
                "velocity": cfg.pij_velocity_weight,
            }
            metadata = {
                "feature_columns_used": velocity_cols,
                "feature_aggregation": cfg.pij_feature_aggregation,
                "missing_feature_policy": cfg.pij_missing_feature_policy,
                "developmental_features": {
                    "source": source_table.metadata,
                    "target": table(space, t1).metadata,
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
