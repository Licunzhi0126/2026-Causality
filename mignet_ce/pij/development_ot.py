from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.developmental_features import (
    DevelopmentalFeatureTable,
    load_developmental_features_for_pij,
    require_columns,
    velocity_columns,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.transition.cost_components import (
    pairwise_feature_cost,
    pairwise_scalar_cost,
    pairwise_spatial_cost,
    pairwise_velocity_direction_cost,
)


class DevelopmentOTPijMethod:
    name = "development_ot"

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
            components = {
                "expression": pairwise_feature_cost(
                    source_features,
                    target_features,
                    metric=cfg.pij_cost_metric,
                )
            }
            weights = {"expression": cfg.pij_expr_weight}
            feature_columns_used: list[str] = []
            skipped_feature_columns: list[str] = []

            if cfg.pij_spatial_weight > 0:
                if source_coords is None or target_coords is None:
                    raise ValueError("development_ot requires coordinates when pij_spatial_weight > 0.")
                components["spatial"] = pairwise_spatial_cost(source_coords, target_coords)
                weights["spatial"] = cfg.pij_spatial_weight

            if cfg.pij_pseudotime_weight > 0 or cfg.pij_backward_pseudotime_weight > 0:
                require_columns(source_table.values, ["pseudotime"], self.name)
                require_columns(target_table.values, ["pseudotime"], self.name)
                source_pt = source_table.values["pseudotime"].to_numpy(dtype=float)
                target_pt = target_table.values["pseudotime"].to_numpy(dtype=float)
                components["pseudotime"] = pairwise_scalar_cost(source_pt, target_pt)
                weights["pseudotime"] = cfg.pij_pseudotime_weight
                feature_columns_used.append("pseudotime")
                if cfg.pij_backward_pseudotime_weight > 0:
                    components["backward_pseudotime"] = np.maximum(0.0, source_pt[:, None] - target_pt[None, :])
                    weights["backward_pseudotime"] = cfg.pij_backward_pseudotime_weight

            has_sr = "sr" in source_table.values.columns and "sr" in target_table.values.columns
            has_potency = "potency_score" in source_table.values.columns and "potency_score" in target_table.values.columns
            added_potency_like = False
            if cfg.pij_sr_weight > 0:
                if has_sr:
                    components["sr"] = pairwise_scalar_cost(
                        source_table.values["sr"].to_numpy(dtype=float),
                        target_table.values["sr"].to_numpy(dtype=float),
                    )
                    weights["sr"] = cfg.pij_sr_weight
                    feature_columns_used.append("sr")
                    added_potency_like = True
                else:
                    skipped_feature_columns.append("sr")
            if cfg.pij_potency_weight > 0:
                if has_potency:
                    components["potency"] = pairwise_scalar_cost(
                        source_table.values["potency_score"].to_numpy(dtype=float),
                        target_table.values["potency_score"].to_numpy(dtype=float),
                    )
                    weights["potency"] = cfg.pij_potency_weight
                    feature_columns_used.append("potency_score")
                    added_potency_like = True
                else:
                    skipped_feature_columns.append("potency_score")
            if (cfg.pij_sr_weight > 0 or cfg.pij_potency_weight > 0 or cfg.pij_reverse_potency_weight > 0) and not added_potency_like:
                raise ValueError(
                    "development_ot requires at least one shared developmental feature column from ['sr', 'potency_score'] "
                    "when SR/potency weights are positive."
                )
            if cfg.pij_reverse_potency_weight > 0:
                reverse_column = "potency_score" if has_potency else "sr"
                source_values = source_table.values[reverse_column].to_numpy(dtype=float)
                target_values = target_table.values[reverse_column].to_numpy(dtype=float)
                components["reverse_potency"] = np.maximum(0.0, target_values[None, :] - source_values[:, None])
                weights["reverse_potency"] = cfg.pij_reverse_potency_weight
                if reverse_column not in feature_columns_used:
                    feature_columns_used.append(reverse_column)

            if cfg.pij_velocity_weight > 0:
                velocity_cols = velocity_columns(source_table.values)
                if not velocity_cols:
                    raise ValueError("development_ot requires velocity_* columns when pij_velocity_weight > 0.")
                source_velocity = source_table.values.loc[:, velocity_cols].to_numpy(dtype=float)
                if source_velocity.shape[1] != source_features.shape[1]:
                    raise ValueError("development_ot requires velocity_* dimension to match graph feature dimension.")
                components["velocity"] = pairwise_velocity_direction_cost(source_features, target_features, source_velocity)
                weights["velocity"] = cfg.pij_velocity_weight
                feature_columns_used.extend(velocity_cols)

            metadata = {
                "feature_columns_used": feature_columns_used,
                "skipped_feature_columns": skipped_feature_columns,
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
