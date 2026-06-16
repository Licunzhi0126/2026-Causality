from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.developmental_features import DevelopmentalFeatureTable, load_developmental_features_for_pij, require_columns
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.transition.cost_components import pairwise_feature_cost, pairwise_scalar_cost


class PseudotimeOTPijMethod:
    name = "pseudotime_ot"

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
            require_columns(source_table.values, ["pseudotime"], self.name)
            require_columns(target_table.values, ["pseudotime"], self.name)
            source_pt = source_table.values["pseudotime"].to_numpy(dtype=float)
            target_pt = target_table.values["pseudotime"].to_numpy(dtype=float)
            components = {
                "expression": pairwise_feature_cost(
                    source_features,
                    target_features,
                    metric=cfg.pij_cost_metric,
                ),
                "pseudotime": pairwise_scalar_cost(source_pt, target_pt),
            }
            if cfg.pij_backward_pseudotime_weight > 0:
                components["backward_pseudotime"] = np.maximum(0.0, source_pt[:, None] - target_pt[None, :])
            weights = {
                "expression": cfg.pij_expr_weight,
                "pseudotime": cfg.pij_pseudotime_weight,
                "backward_pseudotime": cfg.pij_backward_pseudotime_weight,
            }
            metadata = {
                "feature_columns_used": ["pseudotime"],
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
