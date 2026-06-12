from __future__ import annotations

from typing import List

import numpy as np

from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.pij.base import MethodResult
from mignet_ce.pipelines.vertical_context import VerticalPairContext
from mignet_ce.representations.common import global_scale_feature_lists


def build_joint_nmf_result(
    context: VerticalPairContext,
    n_components: int,
    max_iter: int,
    seed: int,
) -> MethodResult:
    engine = TemporalMetricsEngine()
    w_lower_cells, _ = engine.temporal_joint_nmf(
        context.lower_mats,
        n_components=n_components,
        max_iter=max_iter,
        seed=seed,
    )
    w_upper_current, _ = engine.temporal_joint_nmf(
        context.upper_mats,
        n_components=n_components,
        max_iter=max_iter,
        seed=seed,
    )
    lower_raw: List[np.ndarray] = []
    upper_raw: List[np.ndarray] = []
    for t in range(len(context.time_points)):
        lower_feat, _ = aggregate_lower_features_to_upper(w_lower_cells[t], context.overlaps[t])
        upper_feat = align_upper_features(w_upper_current[t], context.upper_units_by_time[t], context.stable_upper_units)
        lower_raw.append(lower_feat)
        upper_raw.append(upper_feat)
    lower_scaled, upper_scaled = global_scale_feature_lists(lower_raw, upper_raw)
    return MethodResult(
        lower_features=lower_scaled,
        upper_features=upper_scaled,
        lower_coords=context.upper_coords_by_time,
        upper_coords=context.upper_coords_by_time,
        method_metadata={
            "representation": "joint_nmf",
            "nmf_components": int(n_components),
            "nmf_max_iter": int(max_iter),
            "nmf_seed": int(seed),
        },
    )
