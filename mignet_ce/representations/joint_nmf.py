from __future__ import annotations

from typing import List

import numpy as np

from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.networks.base import NetworkContext
from mignet_ce.representations.common import global_scale_feature_lists
from mignet_ce.pij.base import MethodResult


def build_joint_nmf_result(
    context: NetworkContext,
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
    if context.feature_alignment_space == "native_units":
        lower_raw = [np.asarray(matrix, dtype=float) for matrix in w_lower_cells]
        upper_raw = [np.asarray(matrix, dtype=float) for matrix in w_upper_current]
        lower_coords = context.lower_coords_by_time
        upper_coords = context.upper_coords_by_time
    else:
        lower_raw = []
        upper_raw = []
        for t in range(len(context.time_points)):
            lower_feat, _ = aggregate_lower_features_to_upper(w_lower_cells[t], context.overlaps[t])
            upper_feat = align_upper_features(w_upper_current[t], context.upper_units_by_time[t], context.stable_upper_units)
            lower_raw.append(lower_feat)
            upper_raw.append(upper_feat)
        lower_coords = context.upper_coords_by_time
        upper_coords = context.upper_coords_by_time
    lower_scaled, upper_scaled = global_scale_feature_lists(lower_raw, upper_raw)
    return MethodResult(
        lower_features=lower_scaled,
        upper_features=upper_scaled,
        lower_coords=lower_coords,
        upper_coords=upper_coords,
        method_metadata={
            "representation": "joint_nmf",
            "feature_alignment_space": context.feature_alignment_space,
            "nmf_components": int(n_components),
            "nmf_max_iter": int(max_iter),
            "nmf_seed": int(seed),
        },
    )
