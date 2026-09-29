from __future__ import annotations

from typing import List

import numpy as np

from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.networks.base import NetworkContext
from mignet_ce.representations.common import global_scale_feature_lists, reduce_feature_lists
from mignet_ce.pij.base import MethodResult


def build_graph_feature_result(
    context: NetworkContext,
    n_components: int | None,
    seed: int = 42,
) -> MethodResult:
    if context.feature_alignment_space == "native_units":
        lower_raw = [np.asarray(matrix, dtype=float) for matrix in context.lower_mats]
        upper_raw = [np.asarray(matrix, dtype=float) for matrix in context.upper_mats]
        lower_coords = context.lower_coords_by_time
        upper_coords = context.upper_coords_by_time
    else:
        lower_raw = []
        upper_raw = []
        for t in range(len(context.time_points)):
            lower_feat, _ = aggregate_lower_features_to_upper(context.lower_mats[t], context.overlaps[t])
            upper_feat = align_upper_features(context.upper_mats[t], context.upper_units_by_time[t], context.stable_upper_units)
            lower_raw.append(lower_feat)
            upper_raw.append(upper_feat)
        lower_coords = context.upper_coords_by_time
        upper_coords = context.upper_coords_by_time
    lower_reduced, upper_reduced, reduction_metadata = reduce_feature_lists(
        lower_raw,
        upper_raw,
        n_components=n_components,
        seed=seed,
    )
    lower_scaled, upper_scaled = global_scale_feature_lists(lower_reduced, upper_reduced)
    return MethodResult(
        lower_features=lower_scaled,
        upper_features=upper_scaled,
        lower_coords=lower_coords,
        upper_coords=upper_coords,
        method_metadata={
            "representation": "graph_features",
            "feature_alignment_space": context.feature_alignment_space,
            "network_feature_names": context.feature_names,
            "feature_reduction": reduction_metadata,
        },
    )
