from __future__ import annotations

from typing import List

import numpy as np

from mignet_ce.embeddings import layer_graph_laplacian_features
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.networks.base import NetworkContext
from mignet_ce.representations.common import global_scale_feature_lists
from mignet_ce.pij.base import MethodResult


def build_laplacian_result(
    context: NetworkContext,
    n_components: int,
    normalized: bool,
) -> MethodResult:
    lower_raw: List[np.ndarray] = []
    upper_raw: List[np.ndarray] = []
    for t in range(len(context.time_points)):
        lower_embedding = layer_graph_laplacian_features(
            context.lower_graphs[t],
            n_components=n_components,
            normalized=normalized,
        )
        upper_embedding = layer_graph_laplacian_features(
            context.upper_graphs[t],
            n_components=n_components,
            normalized=normalized,
        )
        if context.feature_alignment_space == "native_units":
            lower_feat = lower_embedding
            upper_feat = upper_embedding
        else:
            lower_feat, _ = aggregate_lower_features_to_upper(lower_embedding, context.overlaps[t])
            upper_feat = align_upper_features(upper_embedding, context.upper_units_by_time[t], context.stable_upper_units)
        lower_raw.append(lower_feat)
        upper_raw.append(upper_feat)
    if context.feature_alignment_space == "native_units":
        lower_coords = context.lower_coords_by_time
        upper_coords = context.upper_coords_by_time
    else:
        lower_coords = context.upper_coords_by_time
        upper_coords = context.upper_coords_by_time
    lower_scaled, upper_scaled = global_scale_feature_lists(lower_raw, upper_raw)
    return MethodResult(
        lower_features=lower_scaled,
        upper_features=upper_scaled,
        lower_coords=lower_coords,
        upper_coords=upper_coords,
        method_metadata={
            "representation": "laplacian",
            "feature_alignment_space": context.feature_alignment_space,
            "laplacian_components": int(n_components),
            "laplacian_normalized": bool(normalized),
        },
    )
