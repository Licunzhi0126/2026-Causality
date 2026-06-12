from __future__ import annotations

from typing import List

import numpy as np

from mignet_ce.embeddings import layer_graph_laplacian_features
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.pij.base import MethodResult
from mignet_ce.pipelines.vertical_context import VerticalPairContext
from mignet_ce.representations.common import global_scale_feature_lists


def build_laplacian_result(
    context: VerticalPairContext,
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
        lower_feat, _ = aggregate_lower_features_to_upper(lower_embedding, context.overlaps[t])
        upper_feat = align_upper_features(upper_embedding, context.upper_units_by_time[t], context.stable_upper_units)
        lower_raw.append(lower_feat)
        upper_raw.append(upper_feat)
    lower_scaled, upper_scaled = global_scale_feature_lists(lower_raw, upper_raw)
    return MethodResult(
        lower_features=lower_scaled,
        upper_features=upper_scaled,
        lower_coords=context.upper_coords_by_time,
        upper_coords=context.upper_coords_by_time,
        method_metadata={
            "representation": "laplacian",
            "laplacian_components": int(n_components),
            "laplacian_normalized": bool(normalized),
        },
    )
