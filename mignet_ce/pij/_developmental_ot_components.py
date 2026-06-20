from __future__ import annotations

from typing import Callable

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.developmental_features import (
    DevelopmentalFeatureTable,
    load_developmental_features_for_pij,
    require_columns,
    select_first_available_column,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.transition.cost_components import (
    pairwise_feature_cost,
    pairwise_scalar_cost,
    pairwise_spatial_cost,
)


DevelopmentalTableLoader = Callable[[str, int], DevelopmentalFeatureTable]


def make_developmental_table_loader(
    context: NetworkContext,
    cfg: TemporalRunConfig,
) -> DevelopmentalTableLoader:
    cache: dict[tuple[str, int], DevelopmentalFeatureTable] = {}

    def load(space: str, time_index: int) -> DevelopmentalFeatureTable:
        key = (space, time_index)
        if key not in cache:
            cache[key] = load_developmental_features_for_pij(context, cfg, time_index, space)
        return cache[key]

    return load


def build_sr_cost(
    source_table: DevelopmentalFeatureTable,
    target_table: DevelopmentalFeatureTable,
    method_name: str,
) -> tuple[str, np.ndarray, str]:
    column = select_first_available_column(source_table.values, ["sr", "potency_score"], method_name)
    require_columns(target_table.values, [column], method_name)
    source_values = source_table.values[column].to_numpy(dtype=float)
    target_values = target_table.values[column].to_numpy(dtype=float)
    component_name = "sr" if column == "sr" else "potency"
    return component_name, pairwise_scalar_cost(source_values, target_values), column


def build_pseudotime_cost(
    source_table: DevelopmentalFeatureTable,
    target_table: DevelopmentalFeatureTable,
    method_name: str,
) -> np.ndarray:
    require_columns(source_table.values, ["pseudotime"], method_name)
    require_columns(target_table.values, ["pseudotime"], method_name)
    source_values = source_table.values["pseudotime"].to_numpy(dtype=float)
    target_values = target_table.values["pseudotime"].to_numpy(dtype=float)
    return pairwise_scalar_cost(source_values, target_values)


def build_spatial_cost(
    source_coords: np.ndarray | None,
    target_coords: np.ndarray | None,
    method_name: str,
) -> np.ndarray:
    if source_coords is None or target_coords is None:
        raise ValueError(f"{method_name} requires coordinates.")
    return pairwise_spatial_cost(source_coords, target_coords)


def build_expression_cost(
    source_features: np.ndarray,
    target_features: np.ndarray,
    cfg: TemporalRunConfig,
) -> np.ndarray:
    return pairwise_feature_cost(source_features, target_features, metric=cfg.pij_cost_metric)


def build_graph_energy_cost(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    return pairwise_scalar_cost(
        np.linalg.norm(source_features, axis=1),
        np.linalg.norm(target_features, axis=1),
    )


def developmental_metadata(
    source_table: DevelopmentalFeatureTable,
    target_table: DevelopmentalFeatureTable,
    used_columns: list[str],
    cfg: TemporalRunConfig,
) -> dict[str, object]:
    return {
        "feature_columns_used": list(used_columns),
        "feature_aggregation": cfg.pij_feature_aggregation,
        "missing_feature_policy": cfg.pij_missing_feature_policy,
        "developmental_features": {
            "source": source_table.metadata,
            "target": target_table.metadata,
        },
    }
