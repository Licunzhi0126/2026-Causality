from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.mapping import OverlapMapping


@dataclass
class NetworkContext:
    organ: str
    pair: VerticalPairSpec
    time_points: List[str]
    network_method: str
    stable_upper_units: List[str]
    shared_genes: List[str]
    lower_mats: List[np.ndarray]
    upper_mats: List[np.ndarray]
    overlaps: List[OverlapMapping]
    lower_units_by_time: List[List[str]]
    upper_units_by_time: List[List[str]]
    upper_coords_by_time: List[np.ndarray]
    feature_names: List[str]
    feature_blocks: Dict[str, List[str]]
    graph_summaries: List[dict[str, object]]
    lower_coords_by_time: List[np.ndarray] = field(default_factory=list)
    feature_alignment_space: str = "stable_upper_units"
    exports: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    lower_assignments_by_time: List[pd.DataFrame] = field(default_factory=list)
    upper_assignments_by_time: List[pd.DataFrame] = field(default_factory=list)
    lower_graphs: List[LayerGraph] = field(default_factory=list)
    upper_graphs: List[LayerGraph] = field(default_factory=list)
    coverage_tables: List[pd.DataFrame] = field(default_factory=list)
    spot_correspondence_tables: List[pd.DataFrame] = field(default_factory=list)
    overlap_edge_tables: List[pd.DataFrame] = field(default_factory=list)
    overlap_quality_summaries: List[dict[str, object]] = field(default_factory=list)


class NetworkBuilder(Protocol):
    network_method: str

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        ...
