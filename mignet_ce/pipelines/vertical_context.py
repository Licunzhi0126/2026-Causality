from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from mignet_ce.config import VerticalPairSpec
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.mapping import OverlapMapping


@dataclass
class VerticalPairContext:
    organ: str
    pair: VerticalPairSpec
    time_points: List[str]
    stable_upper_units: List[str]
    shared_genes: List[str]
    lower_mats: List[np.ndarray]
    upper_mats: List[np.ndarray]
    overlaps: List[OverlapMapping]
    lower_units_by_time: List[List[str]]
    upper_units_by_time: List[List[str]]
    lower_assignments_by_time: List[pd.DataFrame]
    upper_assignments_by_time: List[pd.DataFrame]
    lower_graphs: List[LayerGraph]
    upper_graphs: List[LayerGraph]
    upper_coords_by_time: List[np.ndarray]
    coverage_tables: List[pd.DataFrame]
    spot_correspondence_tables: List[pd.DataFrame]
    overlap_edge_tables: List[pd.DataFrame]
    overlap_quality_summaries: List[dict[str, object]]
    graph_summaries: List[dict[str, object]]
