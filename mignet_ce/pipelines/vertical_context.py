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
    upper_units_by_time: List[List[str]]
    lower_graphs: List[LayerGraph]
    upper_graphs: List[LayerGraph]
    upper_coords_by_time: List[np.ndarray]
    coverage_tables: List[pd.DataFrame]
    graph_summaries: List[dict[str, object]]
