from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from mignet_ce.graph.builder import LayerGraph
from mignet_ce.mapping import OverlapMapping


def build_lower_graph_matrix(graph: LayerGraph, overlap: OverlapMapping, feature_log1p: bool = True) -> np.ndarray:
    lower_units = overlap.lower_units
    n_upper = len(overlap.upper_units)
    lower_to_row = {unit: idx for idx, unit in enumerate(lower_units)}
    mat = np.zeros((len(lower_units), 2 * n_upper), dtype=float)

    if not graph.intra_edges.empty:
        grouped = graph.intra_edges.groupby("src_unit")["influence_score"].sum()
        for unit, value in grouped.items():
            if unit not in lower_to_row:
                continue
            row = lower_to_row[unit]
            mat[row, :n_upper] += float(value) * overlap.weights[row]

    if not graph.inter_edges.empty:
        tmp = graph.inter_edges[["src_unit", "dst_unit", "influence_score"]].copy()
        tmp = tmp[tmp["src_unit"].isin(lower_to_row) & tmp["dst_unit"].isin(lower_to_row)]
        if not tmp.empty:
            grouped = tmp.groupby(["src_unit", "dst_unit"])["influence_score"].sum().reset_index()
            for row in grouped.itertuples(index=False):
                src_row = lower_to_row[row.src_unit]
                dst_row = lower_to_row[row.dst_unit]
                mat[src_row, n_upper:] += float(row.influence_score) * overlap.weights[dst_row]

    return np.log1p(mat) if feature_log1p else mat


def build_upper_graph_matrix(graph: LayerGraph, upper_units: Sequence[str], feature_log1p: bool = True) -> np.ndarray:
    upper_units = list(map(str, upper_units))
    n_upper = len(upper_units)
    current_units = graph.units
    current_to_row = {unit: idx for idx, unit in enumerate(current_units)}
    upper_to_col = {unit: idx for idx, unit in enumerate(upper_units)}
    mat = np.zeros((len(current_units), 2 * n_upper), dtype=float)

    if not graph.intra_edges.empty:
        grouped = graph.intra_edges.groupby("src_unit")["influence_score"].sum()
        for unit, value in grouped.items():
            if unit in current_to_row and unit in upper_to_col:
                mat[current_to_row[unit], upper_to_col[unit]] += float(value)

    if not graph.inter_edges.empty:
        tmp = graph.inter_edges[["src_unit", "dst_unit", "influence_score"]].copy()
        tmp = tmp[tmp["src_unit"].isin(current_to_row) & tmp["dst_unit"].isin(upper_to_col)]
        if not tmp.empty:
            grouped = tmp.groupby(["src_unit", "dst_unit"])["influence_score"].sum().reset_index()
            for row in grouped.itertuples(index=False):
                mat[current_to_row[row.src_unit], n_upper + upper_to_col[row.dst_unit]] += float(row.influence_score)

    return np.log1p(mat) if feature_log1p else mat


def aggregate_lower_features_to_upper(W_cells: np.ndarray, overlap: OverlapMapping) -> Tuple[np.ndarray, np.ndarray]:
    counts = overlap.counts
    weighted = counts.T @ W_cells
    denom = counts.sum(axis=0)
    feat = np.divide(weighted, denom[:, None], out=np.zeros_like(weighted), where=denom[:, None] > 0)
    return feat, denom


def align_upper_features(W_upper_current: np.ndarray, current_units: Sequence[str], stable_upper_units: Sequence[str]) -> np.ndarray:
    current_units = list(map(str, current_units))
    stable_upper_units = list(map(str, stable_upper_units))
    stable_index = {unit: idx for idx, unit in enumerate(stable_upper_units)}
    aligned = np.zeros((len(stable_upper_units), W_upper_current.shape[1]), dtype=float)
    for src_row, unit in enumerate(current_units):
        if unit in stable_index:
            aligned[stable_index[unit]] = W_upper_current[src_row]
    return aligned


def coverage_table(
    time_point: str,
    upper_units: Sequence[str],
    lower_coverage: np.ndarray,
    upper_present_units: Sequence[str],
) -> pd.DataFrame:
    present = set(map(str, upper_present_units))
    return pd.DataFrame(
        {
            "time_point": time_point,
            "upper_unit": list(map(str, upper_units)),
            "lower_overlap_count": lower_coverage.astype(float),
            "upper_unit_present": [unit in present for unit in map(str, upper_units)],
        }
    )
