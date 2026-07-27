from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.io.loaders import LayerDataResolver, read_expression_h5ad, read_grn_edges
from mignet_ce.networks.base import NetworkContext


@dataclass(frozen=True)
class RawGRNInputs:
    expression: pd.DataFrame
    grn_edges: pd.DataFrame
    units: tuple[str, ...]
    h5ad_path: Path
    grn_path: Path


def side_layer(context: NetworkContext, side: str) -> str:
    if side == "lower":
        return context.pair.lower_layer
    if side == "upper":
        return context.pair.upper_layer
    raise ValueError("side must be 'lower' or 'upper'.")


def native_side_units(context: NetworkContext, side: str, time_index: int) -> list[str]:
    if side == "lower":
        return list(map(str, context.lower_units_by_time[time_index]))
    if side == "upper":
        return list(map(str, context.upper_units_by_time[time_index]))
    raise ValueError("side must be 'lower' or 'upper'.")


def output_side_units(context: NetworkContext, side: str, time_index: int) -> list[str]:
    if context.feature_alignment_space == "native_units":
        return native_side_units(context, side, time_index)
    if side in {"lower", "upper"}:
        return list(map(str, context.stable_upper_units))
    raise ValueError("side must be 'lower' or 'upper'.")


def _side_graph(context: NetworkContext, side: str, time_index: int):
    graphs = context.lower_graphs if side == "lower" else context.upper_graphs
    if len(graphs) <= time_index:
        raise ValueError(f"Missing {side} LayerGraph for time index {time_index}.")
    return graphs[time_index]


def _align_square(matrix: sp.spmatrix, source_units: Sequence[str], target_units: Sequence[str]) -> sp.csr_matrix:
    source_lookup = {str(unit): index for index, unit in enumerate(source_units)}
    source_indices: list[int] = []
    target_indices: list[int] = []
    for target_index, unit in enumerate(map(str, target_units)):
        source_index = source_lookup.get(unit)
        if source_index is not None:
            source_indices.append(source_index)
            target_indices.append(target_index)
    if not source_indices:
        return sp.csr_matrix((len(target_units), len(target_units)), dtype=float)
    coo = matrix.tocsr()[source_indices, :][:, source_indices].tocoo()
    rows = np.asarray([target_indices[int(index)] for index in coo.row], dtype=int)
    cols = np.asarray([target_indices[int(index)] for index in coo.col], dtype=int)
    return sp.coo_matrix((coo.data, (rows, cols)), shape=(len(target_units), len(target_units))).tocsr()


def load_cci_adjacency(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
    time_index: int,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    graph = _side_graph(context, side, time_index)
    stored = graph.metadata.get("adjacency_csr")
    if stored is None:
        path_text = graph.metadata.get("adjacency_path")
        if path_text is None:
            raise ValueError(f"{side} graph metadata has neither adjacency_csr nor adjacency_path.")
        stored = sp.load_npz(Path(str(path_text)))
    matrix = sp.csr_matrix(stored).astype(float)
    units = native_side_units(context, side, time_index)
    aligned = _align_square(matrix, graph.units, units)
    if aligned.nnz:
        if not np.all(np.isfinite(aligned.data)):
            raise ValueError("CCI adjacency contains non-finite values.")
        aligned.data = np.maximum(aligned.data, 0.0)
        if cfg.cci_min > 0.0:
            aligned.data[aligned.data < float(cfg.cci_min)] = 0.0
            aligned.eliminate_zeros()
    return aligned, {
        "source": "NetworkContext.LayerGraph.metadata.adjacency_csr",
        "path": graph.metadata.get("adjacency_path"),
        "side": side,
        "stage": str(context.time_points[time_index]),
        "layer": side_layer(context, side),
        "shape": list(aligned.shape),
        "nnz": int(aligned.nnz),
        "cci_min": float(cfg.cci_min),
        "read_only": True,
    }


def align_feature_to_context(
    values: np.ndarray,
    context: NetworkContext,
    side: str,
    time_index: int,
) -> tuple[np.ndarray, dict[str, object]]:
    arr = np.asarray(values, dtype=float)
    expected_rows = len(native_side_units(context, side, time_index))
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise ValueError(
            f"{side} feature rows at time index {time_index} have shape {arr.shape}; expected {expected_rows} rows."
        )
    if context.feature_alignment_space == "native_units":
        return arr, {"aligned_to": "native_units", "time_index": int(time_index)}
    if side == "lower":
        aligned, coverage = aggregate_lower_features_to_upper(arr, context.overlaps[time_index])
        return aligned, {
            "aligned_to": "stable_upper_units",
            "lower_aggregation": "overlap_weighted_average",
            "covered_units": int(np.count_nonzero(coverage > 0.0)),
            "time_index": int(time_index),
        }
    aligned = align_upper_features(arr, context.upper_units_by_time[time_index], context.stable_upper_units)
    return aligned, {
        "aligned_to": "stable_upper_units",
        "upper_alignment": "zero_fill_missing_units",
        "time_index": int(time_index),
    }


def load_merged_grn_state(
    context: NetworkContext,
    side: str,
    time_index: int,
) -> tuple[np.ndarray, dict[str, object]]:
    graph = _side_graph(context, side, time_index)
    stored = graph.metadata.get("grn_state_csr")
    if stored is None:
        raise ValueError(f"{side} graph metadata is missing the frozen merged grn_state_csr payload.")
    raw = sp.csr_matrix(stored).toarray().astype(float, copy=False)
    aligned, alignment = align_feature_to_context(raw, context, side, time_index)
    if not np.all(np.isfinite(aligned)):
        raise ValueError("Merged GRN state contains non-finite values.")
    return aligned, {
        "source": "frozen_light_cci_grn_merged_grn_state_csr",
        "definition": "regulator_projection_plus_target_projection",
        "shape": list(aligned.shape),
        "alignment": alignment,
        "read_only": True,
    }


def load_raw_grn_inputs(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
    time_index: int,
) -> RawGRNInputs:
    layer = side_layer(context, side)
    if layer == "gene":
        raise ValueError("Feature-version GRN recomputation is only defined for non-gene layers.")
    stage = str(context.time_points[time_index])
    paths = LayerDataResolver(cfg.data_root).paths(layer, context.organ, stage)
    if not paths.h5ad.exists():
        raise FileNotFoundError(f"Expression input is missing: {paths.h5ad}")
    if not paths.grn_edges.exists():
        raise FileNotFoundError(f"GRN edge input is missing: {paths.grn_edges}")
    expression = read_expression_h5ad(paths.h5ad).expr
    units = tuple(native_side_units(context, side, time_index))
    edges = read_grn_edges(paths.grn_edges, top_k_targets_per_regulator=None)
    return RawGRNInputs(
        expression=expression,
        grn_edges=edges,
        units=units,
        h5ad_path=paths.h5ad,
        grn_path=paths.grn_edges,
    )


def standardize_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError("Pairwise standardization requires matching feature dimensions.")
    stacked = np.vstack([source_arr, target_arr])
    means = stacked.mean(axis=0, keepdims=True)
    stds = stacked.std(axis=0, keepdims=True)
    safe = np.where(stds > 0.0, stds, 1.0)
    return (source_arr - means) / safe, (target_arr - means) / safe, {
        "mode": "pairwise_joint_zscore",
        "zero_variance_columns": int(np.count_nonzero(stds <= 0.0)),
    }
