from __future__ import annotations

import heapq
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features import coverage_table
from mignet_ce.graph.builder import EDGE_COLUMNS, LayerGraph
from mignet_ce.io.loaders import (
    ExpressionData,
    LayerDataResolver,
    LayerPaths,
    natural_sort,
    peek_h5ad_genes,
    peek_h5ad_units,
    read_commot_index,
    read_commot_manifest,
    read_expression_h5ad,
    read_grn_edges,
)
from mignet_ce.mapping import (
    build_overlap_edge_table,
    build_overlap_mapping,
    build_spot_correspondence_table,
    load_unit_assignments,
    summarize_overlap_quality,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.utils.coords import align_coords


TARGET_AWARE_BLOCK_ORDER = ("grn_target", "cci_out_target", "cci_in_source", "lr_target")


@dataclass
class _LayerFeatureResult:
    matrix: np.ndarray
    graph: LayerGraph
    cci: sp.csr_matrix
    cci_source: str
    mode: str
    grn_self_edges: pd.DataFrame
    cell_edges: pd.DataFrame
    lr_edges: pd.DataFrame


def _minmax_scale(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full(arr.shape, 1.0, dtype=float)
    return floor + (1.0 - floor) * (arr - vmin) / (vmax - vmin)


def _split_complex_genes(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in re.split(r"[_+|;/,]", text) if part]


def _empty_edge_frame(include_stage: bool = True) -> pd.DataFrame:
    columns = EDGE_COLUMNS + (["stage"] if include_stage else [])
    return pd.DataFrame(columns=columns)


def _threshold_sparse(mat: sp.spmatrix, cci_min: float) -> sp.csr_matrix:
    out = mat.tocsr(copy=True).astype(float)
    if cci_min > 0 and out.nnz:
        out.data[out.data < cci_min] = 0.0
        out.eliminate_zeros()
    if out.shape[0] == out.shape[1] and out.shape[0] > 0:
        out.setdiag(0.0)
        out.eliminate_zeros()
    return out


def _align_square_matrix(mat: sp.spmatrix, index_names: Sequence[str], target_units: Sequence[str]) -> sp.csr_matrix:
    target_units = list(map(str, target_units))
    index_lookup = {unit: idx for idx, unit in enumerate(map(str, index_names))}
    present_target_rows: List[int] = []
    present_source_rows: List[int] = []
    for out_idx, unit in enumerate(target_units):
        if unit in index_lookup:
            present_target_rows.append(out_idx)
            present_source_rows.append(index_lookup[unit])
    if not present_target_rows:
        return sp.csr_matrix((len(target_units), len(target_units)), dtype=float)
    sub = mat.tocsr()[present_source_rows, :][:, present_source_rows].tocoo()
    rows = np.asarray([present_target_rows[int(row)] for row in sub.row], dtype=int)
    cols = np.asarray([present_target_rows[int(col)] for col in sub.col], dtype=int)
    return sp.coo_matrix((sub.data, (rows, cols)), shape=(len(target_units), len(target_units)), dtype=float).tocsr()


def _read_cci_matrix(paths: LayerPaths, units: Sequence[str], cci_min: float) -> Tuple[sp.csr_matrix, str]:
    index_names = read_commot_index(paths.cci_index)
    if paths.cci_total.exists():
        mat = sp.load_npz(paths.cci_total)
        if mat.shape[0] != len(index_names) or mat.shape[1] != len(index_names):
            raise ValueError(f"CCI total shape {mat.shape} does not match index length {len(index_names)} for {paths.cci_total}")
        return _threshold_sparse(_align_square_matrix(mat, index_names, units), cci_min), "total"

    manifest = read_commot_manifest(paths.cci_manifest)
    total: sp.csr_matrix | None = None
    for row in manifest.itertuples(index=False):
        matrix_path = paths.cci_lr_dir / str(row.filename)
        lr_mat = sp.load_npz(matrix_path)
        if lr_mat.shape[0] != len(index_names) or lr_mat.shape[1] != len(index_names):
            raise ValueError(f"COMMOT matrix shape {lr_mat.shape} does not match index length {len(index_names)} for {matrix_path}")
        total = lr_mat.tocsr() if total is None else total + lr_mat.tocsr()
    if total is None:
        total = sp.csr_matrix((len(index_names), len(index_names)), dtype=float)
    return _threshold_sparse(_align_square_matrix(total, index_names, units), cci_min), "lr_aggregate"


def _restrict_grn(grn: pd.DataFrame, shared_genes: Sequence[str]) -> pd.DataFrame:
    shared = set(shared_genes)
    out = grn[grn["regulator"].isin(shared) & grn["target"].isin(shared)].copy()
    out["grn_weight_norm"] = _minmax_scale(out["weight"].to_numpy())
    return out


def _intra_strength(expr: pd.DataFrame, grn: pd.DataFrame, expr_threshold: float) -> np.ndarray:
    values = expr.to_numpy(dtype=float)
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.astype(str).tolist())}
    out = np.zeros(values.shape[0], dtype=float)
    for edge in grn.itertuples(index=False):
        reg_idx = gene_to_idx.get(str(edge.regulator))
        target_idx = gene_to_idx.get(str(edge.target))
        if reg_idx is None or target_idx is None:
            continue
        source = values[:, reg_idx]
        target = values[:, target_idx]
        active = (source > expr_threshold) & (target > expr_threshold)
        out += float(edge.grn_weight_norm) * source * target * active.astype(float)
    return out


def _push_top_record(heap: list[tuple[float, int, dict[str, object]]], limit: int, seq: int, score: float, record: dict[str, object]) -> None:
    item = (float(score), seq, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _heap_to_frame(heap: list[tuple[float, int, dict[str, object]]]) -> pd.DataFrame:
    records = [record for _, _, record in sorted(heap, key=lambda item: item[0], reverse=True)]
    return pd.DataFrame(records)


def _build_target_aware_feature_names(stable_upper_units: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    stable_upper_units = list(map(str, stable_upper_units))
    blocks = {
        "grn_target": [f"grntarget_to_{unit}" for unit in stable_upper_units],
        "cci_out_target": [f"cciout_to_{unit}" for unit in stable_upper_units],
        "cci_in_source": [f"cciin_from_{unit}" for unit in stable_upper_units],
        "lr_target": [f"lr_to_{unit}" for unit in stable_upper_units],
    }
    names: List[str] = []
    for block in TARGET_AWARE_BLOCK_ORDER:
        names.extend(blocks[block])
    return names, blocks


def _identity_projection_weights(current_units: Sequence[str], stable_upper_units: Sequence[str]) -> np.ndarray:
    current_units = list(map(str, current_units))
    stable_upper_units = list(map(str, stable_upper_units))
    stable_index = {unit: idx for idx, unit in enumerate(stable_upper_units)}
    weights = np.zeros((len(current_units), len(stable_upper_units)), dtype=float)
    for row, unit in enumerate(current_units):
        col = stable_index.get(unit)
        if col is not None:
            weights[row, col] = 1.0
    return weights


def _project_grn_to_targets(intra_strength: np.ndarray, projection_weights: np.ndarray) -> np.ndarray:
    return np.asarray(intra_strength, dtype=float)[:, None] * np.asarray(projection_weights, dtype=float)


def _project_cci_out_to_targets(cci_matrix: sp.spmatrix, projection_weights: np.ndarray) -> np.ndarray:
    return np.asarray(cci_matrix.tocsr() @ np.asarray(projection_weights, dtype=float), dtype=float)


def _project_cci_in_from_targets(cci_matrix: sp.spmatrix, projection_weights: np.ndarray) -> np.ndarray:
    return np.asarray(cci_matrix.tocsr().T @ np.asarray(projection_weights, dtype=float), dtype=float)


def _edge_table_from_sparse(
    mat: sp.spmatrix,
    units: Sequence[str],
    layer_name: str,
    stage: str,
    edge_type: str,
    limit: int,
) -> pd.DataFrame:
    units = list(map(str, units))
    coo = mat.tocoo()
    heap: list[tuple[float, int, dict[str, object]]] = []
    for seq, (row, col, value) in enumerate(zip(coo.row, coo.col, coo.data)):
        value = float(value)
        if value <= 0:
            continue
        _push_top_record(
            heap,
            limit,
            seq,
            value,
            {
                "src_layer": layer_name,
                "src_unit": units[int(row)],
                "src_gene": np.nan,
                "dst_layer": layer_name,
                "dst_unit": units[int(col)],
                "dst_gene": np.nan,
                "edge_type": edge_type,
                "commot_lr_key": np.nan,
                "commot_ligand": np.nan,
                "commot_receptor": np.nan,
                "grn_weight_raw": np.nan,
                "grn_weight_norm": np.nan,
                "cci_score_raw": value,
                "cci_score_norm": np.nan,
                "distance_raw": np.nan,
                "influence_score": value,
                "stage": stage,
            },
        )
    frame = _heap_to_frame(heap)
    if frame.empty:
        return _empty_edge_frame(include_stage=True)
    return frame.loc[:, EDGE_COLUMNS + ["stage"]]


def _self_loop_edges(values: np.ndarray, units: Sequence[str], layer_name: str, stage: str, edge_type: str) -> pd.DataFrame:
    records = []
    for unit, value in zip(map(str, units), values):
        value = float(value)
        if value <= 0:
            continue
        records.append(
            {
                "src_layer": layer_name,
                "src_unit": unit,
                "src_gene": np.nan,
                "dst_layer": layer_name,
                "dst_unit": unit,
                "dst_gene": np.nan,
                "edge_type": edge_type,
                "commot_lr_key": np.nan,
                "commot_ligand": np.nan,
                "commot_receptor": np.nan,
                "grn_weight_raw": np.nan,
                "grn_weight_norm": np.nan,
                "cci_score_raw": np.nan,
                "cci_score_norm": np.nan,
                "distance_raw": np.nan,
                "influence_score": value,
                "stage": stage,
            }
        )
    if not records:
        return _empty_edge_frame(include_stage=True)
    return pd.DataFrame.from_records(records).loc[:, EDGE_COLUMNS + ["stage"]]


def _project_lr_to_targets(
    layer_name: str,
    stage: str,
    expr: pd.DataFrame,
    paths: LayerPaths,
    projection_weights: np.ndarray,
    grn: pd.DataFrame,
    cfg: TemporalRunConfig,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    n_units, n_targets = projection_weights.shape
    lr_features = np.zeros((n_units, n_targets), dtype=float)
    if not paths.cci_manifest.exists() or not paths.cci_lr_dir.exists():
        return lr_features, _empty_edge_frame(include_stage=True), "lr_unavailable"

    manifest = read_commot_manifest(paths.cci_manifest)
    index_names = read_commot_index(paths.cci_index)
    unit_names = expr.index.astype(str).tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.astype(str).tolist())}
    grn_lookup = {
        (str(row.regulator), str(row.target)): float(row.grn_weight_norm)
        for row in grn.itertuples(index=False)
    }
    expr_values = expr.to_numpy(dtype=float)
    heap: list[tuple[float, int, dict[str, object]]] = []
    seq = 0

    for row in manifest.itertuples(index=False):
        ligand_genes = [gene for gene in _split_complex_genes(row.ligand) if gene in gene_to_idx]
        receptor_genes = [gene for gene in _split_complex_genes(row.receptor) if gene in gene_to_idx]
        if not ligand_genes or not receptor_genes:
            continue

        matrix_path = paths.cci_lr_dir / str(row.filename)
        mat = sp.load_npz(matrix_path)
        if mat.shape[0] != len(index_names) or mat.shape[1] != len(index_names):
            raise ValueError(f"COMMOT matrix shape {mat.shape} does not match index length {len(index_names)} for {matrix_path}")
        aligned = _threshold_sparse(_align_square_matrix(mat, index_names, unit_names), cfg.cci_min).tocoo()
        if aligned.nnz == 0:
            continue

        rows = np.asarray(aligned.row, dtype=int)
        cols = np.asarray(aligned.col, dtype=int)
        cci_values = np.asarray(aligned.data, dtype=float)
        for ligand in ligand_genes:
            ligand_idx = gene_to_idx[ligand]
            ligand_expr = expr_values[rows, ligand_idx]
            for receptor in receptor_genes:
                receptor_idx = gene_to_idx[receptor]
                receptor_expr = expr_values[cols, receptor_idx]
                scores = cci_values * ligand_expr * receptor_expr
                if cfg.require_target_expression_for_inter:
                    scores = scores * (receptor_expr > cfg.expr_threshold)
                grn_weight = np.nan
                if cfg.cross_cell_lr_use_grn_gate:
                    grn_weight = grn_lookup.get((ligand, receptor), 0.0)
                    if grn_weight <= 0:
                        continue
                    scores = scores * float(grn_weight)
                keep = np.flatnonzero(scores > 0)
                if keep.size == 0:
                    continue

                for src_idx in np.unique(rows[keep]):
                    src_keep = keep[rows[keep] == src_idx]
                    if src_keep.size == 0:
                        continue
                    weighted_projection = scores[src_keep, None] * projection_weights[cols[src_keep], :]
                    lr_features[int(src_idx), :] += weighted_projection.sum(axis=0)

                for idx in keep:
                    score = float(scores[idx])
                    seq += 1
                    _push_top_record(
                        heap,
                        cfg.cross_cell_top_k_edges,
                        seq,
                        score,
                        {
                            "src_layer": layer_name,
                            "src_unit": unit_names[int(rows[idx])],
                            "src_gene": ligand,
                            "dst_layer": layer_name,
                            "dst_unit": unit_names[int(cols[idx])],
                            "dst_gene": receptor,
                            "edge_type": "target_aware_lr",
                            "commot_lr_key": str(getattr(row, "lr_key", f"{row.ligand}-{row.receptor}")),
                            "commot_ligand": str(row.ligand),
                            "commot_receptor": str(row.receptor),
                            "grn_weight_raw": np.nan,
                            "grn_weight_norm": grn_weight,
                            "cci_score_raw": float(cci_values[idx]),
                            "cci_score_norm": np.nan,
                            "distance_raw": np.nan,
                            "influence_score": score,
                            "stage": stage,
                        },
                    )

    frame = _heap_to_frame(heap)
    if frame.empty:
        frame = _empty_edge_frame(include_stage=True)
    else:
        frame = frame.loc[:, EDGE_COLUMNS + ["stage"]]
    mode = "lr_grn_gate" if cfg.cross_cell_lr_use_grn_gate else "lr_no_grn_gate"
    return lr_features, frame, mode


class CrossCellMultilayerBuilder:
    network_method = "cross_cell_multilayer"

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        all_paths = self._check_pair_paths(cfg, resolver, organ, pair)
        shared_genes = self._compute_shared_genes(cfg, all_paths, pair)
        stable_upper_units = self._stable_upper_units(cfg, all_paths, pair.upper_layer)
        feature_names, feature_blocks = _build_target_aware_feature_names(stable_upper_units)

        lower_mats: List[np.ndarray] = []
        upper_mats: List[np.ndarray] = []
        overlaps = []
        lower_units_by_time: List[List[str]] = []
        upper_units_by_time: List[List[str]] = []
        lower_assignments_by_time: List[pd.DataFrame] = []
        upper_assignments_by_time: List[pd.DataFrame] = []
        coverage_tables: List[pd.DataFrame] = []
        spot_correspondence_tables: List[pd.DataFrame] = []
        overlap_edge_tables: List[pd.DataFrame] = []
        overlap_quality_summaries: List[dict[str, object]] = []
        graph_summaries: List[dict[str, object]] = []
        lower_graphs: List[LayerGraph] = []
        upper_graphs: List[LayerGraph] = []
        upper_coords_by_time: List[np.ndarray] = []
        exports: dict[str, pd.DataFrame] = {}
        stage_metadata: List[dict[str, object]] = []

        for stage in map(str, cfg.time_points):
            lower_paths = all_paths[(stage, pair.lower_layer)]
            upper_paths = all_paths[(stage, pair.upper_layer)]
            lower_expr = read_expression_h5ad(lower_paths.h5ad)
            upper_expr = read_expression_h5ad(upper_paths.h5ad)

            lower_assignments = load_unit_assignments(pair.lower_layer, lower_expr, lower_paths.spot_domain_map)
            upper_assignments = load_unit_assignments(pair.upper_layer, upper_expr, upper_paths.spot_domain_map)
            overlap = build_overlap_mapping(
                lower=lower_assignments,
                upper=upper_assignments,
                lower_units=lower_expr.units,
                upper_units=stable_upper_units,
            )
            spot_correspondence = build_spot_correspondence_table(
                lower=lower_assignments,
                upper=upper_assignments,
                stage=stage,
                lower_layer=pair.lower_layer,
                upper_layer=pair.upper_layer,
            )
            overlap_edges = build_overlap_edge_table(overlap, stage, pair.lower_layer, pair.upper_layer)
            overlap_quality = summarize_overlap_quality(overlap_edges)
            overlap_quality["stage"] = stage

            lower_projection = overlap.weights
            upper_projection = _identity_projection_weights(upper_expr.units, stable_upper_units)
            lower_result = self._build_layer_features(
                pair.lower_layer,
                stage,
                lower_expr,
                lower_paths,
                shared_genes,
                lower_projection,
                stable_upper_units,
                cfg,
            )
            upper_result = self._build_layer_features(
                pair.upper_layer,
                stage,
                upper_expr,
                upper_paths,
                shared_genes,
                upper_projection,
                stable_upper_units,
                cfg,
            )

            if cfg.feature_log1p:
                lower_mat = np.log1p(lower_result.matrix)
                upper_mat = np.log1p(upper_result.matrix)
            else:
                lower_mat = lower_result.matrix
                upper_mat = upper_result.matrix

            lower_mats.append(lower_mat)
            upper_mats.append(upper_mat)
            overlaps.append(overlap)
            lower_units_by_time.append(lower_expr.units)
            upper_units_by_time.append(upper_expr.units)
            lower_assignments_by_time.append(lower_assignments.rows.copy())
            upper_assignments_by_time.append(upper_assignments.rows.copy())
            lower_graphs.append(lower_result.graph)
            upper_graphs.append(upper_result.graph)
            upper_coords_by_time.append(align_coords(upper_expr.coords, stable_upper_units))
            coverage_tables.append(coverage_table(stage, stable_upper_units, overlap.coverage_counts(), upper_expr.units))
            spot_correspondence_tables.append(spot_correspondence)
            overlap_edge_tables.append(overlap_edges)
            overlap_quality_summaries.append(overlap_quality)
            graph_summaries.append(self._graph_summary(stage, lower_result, upper_result, lower_mat, upper_mat, feature_blocks))
            stage_metadata.append(
                {
                    "stage": stage,
                    "lower_cci_source": lower_result.cci_source,
                    "upper_cci_source": upper_result.cci_source,
                    "lower_mode": lower_result.mode,
                    "upper_mode": upper_result.mode,
                    "feature_dim": int(len(feature_names)),
                    "stable_upper_unit_count": int(len(stable_upper_units)),
                    "ddi_handling": "disabled_in_target_aware_mode",
                    "lr_grn_gate": bool(cfg.cross_cell_lr_use_grn_gate),
                }
            )

            exports[f"network_exports/{stage}_lower_target_aware_lr_edges_topk.csv"] = lower_result.lr_edges
            exports[f"network_exports/{stage}_upper_target_aware_lr_edges_topk.csv"] = upper_result.lr_edges
            exports[f"network_exports/{stage}_lower_cell_comm_edges_topk.csv"] = lower_result.cell_edges
            exports[f"network_exports/{stage}_upper_cell_comm_edges_topk.csv"] = upper_result.cell_edges
            exports[f"network_exports/{stage}_lower_grn_self_edges.csv"] = lower_result.grn_self_edges
            exports[f"network_exports/{stage}_upper_grn_self_edges.csv"] = upper_result.grn_self_edges

        return NetworkContext(
            organ=organ,
            pair=pair,
            time_points=list(map(str, cfg.time_points)),
            network_method=self.network_method,
            stable_upper_units=stable_upper_units,
            shared_genes=shared_genes,
            lower_mats=lower_mats,
            upper_mats=upper_mats,
            overlaps=overlaps,
            lower_units_by_time=lower_units_by_time,
            upper_units_by_time=upper_units_by_time,
            upper_coords_by_time=upper_coords_by_time,
            feature_names=feature_names,
            feature_blocks={key: list(value) for key, value in feature_blocks.items()},
            graph_summaries=graph_summaries,
            exports=exports,
            metadata={
                "network_method": self.network_method,
                "cross_cell_feature_mode": "target_aware_multichannel",
                "feature_alignment_space": "stable_upper_units",
                "feature_dim": int(len(feature_names)),
                "feature_block_count": len(feature_blocks),
                "feature_blocks": {key: list(value) for key, value in feature_blocks.items()},
                "ddi_handling": "disabled_in_target_aware_mode",
                "lr_grn_gate": bool(cfg.cross_cell_lr_use_grn_gate),
                "stages": stage_metadata,
            },
            lower_assignments_by_time=lower_assignments_by_time,
            upper_assignments_by_time=upper_assignments_by_time,
            lower_graphs=lower_graphs,
            upper_graphs=upper_graphs,
            coverage_tables=coverage_tables,
            spot_correspondence_tables=spot_correspondence_tables,
            overlap_edge_tables=overlap_edge_tables,
            overlap_quality_summaries=overlap_quality_summaries,
        )

    def _build_layer_features(
        self,
        layer_name: str,
        stage: str,
        expression: ExpressionData,
        paths: LayerPaths,
        shared_genes: Sequence[str],
        projection_weights: np.ndarray,
        stable_upper_units: Sequence[str],
        cfg: TemporalRunConfig,
    ) -> _LayerFeatureResult:
        expr = expression.expr.loc[:, list(shared_genes)].copy()
        units = expr.index.astype(str).tolist()
        projection_weights = np.asarray(projection_weights, dtype=float)
        if projection_weights.shape != (expr.shape[0], len(stable_upper_units)):
            raise ValueError(
                f"Projection matrix shape {projection_weights.shape} does not match "
                f"{layer_name} units x stable upper units {(expr.shape[0], len(stable_upper_units))}."
            )

        grn = _restrict_grn(read_grn_edges(paths.grn_edges, cfg.top_k_targets_per_regulator), shared_genes)
        cci, cci_source = _read_cci_matrix(paths, units, cfg.cci_min)
        intra = _intra_strength(expr, grn, cfg.expr_threshold)
        grn_block = _project_grn_to_targets(intra, projection_weights)
        cci_out_block = _project_cci_out_to_targets(cci, projection_weights)
        cci_in_block = _project_cci_in_from_targets(cci, projection_weights)
        lr_block, lr_edges, lr_mode = _project_lr_to_targets(layer_name, stage, expr, paths, projection_weights, grn, cfg)
        matrix = np.hstack([grn_block, cci_out_block, cci_in_block, lr_block])

        grn_self_edges = _self_loop_edges(intra, units, layer_name, stage, "grn_self")
        cell_edges = _edge_table_from_sparse(cci, units, layer_name, stage, "cell_communication", cfg.cross_cell_top_k_edges)
        graph = LayerGraph(
            layer=layer_name,
            time_point=stage,
            units=units,
            genes=list(shared_genes),
            intra_edges=grn_self_edges.loc[:, EDGE_COLUMNS].copy(),
            inter_edges=cell_edges.loc[:, EDGE_COLUMNS].copy(),
            shared_genes=list(shared_genes),
        )
        return _LayerFeatureResult(
            matrix=matrix,
            graph=graph,
            cci=cci,
            cci_source=cci_source,
            mode=lr_mode,
            grn_self_edges=grn_self_edges,
            cell_edges=cell_edges,
            lr_edges=lr_edges,
        )

    def _required_paths(self, paths: LayerPaths) -> List[Path]:
        required = [paths.h5ad, paths.grn_edges, paths.cci_manifest, paths.cci_index, paths.cci_lr_dir]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        return required

    def _check_pair_paths(
        self,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
        organ: str,
        pair: VerticalPairSpec,
    ) -> Dict[Tuple[str, str], LayerPaths]:
        all_paths: Dict[Tuple[str, str], LayerPaths] = {}
        missing: List[str] = []
        for stage in cfg.time_points:
            for layer in (pair.lower_layer, pair.upper_layer):
                paths = resolver.paths(layer, organ, str(stage))
                all_paths[(str(stage), layer)] = paths
                for required in self._required_paths(paths):
                    if not required.exists():
                        missing.append(str(required))
        if missing:
            preview = "\n".join(missing[:20])
            extra = f"\n... {len(missing) - 20} more" if len(missing) > 20 else ""
            raise FileNotFoundError(f"Missing required inputs for {organ} {pair.label()}:\n{preview}{extra}")
        return all_paths

    def _compute_shared_genes(
        self,
        cfg: TemporalRunConfig,
        all_paths: Dict[Tuple[str, str], LayerPaths],
        pair: VerticalPairSpec,
    ) -> List[str]:
        intersections: List[set[str]] = []
        for stage in cfg.time_points:
            lower_paths = all_paths[(str(stage), pair.lower_layer)]
            upper_paths = all_paths[(str(stage), pair.upper_layer)]
            lower_expr_genes = set(peek_h5ad_genes(lower_paths.h5ad))
            upper_expr_genes = set(peek_h5ad_genes(upper_paths.h5ad))
            intersections.append(lower_expr_genes & upper_expr_genes)
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError(f"Shared expression gene intersection is empty for {pair.label()}.")
        return shared

    def _stable_upper_units(
        self,
        cfg: TemporalRunConfig,
        all_paths: Dict[Tuple[str, str], LayerPaths],
        upper_layer: str,
    ) -> List[str]:
        units = set()
        for stage in cfg.time_points:
            units.update(peek_h5ad_units(all_paths[(str(stage), upper_layer)].h5ad))
        stable = natural_sort(units)
        if not stable:
            raise ValueError(f"No upper units found for {upper_layer}.")
        return stable

    @staticmethod
    def _graph_summary(
        stage: str,
        lower_result: _LayerFeatureResult,
        upper_result: _LayerFeatureResult,
        lower_mat: np.ndarray,
        upper_mat: np.ndarray,
        feature_blocks: Dict[str, List[str]],
    ) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": CrossCellMultilayerBuilder.network_method,
            "cross_cell_feature_mode": "target_aware_multichannel",
            "feature_blocks": {key: list(value) for key, value in feature_blocks.items()},
            "lower_units": len(lower_result.graph.units),
            "upper_units": len(upper_result.graph.units),
            "shared_genes": len(lower_result.graph.shared_genes),
            "lower_grn_self_edges": int(len(lower_result.grn_self_edges)),
            "upper_grn_self_edges": int(len(upper_result.grn_self_edges)),
            "lower_cell_comm_edges_topk": int(len(lower_result.cell_edges)),
            "upper_cell_comm_edges_topk": int(len(upper_result.cell_edges)),
            "lower_lr_edges_topk": int(len(lower_result.lr_edges)),
            "upper_lr_edges_topk": int(len(upper_result.lr_edges)),
            "ddi_edges_topk": 0,
            "ddi_handling": "disabled_in_target_aware_mode",
            "lower_matrix_shape": list(lower_mat.shape),
            "upper_matrix_shape": list(upper_mat.shape),
        }
