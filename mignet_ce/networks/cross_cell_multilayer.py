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


MICRO_FEATURES = [
    "micro_intra_strength",
    "micro_cross_out_strength",
    "micro_cross_in_strength",
    "micro_cross_total_strength",
]
CELL_COMM_FEATURES = [
    "cell_comm_out_strength",
    "cell_comm_in_strength",
    "cell_comm_total_strength",
    "cell_comm_out_entropy",
    "cell_comm_in_entropy",
    "cell_comm_topk_out_strength",
    "cell_comm_topk_in_strength",
]
MACRO_DDI_FEATURES = [
    "macro_ddi_out_strength",
    "macro_ddi_in_strength",
    "macro_ddi_total_strength",
]
FEATURE_NAMES = MICRO_FEATURES + CELL_COMM_FEATURES + MACRO_DDI_FEATURES
FEATURE_BLOCKS = {
    "micro_regulatory": MICRO_FEATURES,
    "cell_communication": CELL_COMM_FEATURES,
    "macro_ddi": MACRO_DDI_FEATURES,
}


@dataclass
class _LayerFeatureResult:
    matrix: np.ndarray
    graph: LayerGraph
    cci: sp.csr_matrix
    cci_source: str
    mode: str
    micro_edges: pd.DataFrame
    cell_edges: pd.DataFrame


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


def _row_entropy(mat: sp.spmatrix) -> np.ndarray:
    csr = mat.tocsr()
    out = np.zeros(csr.shape[0], dtype=float)
    for row in range(csr.shape[0]):
        data = csr.data[csr.indptr[row] : csr.indptr[row + 1]]
        data = data[data > 0]
        total = float(data.sum())
        if total <= 0:
            continue
        prob = data / total
        out[row] = float(-np.sum(prob * np.log(prob + 1e-12)))
    return out


def _row_topk_sum(mat: sp.spmatrix, k: int) -> np.ndarray:
    csr = mat.tocsr()
    out = np.zeros(csr.shape[0], dtype=float)
    for row in range(csr.shape[0]):
        data = csr.data[csr.indptr[row] : csr.indptr[row + 1]]
        data = data[data > 0]
        if data.size == 0:
            continue
        keep = min(int(k), data.size)
        out[row] = float(np.partition(data, data.size - keep)[data.size - keep :].sum())
    return out


def _communication_features(mat: sp.spmatrix, top_k: int) -> np.ndarray:
    csr = mat.tocsr()
    out_strength = np.asarray(csr.sum(axis=1)).ravel()
    in_strength = np.asarray(csr.sum(axis=0)).ravel()
    return np.column_stack(
        [
            out_strength,
            in_strength,
            out_strength + in_strength,
            _row_entropy(csr),
            _row_entropy(csr.T),
            _row_topk_sum(csr, top_k),
            _row_topk_sum(csr.T, top_k),
        ]
    )


def _macro_features(mat: sp.spmatrix) -> np.ndarray:
    csr = mat.tocsr()
    out_strength = np.asarray(csr.sum(axis=1)).ravel()
    in_strength = np.asarray(csr.sum(axis=0)).ravel()
    return np.column_stack([out_strength, in_strength, out_strength + in_strength])


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


def _micro_cross_features(
    layer_name: str,
    stage: str,
    expr: pd.DataFrame,
    paths: LayerPaths,
    grn: pd.DataFrame,
    cfg: TemporalRunConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    n_units = expr.shape[0]
    cross_out = np.zeros(n_units, dtype=float)
    cross_in = np.zeros(n_units, dtype=float)
    if not paths.cci_manifest.exists() or not paths.cci_lr_dir.exists():
        return cross_out, cross_in, pd.DataFrame(), "lr_unavailable_intra_only"

    manifest = read_commot_manifest(paths.cci_manifest)
    index_names = read_commot_index(paths.cci_index)
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.astype(str).tolist())}
    pair_lookup = {
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
        usable_pairs = [
            (ligand, receptor, pair_lookup[(ligand, receptor)])
            for ligand in ligand_genes
            for receptor in receptor_genes
            if (ligand, receptor) in pair_lookup
        ]
        if not usable_pairs:
            continue
        matrix_path = paths.cci_lr_dir / str(row.filename)
        mat = sp.load_npz(matrix_path)
        aligned = _threshold_sparse(_align_square_matrix(mat, index_names, expr.index.astype(str).tolist()), cfg.cci_min).tocoo()
        if aligned.nnz == 0:
            continue
        rows = np.asarray(aligned.row, dtype=int)
        cols = np.asarray(aligned.col, dtype=int)
        cci_values = np.asarray(aligned.data, dtype=float)
        for ligand, receptor, grn_weight_norm in usable_pairs:
            ligand_idx = gene_to_idx[ligand]
            receptor_idx = gene_to_idx[receptor]
            scores = cci_values * grn_weight_norm * expr_values[rows, ligand_idx] * expr_values[cols, receptor_idx]
            if cfg.require_target_expression_for_inter:
                scores = scores * (expr_values[cols, receptor_idx] > cfg.expr_threshold)
            keep = np.flatnonzero(scores > 0)
            if keep.size == 0:
                continue
            cross_out += np.bincount(rows[keep], weights=scores[keep], minlength=n_units)
            cross_in += np.bincount(cols[keep], weights=scores[keep], minlength=n_units)
            for idx in keep:
                score = float(scores[idx])
                seq += 1
                _push_top_record(
                    heap,
                    cfg.cross_cell_top_k_edges,
                    seq,
                    score,
                    {
                        "stage": stage,
                        "layer": layer_name,
                        "src_unit": str(expr.index[int(rows[idx])]),
                        "dst_unit": str(expr.index[int(cols[idx])]),
                        "ligand": ligand,
                        "receptor": receptor,
                        "lr_key": str(getattr(row, "lr_key", f"{row.ligand}-{row.receptor}")),
                        "cci_score_raw": float(cci_values[idx]),
                        "grn_weight_norm": float(grn_weight_norm),
                        "influence_score": score,
                    },
                )
    return cross_out, cross_in, _heap_to_frame(heap), "lr_grn_fallback"


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
        return pd.DataFrame(columns=EDGE_COLUMNS + ["stage"])
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
        return pd.DataFrame(columns=EDGE_COLUMNS + ["stage"])
    return pd.DataFrame.from_records(records).loc[:, EDGE_COLUMNS + ["stage"]]


def _coarse_grain_cci(lower_cci: sp.spmatrix, overlap_weights: np.ndarray) -> sp.csr_matrix:
    weights = sp.csr_matrix(overlap_weights)
    return (weights.T @ lower_cci.tocsr() @ weights).tocsr()


def _select_rows(features: np.ndarray, source_units: Sequence[str], target_units: Sequence[str]) -> np.ndarray:
    source_index = {unit: idx for idx, unit in enumerate(map(str, source_units))}
    out = np.zeros((len(target_units), features.shape[1]), dtype=float)
    for row, unit in enumerate(map(str, target_units)):
        if unit in source_index:
            out[row] = features[source_index[unit]]
    return out


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

            lower_result = self._build_layer_features(pair.lower_layer, stage, lower_expr, lower_paths, shared_genes, cfg)
            upper_result = self._build_layer_features(pair.upper_layer, stage, upper_expr, upper_paths, shared_genes, cfg)

            ddi_matrix = upper_result.cci
            ddi_units = upper_expr.units
            if cfg.cross_cell_ddi_source == "coarse_grained":
                ddi_matrix = _coarse_grain_cci(lower_result.cci, overlap.weights)
                ddi_units = stable_upper_units
                macro = _select_rows(_macro_features(ddi_matrix), stable_upper_units, upper_expr.units)
                macro_start = len(MICRO_FEATURES) + len(CELL_COMM_FEATURES)
                upper_result.matrix[:, macro_start : macro_start + len(MACRO_DDI_FEATURES)] = macro

            if cfg.feature_log1p:
                lower_mat = np.log1p(lower_result.matrix)
                upper_mat = np.log1p(upper_result.matrix)
            else:
                lower_mat = lower_result.matrix
                upper_mat = upper_result.matrix

            upper_ddi_edges = _edge_table_from_sparse(
                ddi_matrix,
                ddi_units,
                pair.upper_layer,
                stage,
                "macro_ddi",
                cfg.cross_cell_top_k_edges,
            )
            lower_mats.append(lower_mat)
            upper_mats.append(upper_mat)
            overlaps.append(overlap)
            lower_units_by_time.append(lower_expr.units)
            upper_units_by_time.append(upper_expr.units)
            lower_assignments_by_time.append(lower_assignments.rows.copy())
            upper_assignments_by_time.append(upper_assignments.rows.copy())
            lower_graphs.append(lower_result.graph)
            upper_graphs.append(
                LayerGraph(
                    layer=pair.upper_layer,
                    time_point=stage,
                    units=upper_expr.units,
                    genes=list(shared_genes),
                    intra_edges=upper_result.graph.intra_edges,
                    inter_edges=upper_ddi_edges.loc[:, EDGE_COLUMNS].copy() if not upper_ddi_edges.empty else upper_ddi_edges.loc[:, EDGE_COLUMNS].copy(),
                    shared_genes=list(shared_genes),
                )
            )
            upper_coords_by_time.append(align_coords(upper_expr.coords, stable_upper_units))
            coverage_tables.append(coverage_table(stage, stable_upper_units, overlap.coverage_counts(), upper_expr.units))
            spot_correspondence_tables.append(spot_correspondence)
            overlap_edge_tables.append(overlap_edges)
            overlap_quality_summaries.append(overlap_quality)
            graph_summaries.append(self._graph_summary(stage, lower_result, upper_result, lower_mat, upper_mat, upper_ddi_edges))
            stage_metadata.append(
                {
                    "stage": stage,
                    "lower_cci_source": lower_result.cci_source,
                    "upper_cci_source": upper_result.cci_source,
                    "lower_mode": lower_result.mode,
                    "upper_mode": upper_result.mode,
                    "ddi_source": cfg.cross_cell_ddi_source,
                }
            )

            exports[f"network_exports/{stage}_lower_topk_micro_edges.csv"] = lower_result.micro_edges
            exports[f"network_exports/{stage}_upper_topk_micro_edges.csv"] = upper_result.micro_edges
            exports[f"network_exports/{stage}_lower_cell_comm_edges_topk.csv"] = lower_result.cell_edges
            exports[f"network_exports/{stage}_upper_cell_comm_edges_topk.csv"] = upper_result.cell_edges
            exports[f"network_exports/{stage}_ddi_edges.csv"] = upper_ddi_edges

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
            feature_names=list(FEATURE_NAMES),
            feature_blocks={key: list(value) for key, value in FEATURE_BLOCKS.items()},
            graph_summaries=graph_summaries,
            exports=exports,
            metadata={
                "network_method": self.network_method,
                "cross_cell_multilayer_mode": "lr_grn_fallback",
                "cross_cell_ddi_source": cfg.cross_cell_ddi_source,
                "feature_block_count": len(FEATURE_BLOCKS),
                "feature_blocks": {key: list(value) for key, value in FEATURE_BLOCKS.items()},
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
        cfg: TemporalRunConfig,
    ) -> _LayerFeatureResult:
        expr = expression.expr.loc[:, list(shared_genes)].copy()
        grn = _restrict_grn(read_grn_edges(paths.grn_edges, cfg.top_k_targets_per_regulator), shared_genes)
        cci, cci_source = _read_cci_matrix(paths, expr.index.astype(str).tolist(), cfg.cci_min)
        intra = _intra_strength(expr, grn, cfg.expr_threshold)
        cross_out, cross_in, micro_edges, mode = _micro_cross_features(layer_name, stage, expr, paths, grn, cfg)
        matrix = np.zeros((expr.shape[0], len(FEATURE_NAMES)), dtype=float)
        matrix[:, 0] = intra
        matrix[:, 1] = cross_out
        matrix[:, 2] = cross_in
        matrix[:, 3] = cross_out + cross_in
        cell_start = len(MICRO_FEATURES)
        matrix[:, cell_start : cell_start + len(CELL_COMM_FEATURES)] = _communication_features(cci, cfg.cross_cell_top_k_edges_per_unit)
        macro_start = cell_start + len(CELL_COMM_FEATURES)
        matrix[:, macro_start : macro_start + len(MACRO_DDI_FEATURES)] = _macro_features(cci)

        intra_edges = _self_loop_edges(intra, expr.index.astype(str).tolist(), layer_name, stage, "micro_intra")
        cell_edges = _edge_table_from_sparse(cci, expr.index.astype(str).tolist(), layer_name, stage, "cell_communication", cfg.cross_cell_top_k_edges)
        graph = LayerGraph(
            layer=layer_name,
            time_point=stage,
            units=expr.index.astype(str).tolist(),
            genes=list(shared_genes),
            intra_edges=intra_edges.loc[:, EDGE_COLUMNS].copy() if not intra_edges.empty else intra_edges.loc[:, EDGE_COLUMNS].copy(),
            inter_edges=cell_edges.loc[:, EDGE_COLUMNS].copy() if not cell_edges.empty else cell_edges.loc[:, EDGE_COLUMNS].copy(),
            shared_genes=list(shared_genes),
        )
        return _LayerFeatureResult(
            matrix=matrix,
            graph=graph,
            cci=cci,
            cci_source=cci_source,
            mode=mode,
            micro_edges=micro_edges,
            cell_edges=cell_edges,
        )

    def _required_paths(self, paths: LayerPaths) -> List[Path]:
        required = [paths.h5ad, paths.grn_edges, paths.cci_index]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        if not paths.cci_total.exists():
            required.extend([paths.cci_manifest, paths.cci_lr_dir])
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
            lower_grn = read_grn_edges(lower_paths.grn_edges, cfg.top_k_targets_per_regulator)
            upper_grn = read_grn_edges(upper_paths.grn_edges, cfg.top_k_targets_per_regulator)
            lower_grn_genes = set(lower_grn["regulator"]).union(lower_grn["target"])
            upper_grn_genes = set(upper_grn["regulator"]).union(upper_grn["target"])
            intersections.append(lower_expr_genes & upper_expr_genes & lower_grn_genes & upper_grn_genes)
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError(f"Shared gene intersection is empty for {pair.label()}.")
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
        ddi_edges: pd.DataFrame,
    ) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": CrossCellMultilayerBuilder.network_method,
            "feature_blocks": {key: list(value) for key, value in FEATURE_BLOCKS.items()},
            "lower_units": len(lower_result.graph.units),
            "upper_units": len(upper_result.graph.units),
            "shared_genes": len(lower_result.graph.shared_genes),
            "lower_micro_edges_topk": int(len(lower_result.micro_edges)),
            "upper_micro_edges_topk": int(len(upper_result.micro_edges)),
            "lower_cell_comm_edges_topk": int(len(lower_result.cell_edges)),
            "upper_cell_comm_edges_topk": int(len(upper_result.cell_edges)),
            "ddi_edges_topk": int(len(ddi_edges)),
            "lower_matrix_shape": list(lower_mat.shape),
            "upper_matrix_shape": list(upper_mat.shape),
        }
