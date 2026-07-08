from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from mignet_ce.config import TemporalRunConfig
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.developmental_features import load_developmental_features_for_pij
from mignet_ce.io.loaders import (
    LayerDataResolver,
    LayerPaths,
    read_commot_index,
    read_commot_manifest,
    read_expression_h5ad,
)
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.networks.base import NetworkContext
from mignet_ce.representations.expression_only import (
    _fit_transform_gene_scaler,
    _preprocess_raw_mats,
    _reduce_aligned_features,
    _select_gene_indices,
)


@dataclass
class CompareFeatureSet:
    lower_features: List[np.ndarray]
    upper_features: List[np.ndarray]
    feature_names: List[str]
    metadata: dict[str, object]
    artifacts: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)


@dataclass
class _BaseFeatureResult:
    lower: List[np.ndarray]
    upper: List[np.ndarray]
    names: List[str]
    metadata: dict[str, object]
    artifacts: dict[str, dict[str, object]] = field(default_factory=lambda: {"lower": {}, "upper": {}})


def _as_nonnegative_csr(matrix: sp.spmatrix, cci_min: float) -> sp.csr_matrix:
    out = matrix.tocsr(copy=True).astype(float)
    if out.nnz:
        out.data = np.nan_to_num(out.data, nan=0.0, posinf=0.0, neginf=0.0)
        out.data[out.data < 0.0] = 0.0
        if cci_min > 0:
            out.data[out.data < float(cci_min)] = 0.0
        out.eliminate_zeros()
    return out


def _align_square_matrix(mat: sp.spmatrix, index_names: Sequence[str], target_units: Sequence[str]) -> sp.csr_matrix:
    target_units = list(map(str, target_units))
    index_lookup = {unit: idx for idx, unit in enumerate(map(str, index_names))}
    target_rows: List[int] = []
    source_rows: List[int] = []
    for out_idx, unit in enumerate(target_units):
        src_idx = index_lookup.get(unit)
        if src_idx is not None:
            target_rows.append(out_idx)
            source_rows.append(src_idx)
    if not target_rows:
        return sp.csr_matrix((len(target_units), len(target_units)), dtype=float)
    sub = mat.tocsr()[source_rows, :][:, source_rows].tocoo()
    rows = np.asarray([target_rows[int(row)] for row in sub.row], dtype=int)
    cols = np.asarray([target_rows[int(col)] for col in sub.col], dtype=int)
    return sp.coo_matrix((sub.data, (rows, cols)), shape=(len(target_units), len(target_units)), dtype=float).tocsr()


def read_compare_adjacency(
    paths: LayerPaths,
    units: Sequence[str],
    *,
    cci_min: float = 0.0,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    if not paths.cci_index.exists():
        raise FileNotFoundError(f"Missing COMMOT/CCI index file: {paths.cci_index}")
    index_names = read_commot_index(paths.cci_index)
    source = "total"
    lr_files = 0
    if paths.cci_total.exists():
        matrix = sp.load_npz(paths.cci_total)
        matrix_path = paths.cci_total
        if matrix.shape[0] != len(index_names) or matrix.shape[1] != len(index_names):
            raise ValueError(
                f"CCI total shape {matrix.shape} does not match index length {len(index_names)} for {paths.cci_total}."
            )
    else:
        if not paths.cci_manifest.exists():
            raise FileNotFoundError(f"Missing CCI total and COMMOT LR manifest for {paths.sample_stem}: {paths.cci_manifest}")
        if not paths.cci_lr_dir.exists():
            raise FileNotFoundError(f"Missing CCI total and COMMOT LR directory for {paths.sample_stem}: {paths.cci_lr_dir}")
        manifest = read_commot_manifest(paths.cci_manifest)
        total: sp.csr_matrix | None = None
        for row in manifest.itertuples(index=False):
            lr_path = paths.cci_lr_dir / str(row.filename)
            lr_matrix = sp.load_npz(lr_path)
            if lr_matrix.shape[0] != len(index_names) or lr_matrix.shape[1] != len(index_names):
                raise ValueError(
                    f"COMMOT LR matrix shape {lr_matrix.shape} does not match index length {len(index_names)} for {lr_path}."
                )
            total = lr_matrix.tocsr() if total is None else total + lr_matrix.tocsr()
            lr_files += 1
        matrix = total if total is not None else sp.csr_matrix((len(index_names), len(index_names)), dtype=float)
        matrix_path = paths.cci_lr_dir
        source = "lr_aggregate"
    aligned = _as_nonnegative_csr(_align_square_matrix(matrix, index_names, units), cci_min=cci_min)
    missing_units = [unit for unit in map(str, units) if unit not in set(map(str, index_names))]
    return aligned, {
        "layer": paths.layer,
        "organ": paths.organ,
        "stage": str(paths.stage),
        "sample_stem": paths.sample_stem,
        "source": source,
        "path": str(matrix_path),
        "index_path": str(paths.cci_index),
        "index_rows": int(len(index_names)),
        "requested_units": int(len(list(units))),
        "missing_units": int(len(missing_units)),
        "shape": list(aligned.shape),
        "nnz": int(aligned.nnz),
        "dtype": str(aligned.dtype),
        "cci_min": float(cci_min),
        "lr_files": int(lr_files),
    }


def adjacency_from_lightcci_graph(
    graph: LayerGraph,
    units: Sequence[str] | None = None,
    *,
    cci_min: float = 0.0,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    graph_units = list(map(str, graph.units))
    target_units = list(map(str, units if units is not None else graph_units))
    stored = graph.metadata.get("adjacency_csr")
    if stored is not None:
        matrix = stored if sp.issparse(stored) else sp.csr_matrix(stored)
        if matrix.shape != (len(graph_units), len(graph_units)):
            raise ValueError(
                f"LightCCI graph adjacency shape {matrix.shape} does not match graph units {len(graph_units)} "
                f"for {graph.layer} {graph.time_point}."
            )
        aligned = _align_square_matrix(matrix, graph_units, target_units)
    else:
        index = {unit: idx for idx, unit in enumerate(target_units)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for table in (graph.intra_edges, graph.inter_edges):
            if table.empty or not {"src_unit", "dst_unit", "influence_score"}.issubset(table.columns):
                continue
            work = table.loc[:, ["src_unit", "dst_unit", "influence_score"]].copy()
            work["src_unit"] = work["src_unit"].astype(str)
            work["dst_unit"] = work["dst_unit"].astype(str)
            work["influence_score"] = pd.to_numeric(work["influence_score"], errors="coerce")
            work = work.dropna(subset=["influence_score"])
            for row in work.itertuples(index=False):
                src = index.get(str(row.src_unit))
                dst = index.get(str(row.dst_unit))
                if src is None or dst is None:
                    continue
                value = float(row.influence_score)
                if value <= 0:
                    continue
                rows.append(src)
                cols.append(dst)
                data.append(value)
        aligned = sp.coo_matrix((data, (rows, cols)), shape=(len(target_units), len(target_units)), dtype=float).tocsr()
    aligned = _as_nonnegative_csr(aligned, cci_min=cci_min)
    return aligned, {
        "source": "light_cci_graph",
        "layer": graph.layer,
        "stage": str(graph.time_point),
        "edge_source": graph.metadata.get("edge_source"),
        "adjacency_source": graph.metadata.get("adjacency_source"),
        "adjacency_path": graph.metadata.get("adjacency_path"),
        "layer_semantics": graph.metadata.get("layer_semantics"),
        "uses_grn": bool(graph.metadata.get("uses_grn", False)),
        "uses_cci": bool(graph.metadata.get("uses_cci", False)),
        "shape": list(aligned.shape),
        "nnz": int(aligned.nnz),
        "requested_units": int(len(target_units)),
        "graph_units": int(len(graph_units)),
    }


def _context_units(context: NetworkContext, side: str, time_index: int) -> List[str]:
    if context.feature_alignment_space != "native_units":
        return list(map(str, context.stable_upper_units))
    if side == "lower":
        return list(map(str, context.lower_units_by_time[time_index]))
    if side == "upper":
        return list(map(str, context.upper_units_by_time[time_index]))
    raise ValueError("side must be one of 'lower' or 'upper'.")


def _native_units(context: NetworkContext, side: str, time_index: int) -> List[str]:
    return list(map(str, context.lower_units_by_time[time_index] if side == "lower" else context.upper_units_by_time[time_index]))


def _layer_for_side(context: NetworkContext, side: str) -> str:
    return context.pair.lower_layer if side == "lower" else context.pair.upper_layer


def _paths_for_side(resolver: LayerDataResolver, context: NetworkContext, side: str, stage: str) -> LayerPaths:
    return resolver.paths(_layer_for_side(context, side), context.organ, str(stage))


def _align_side_features(
    raw: List[np.ndarray],
    context: NetworkContext,
    side: str,
) -> tuple[List[np.ndarray], List[dict[str, object]]]:
    if context.feature_alignment_space == "native_units":
        return [np.asarray(matrix, dtype=float) for matrix in raw], [
            {"time_index": idx, "aligned_to": "native_units"} for idx in range(len(raw))
        ]
    aligned: List[np.ndarray] = []
    metadata: List[dict[str, object]] = []
    if side == "lower":
        for idx, matrix in enumerate(raw):
            values, coverage = aggregate_lower_features_to_upper(np.asarray(matrix, dtype=float), context.overlaps[idx])
            aligned.append(values)
            metadata.append(
                {
                    "time_index": idx,
                    "aligned_to": "stable_upper_units",
                    "lower_aggregation": "overlap_weighted_average",
                    "covered_units": int(np.count_nonzero(coverage > 0)),
                }
            )
    else:
        for idx, matrix in enumerate(raw):
            aligned.append(align_upper_features(np.asarray(matrix, dtype=float), context.upper_units_by_time[idx], context.stable_upper_units))
            metadata.append(
                {
                    "time_index": idx,
                    "aligned_to": "stable_upper_units",
                    "upper_alignment": "zero_fill_missing_units",
                }
            )
    return aligned, metadata


def _build_expression_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    resolver = LayerDataResolver(cfg.data_root)
    lower_raw: List[np.ndarray] = []
    upper_raw: List[np.ndarray] = []
    source_metadata: List[dict[str, object]] = []
    use_context_fallback = False
    for idx, stage in enumerate(map(str, context.time_points)):
        lower_paths = _paths_for_side(resolver, context, "lower", stage)
        upper_paths = _paths_for_side(resolver, context, "upper", stage)
        if not lower_paths.h5ad.exists() or not upper_paths.h5ad.exists():
            use_context_fallback = True
            break
        lower_expr = read_expression_h5ad(lower_paths.h5ad)
        upper_expr = read_expression_h5ad(upper_paths.h5ad)
        genes = list(context.shared_genes)
        missing_lower = [gene for gene in genes if gene not in lower_expr.expr.columns]
        missing_upper = [gene for gene in genes if gene not in upper_expr.expr.columns]
        if missing_lower or missing_upper:
            raise ValueError(
                f"Expression feature E is missing shared genes for {stage}: "
                f"lower_missing={missing_lower[:5]}, upper_missing={missing_upper[:5]}."
            )
        lower_units = _native_units(context, "lower", idx)
        upper_units = _native_units(context, "upper", idx)
        missing_lower_units = [unit for unit in lower_units if unit not in lower_expr.expr.index]
        missing_upper_units = [unit for unit in upper_units if unit not in upper_expr.expr.index]
        if missing_lower_units or missing_upper_units:
            raise ValueError(
                f"Expression feature E cannot align h5ad rows to context units for {stage}: "
                f"lower_missing_units={missing_lower_units[:5]}, upper_missing_units={missing_upper_units[:5]}."
            )
        lower_raw.append(lower_expr.expr.loc[lower_units, genes].to_numpy(dtype=float))
        upper_raw.append(upper_expr.expr.loc[upper_units, genes].to_numpy(dtype=float))
        source_metadata.append(
            {
                "stage": stage,
                "lower_h5ad": str(lower_paths.h5ad),
                "upper_h5ad": str(upper_paths.h5ad),
                "genes": int(len(genes)),
            }
        )
    if use_context_fallback:
        lower_raw = [np.asarray(matrix, dtype=float) for matrix in context.lower_mats]
        upper_raw = [np.asarray(matrix, dtype=float) for matrix in context.upper_mats]
        source_metadata.append(
            {
                "source": "network_context_mats_fallback",
                "reason": "one_or_more_h5ad_files_missing",
                "feature_dim": int(lower_raw[0].shape[1] if lower_raw else 0),
            }
        )

    lower_pre = _preprocess_raw_mats(
        lower_raw,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )
    upper_pre = _preprocess_raw_mats(
        upper_raw,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )
    genes_for_selection = context.shared_genes if not use_context_fallback else [f"context_feature_{i + 1}" for i in range(lower_pre[0].shape[1])]
    gene_indices, selected_genes, gene_selection_metadata = _select_gene_indices(
        [*lower_pre, *upper_pre],
        genes=genes_for_selection,
        max_genes=cfg.pure_expression_max_genes,
        mode=cfg.pure_expression_gene_selection,
    )
    lower_selected = [matrix[:, gene_indices] for matrix in lower_pre]
    upper_selected = [matrix[:, gene_indices] for matrix in upper_pre]
    lower_scaled_raw, upper_scaled_raw, scaler_metadata = _fit_transform_gene_scaler(
        lower_selected,
        upper_selected,
        scaler_name=cfg.pure_expression_scaler,
    )
    lower_aligned, lower_alignment = _align_side_features(lower_scaled_raw, context, "lower")
    upper_aligned, upper_alignment = _align_side_features(upper_scaled_raw, context, "upper")
    lower_reduced, upper_reduced, reduction_metadata, feature_names = _reduce_aligned_features(
        lower_aligned,
        upper_aligned,
        n_components=cfg.pij_feature_components,
        seed=cfg.nmf_seed,
    )
    return _BaseFeatureResult(
        lower=lower_reduced,
        upper=upper_reduced,
        names=[f"E:{name}" for name in feature_names],
        metadata={
            "feature_key": "E",
            "feature_source": "expression",
            "source_metadata": source_metadata,
            "normalization": {
                "library_size_normalize": bool(cfg.pure_expression_normalize),
                "log1p": bool(cfg.pure_expression_log1p),
                "scale_factor": float(cfg.pure_expression_scale_factor),
            },
            "gene_selection": gene_selection_metadata,
            "selected_genes": selected_genes,
            "gene_scaler": scaler_metadata,
            "feature_reduction": reduction_metadata,
            "lower_alignment": lower_alignment,
            "upper_alignment": upper_alignment,
        },
    )


def _adjacency_lists_for_side(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
) -> tuple[List[sp.csr_matrix], List[dict[str, object]]]:
    graph_list = context.lower_graphs if side == "lower" else context.upper_graphs
    if context.network_method == "light_cci" and graph_list:
        matrices: List[sp.csr_matrix] = []
        metadata: List[dict[str, object]] = []
        if len(graph_list) < len(context.time_points):
            raise ValueError(
                f"LightCCI context has {len(graph_list)} {side} graphs for {len(context.time_points)} time points."
            )
        for idx, graph in enumerate(graph_list[: len(context.time_points)]):
            units = _native_units(context, side, idx)
            matrix, matrix_metadata = adjacency_from_lightcci_graph(graph, units, cci_min=cfg.cci_min)
            matrices.append(matrix)
            metadata.append(matrix_metadata)
        return matrices, metadata

    resolver = LayerDataResolver(cfg.data_root)
    matrices: List[sp.csr_matrix] = []
    metadata: List[dict[str, object]] = []
    for idx, stage in enumerate(map(str, context.time_points)):
        paths = _paths_for_side(resolver, context, side, stage)
        units = _native_units(context, side, idx)
        matrix, matrix_metadata = read_compare_adjacency(paths, units, cci_min=cfg.cci_min)
        matrices.append(matrix)
        metadata.append(matrix_metadata)
    return matrices, metadata


def _build_nmf_side(
    matrices: List[sp.csr_matrix],
    side: str,
    cfg: TemporalRunConfig,
    metadata: List[dict[str, object]],
) -> tuple[List[np.ndarray], dict[str, object]]:
    shapes = [tuple(matrix.shape) for matrix in matrices]
    col_counts = [shape[1] for shape in shapes]
    if len(set(col_counts)) > 1:
        detail = {
            "side": side,
            "shapes": [list(shape) for shape in shapes],
            "columns": col_counts,
            "sources": metadata,
        }
        raise ValueError(
            "compare N feature requires temporal joint NMF inputs with identical column counts; "
            f"got {detail}."
        )
    dense = [matrix.toarray().astype(float, copy=False) for matrix in matrices]
    engine = TemporalMetricsEngine()
    w_list, h_matrix = engine.temporal_joint_nmf(
        dense,
        n_components=cfg.nmf_components,
        max_iter=cfg.nmf_max_iter,
        seed=cfg.nmf_seed,
    )
    return [np.asarray(w, dtype=float) for w in w_list], {
        "H": np.asarray(h_matrix, dtype=float),
        "W": [np.asarray(w, dtype=float) for w in w_list],
        "shapes": {
            "input_shapes": [list(shape) for shape in shapes],
            "W_shapes": [list(w.shape) for w in w_list],
            "H_shape": list(h_matrix.shape),
        },
        "diagnostics": {
            "side": side,
            "n_components": int(cfg.nmf_components),
            "max_iter": int(cfg.nmf_max_iter),
            "seed": int(cfg.nmf_seed),
            "input_nnz": [int(matrix.nnz) for matrix in matrices],
            "adjacency_sources": metadata,
        },
    }


def _build_nmf_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower_adj, lower_sources = _adjacency_lists_for_side(context, cfg, "lower")
    upper_adj, upper_sources = _adjacency_lists_for_side(context, cfg, "upper")
    lower_raw, lower_artifact = _build_nmf_side(lower_adj, "lower", cfg, lower_sources)
    upper_raw, upper_artifact = _build_nmf_side(upper_adj, "upper", cfg, upper_sources)
    lower_aligned, lower_alignment = _align_side_features(lower_raw, context, "lower")
    upper_aligned, upper_alignment = _align_side_features(upper_raw, context, "upper")
    names = [f"N:nmf_component_{idx + 1}" for idx in range(cfg.nmf_components)]
    return _BaseFeatureResult(
        lower=lower_aligned,
        upper=upper_aligned,
        names=names,
        metadata={
            "feature_key": "N",
            "feature_source": "adjacency_temporal_joint_nmf",
            "definition": "X_t^N = A_t from COMMOT/CCI adjacency; output feature is W_t row factor.",
            "lower_adjacency_sources": lower_sources,
            "upper_adjacency_sources": upper_sources,
            "lower_alignment": lower_alignment,
            "upper_alignment": upper_alignment,
        },
        artifacts={"lower": {"N": lower_artifact}, "upper": {"N": upper_artifact}},
    )


def _laplacian_matrix(adjacency: sp.spmatrix, normalized: bool) -> sp.csr_matrix:
    adj = adjacency.tocsr().astype(float)
    if adj.shape[0] == 0:
        return sp.csr_matrix(adj.shape, dtype=float)
    adj = 0.5 * (adj + adj.T)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    if normalized:
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degree + 1e-12))
        return (sp.eye(adj.shape[0], format="csr") - d_inv_sqrt @ adj @ d_inv_sqrt).tocsr()
    return (sp.diags(degree) - adj).tocsr()


def _solve_hks_eigendecomposition(laplacian: sp.spmatrix, eig_count: int) -> tuple[np.ndarray, np.ndarray]:
    n_units = laplacian.shape[0]
    if n_units == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    if eig_count >= n_units or n_units <= 512:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        return values, vectors
    try:
        values, vectors = eigsh(
            laplacian,
            k=eig_count,
            which="SM",
            tol=1e-6,
            maxiter=max(5000, 20 * n_units),
        )
    except ArpackNoConvergence as exc:
        values = getattr(exc, "eigenvalues", None)
        vectors = getattr(exc, "eigenvectors", None)
        if values is None or vectors is None:
            raise
    order = np.argsort(np.asarray(values, dtype=float))
    return np.asarray(values, dtype=float)[order], np.asarray(vectors, dtype=float)[:, order]


def laplacian_hks_features(
    adjacency: sp.spmatrix,
    *,
    n_components: int,
    normalized: bool,
) -> np.ndarray:
    n_units = adjacency.shape[0]
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_units == 0:
        return np.zeros((0, n_components), dtype=float)
    if n_units == 1 or adjacency.nnz == 0:
        return np.zeros((n_units, n_components), dtype=float)
    laplacian = _laplacian_matrix(adjacency, normalized=normalized)
    eig_count = min(n_units, max(2, n_components + 1))
    eigvals, eigvecs = _solve_hks_eigendecomposition(laplacian, eig_count=eig_count)
    if eigvals.size == 0:
        return np.zeros((n_units, n_components), dtype=float)
    positive = eigvals[eigvals > 1e-12]
    if positive.size:
        t_min = 1.0 / float(np.max(positive))
        t_max = 1.0 / float(np.min(positive))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min <= 0 or t_max <= 0:
            times = np.ones(n_components, dtype=float)
        elif t_max <= t_min:
            times = np.full(n_components, t_min, dtype=float)
        else:
            times = np.geomspace(t_min, t_max, num=n_components)
    else:
        times = np.ones(n_components, dtype=float)
    weights = np.exp(-np.outer(eigvals, times))
    hks = (eigvecs ** 2) @ weights
    means = np.mean(hks, axis=0, keepdims=True)
    stds = np.std(hks, axis=0, keepdims=True)
    return np.divide(hks - means, stds, out=np.zeros_like(hks), where=stds > 0)


def _build_laplacian_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower_adj, lower_sources = _adjacency_lists_for_side(context, cfg, "lower")
    upper_adj, upper_sources = _adjacency_lists_for_side(context, cfg, "upper")
    lower_raw = [
        laplacian_hks_features(matrix, n_components=cfg.laplacian_components, normalized=cfg.laplacian_normalized)
        for matrix in lower_adj
    ]
    upper_raw = [
        laplacian_hks_features(matrix, n_components=cfg.laplacian_components, normalized=cfg.laplacian_normalized)
        for matrix in upper_adj
    ]
    lower_aligned, lower_alignment = _align_side_features(lower_raw, context, "lower")
    upper_aligned, upper_alignment = _align_side_features(upper_raw, context, "upper")
    return _BaseFeatureResult(
        lower=lower_aligned,
        upper=upper_aligned,
        names=[f"L:hks_{idx + 1}" for idx in range(cfg.laplacian_components)],
        metadata={
            "feature_key": "L",
            "feature_source": "commot_cci_laplacian_hks",
            "laplacian_components": int(cfg.laplacian_components),
            "laplacian_normalized": bool(cfg.laplacian_normalized),
            "lower_adjacency_sources": lower_sources,
            "upper_adjacency_sources": upper_sources,
            "lower_alignment": lower_alignment,
            "upper_alignment": upper_alignment,
        },
    )


def _select_sr_column(values: pd.DataFrame) -> str:
    for column in ("sr", "potency_score"):
        if column in values.columns:
            return column
    raise ValueError(f"Sr feature requires 'sr' or 'potency_score'; available columns are {list(values.columns)}.")


def _build_sr_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    lower_meta: List[dict[str, object]] = []
    upper_meta: List[dict[str, object]] = []
    for idx, _stage in enumerate(map(str, context.time_points)):
        lower_table = load_developmental_features_for_pij(context, cfg, idx, "lower")
        upper_table = load_developmental_features_for_pij(context, cfg, idx, "upper")
        lower_col = _select_sr_column(lower_table.values)
        upper_col = _select_sr_column(upper_table.values)
        lower.append(lower_table.values.loc[:, [lower_col]].to_numpy(dtype=float))
        upper.append(upper_table.values.loc[:, [upper_col]].to_numpy(dtype=float))
        lower_meta.append({**lower_table.metadata, "feature_column_used": lower_col})
        upper_meta.append({**upper_table.metadata, "feature_column_used": upper_col})
    return _BaseFeatureResult(
        lower=lower,
        upper=upper,
        names=["Sr:sr"],
        metadata={
            "feature_key": "Sr",
            "feature_source": "developmental_sr",
            "lower_sources": lower_meta,
            "upper_sources": upper_meta,
        },
    )


def _standardize_base(
    lower: Sequence[np.ndarray],
    upper: Sequence[np.ndarray],
) -> tuple[List[np.ndarray], List[np.ndarray], dict[str, object]]:
    all_values = np.vstack([*lower, *upper])
    means = np.nanmean(all_values, axis=0, keepdims=True)
    stds = np.nanstd(all_values, axis=0, keepdims=True)
    lower_out = [
        np.divide(np.nan_to_num(matrix, nan=0.0) - means, stds, out=np.zeros_like(matrix, dtype=float), where=stds > 0)
        for matrix in lower
    ]
    upper_out = [
        np.divide(np.nan_to_num(matrix, nan=0.0) - means, stds, out=np.zeros_like(matrix, dtype=float), where=stds > 0)
        for matrix in upper
    ]
    return lower_out, upper_out, {
        "standardization": "fit_zscore_per_base_feature_across_lower_and_upper_times",
        "zero_variance_columns": int(np.count_nonzero(np.squeeze(stds, axis=0) <= 0)),
    }


def _feature_weight(feature_key: str, cfg: TemporalRunConfig) -> float:
    if feature_key == "E":
        return float(cfg.pij_expr_weight)
    if feature_key == "Sr":
        return float(cfg.pij_sr_weight)
    return 1.0


def _build_base_feature(context: NetworkContext, cfg: TemporalRunConfig, feature_key: str) -> _BaseFeatureResult:
    if feature_key == "E":
        return _build_expression_base(context, cfg)
    if feature_key == "N":
        return _build_nmf_base(context, cfg)
    if feature_key == "L":
        return _build_laplacian_base(context, cfg)
    if feature_key == "Sr":
        return _build_sr_base(context, cfg)
    raise ValueError(f"Unsupported compare feature key {feature_key!r}.")


def build_compare_feature_set(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    feature_keys: Sequence[str],
) -> CompareFeatureSet:
    keys = tuple(feature_keys)
    if not keys:
        raise ValueError("feature_keys cannot be empty.")
    lower_parts_by_time: List[List[np.ndarray]] = [[] for _ in context.time_points]
    upper_parts_by_time: List[List[np.ndarray]] = [[] for _ in context.time_points]
    names: List[str] = []
    metadata: dict[str, object] = {
        "feature_keys": list(keys),
        "base_features": {},
        "combination_rule": "standardize_each_base_then_concat_sqrt_weighted",
        "feature_alignment_space": context.feature_alignment_space,
    }
    artifacts: dict[str, dict[str, dict[str, object]]] = {"lower": {}, "upper": {}}

    for key in keys:
        base = _build_base_feature(context, cfg, key)
        lower_scaled, upper_scaled, standardization_metadata = _standardize_base(base.lower, base.upper)
        weight = max(0.0, _feature_weight(key, cfg))
        scale = float(np.sqrt(weight))
        for idx in range(len(context.time_points)):
            lower_parts_by_time[idx].append(lower_scaled[idx] * scale)
            upper_parts_by_time[idx].append(upper_scaled[idx] * scale)
        names.extend(base.names)
        base_metadata = dict(base.metadata)
        base_metadata.update(standardization_metadata)
        base_metadata["weight"] = float(weight)
        metadata["base_features"][key] = base_metadata
        for side in ("lower", "upper"):
            artifacts[side].update(base.artifacts.get(side, {}))

    lower_features = [np.hstack(parts) if parts else np.zeros((0, 0), dtype=float) for parts in lower_parts_by_time]
    upper_features = [np.hstack(parts) if parts else np.zeros((0, 0), dtype=float) for parts in upper_parts_by_time]
    metadata["feature_dim"] = int(lower_features[0].shape[1] if lower_features else 0)
    metadata["feature_names"] = names
    return CompareFeatureSet(
        lower_features=lower_features,
        upper_features=upper_features,
        feature_names=names,
        metadata=metadata,
        artifacts=artifacts,
    )


def build_compare_features(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    feature_keys: Sequence[str],
    side: str,
) -> tuple[List[np.ndarray], dict[str, object]]:
    feature_set = build_compare_feature_set(context, cfg, feature_keys)
    if side == "lower":
        return feature_set.lower_features, feature_set.metadata
    if side == "upper":
        return feature_set.upper_features, feature_set.metadata
    raise ValueError("side must be one of 'lower' or 'upper'.")
