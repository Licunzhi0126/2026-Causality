from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.decomposition import PCA

from mignet_ce.config import TemporalRunConfig
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.developmental_features import load_developmental_features_for_layer, load_developmental_features_for_pij
from mignet_ce.io.loaders import (
    LayerDataResolver,
    LayerPaths,
    read_commot_index,
    read_commot_manifest,
    read_expression_h5ad,
)
from mignet_ce.metrics import TemporalMetricsEngine, pairwise_joint_nmf, pairwise_shared_core_directed_nmf
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import PairFeatures
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
    pairwise_lower_features: PairFeatures | None = None
    pairwise_upper_features: PairFeatures | None = None


@dataclass
class _BaseFeatureResult:
    lower: List[np.ndarray]
    upper: List[np.ndarray]
    names: List[str]
    metadata: dict[str, object]
    artifacts: dict[str, dict[str, object]] = field(default_factory=lambda: {"lower": {}, "upper": {}})
    pairwise_lower: PairFeatures | None = None
    pairwise_upper: PairFeatures | None = None
    is_pairwise: bool = False


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


def _is_gene_side(context: NetworkContext, side: str) -> bool:
    return _layer_for_side(context, side) == "gene"


def _has_gene_pair(context: NetworkContext) -> bool:
    return context.pair.lower_layer == "gene" or context.pair.upper_layer == "gene"


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


def _align_side_feature_for_time(
    matrix: np.ndarray,
    context: NetworkContext,
    side: str,
    time_index: int,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(matrix, dtype=float)
    if context.feature_alignment_space == "native_units":
        return values, {"time_index": int(time_index), "aligned_to": "native_units"}
    if side == "lower":
        aligned, coverage = aggregate_lower_features_to_upper(values, context.overlaps[time_index])
        return aligned, {
            "time_index": int(time_index),
            "aligned_to": "stable_upper_units",
            "lower_aggregation": "overlap_weighted_average",
            "covered_units": int(np.count_nonzero(coverage > 0)),
        }
    if side == "upper":
        return align_upper_features(values, context.upper_units_by_time[time_index], context.stable_upper_units), {
            "time_index": int(time_index),
            "aligned_to": "stable_upper_units",
            "upper_alignment": "zero_fill_missing_units",
        }
    raise ValueError("side must be one of 'lower' or 'upper'.")


def _preprocess_gene_expression_patterns(
    matrices: Sequence[np.ndarray],
    *,
    normalize: bool,
    log1p: bool,
    scale_factor: float,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for matrix in matrices:
        values = np.maximum(np.asarray(matrix, dtype=float), 0.0)
        if normalize:
            totals = values.sum(axis=1, keepdims=True)
            values = np.divide(values, totals, out=np.zeros_like(values, dtype=float), where=totals > 0) * float(scale_factor)
        if log1p:
            values = np.log1p(values)
        out.append(values)
    return out


def _build_gene_expression_pattern_mats(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
) -> tuple[List[np.ndarray], List[dict[str, object]], List[List[str]]]:
    if not _is_gene_side(context, side):
        raise ValueError("_build_gene_expression_pattern_mats can only be used for a gene layer side.")
    resolver = LayerDataResolver(cfg.data_root)
    matrices: List[np.ndarray] = []
    metadata: List[dict[str, object]] = []
    units_by_time: List[List[str]] = []
    for idx, stage in enumerate(map(str, context.time_points)):
        spot_paths = resolver.paths("spot", context.organ, stage)
        if not spot_paths.h5ad.exists():
            raise FileNotFoundError(f"Gene expression feature E requires spot h5ad: {spot_paths.h5ad}")
        spot_expr = read_expression_h5ad(spot_paths.h5ad)
        gene_units = _native_units(context, side, idx)
        gene_set = set(map(str, spot_expr.expr.columns))
        present = [gene for gene in gene_units if gene in gene_set]
        missing = [gene for gene in gene_units if gene not in gene_set]
        matrix = spot_expr.expr.reindex(columns=gene_units, fill_value=0.0).T.to_numpy(dtype=float)
        matrices.append(matrix)
        units_by_time.append(gene_units)
        metadata.append(
            {
                "stage": stage,
                "side": side,
                "gene_expression_source": "virtual_from_spot_h5ad",
                "gene_expression_representation": "gene_by_spot_expression",
                "spot_h5ad": str(spot_paths.h5ad),
                "spot_sample_stem": spot_paths.sample_stem,
                "spot_units": int(len(spot_expr.units)),
                "spot_genes": int(len(spot_expr.genes)),
                "gene_nodes": int(len(gene_units)),
                "present_gene_count": int(len(present)),
                "missing_gene_count": int(len(missing)),
                "missing_gene_examples": missing[:10],
            }
        )
    return matrices, metadata, units_by_time


def _zero_pad_columns(matrix: np.ndarray, n_columns: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape[1] == n_columns:
        return values
    if values.shape[1] > n_columns:
        return values[:, :n_columns]
    return np.pad(values, ((0, 0), (0, n_columns - values.shape[1])), mode="constant")


def _fit_gene_pattern_pca(matrix: np.ndarray, requested_components: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(matrix, dtype=float)
    n_rows, n_cols = values.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros((n_rows, requested_components), dtype=float), {
            "actual_components": 0,
            "reason": "empty_gene_by_spot_matrix",
        }
    actual = min(int(requested_components), n_rows, n_cols)
    if actual <= 0:
        return np.zeros((n_rows, requested_components), dtype=float), {
            "actual_components": 0,
            "reason": "no_available_components",
        }
    solver = "randomized" if actual < min(n_rows, n_cols) else "full"
    pca = PCA(n_components=actual, svd_solver=solver, random_state=seed)
    transformed = pca.fit_transform(values)
    return _zero_pad_columns(transformed, requested_components), {
        "actual_components": int(actual),
        "svd_solver": solver,
        "explained_variance_ratio_sum": float(np.nansum(pca.explained_variance_ratio_)),
    }


def _align_gene_expression_pcs(
    features: Sequence[np.ndarray],
    units_by_time: Sequence[Sequence[str]],
) -> tuple[List[np.ndarray], List[dict[str, object]]]:
    if not features:
        return [], []
    aligned = [np.asarray(features[0], dtype=float)]
    reference_units = list(map(str, units_by_time[0]))
    reference_index = {unit: idx for idx, unit in enumerate(reference_units)}
    alignment_metadata: List[dict[str, object]] = [
        {
            "time_index": 0,
            "alignment": "reference_timepoint",
            "shared_gene_rows": int(len(reference_units)),
        }
    ]
    for idx in range(1, len(features)):
        current = np.asarray(features[idx], dtype=float)
        current_units = list(map(str, units_by_time[idx]))
        current_index = {unit: row for row, unit in enumerate(current_units)}
        shared = [unit for unit in current_units if unit in reference_index]
        if len(shared) >= 2:
            current_rows = [current_index[unit] for unit in shared]
            reference_rows = [reference_index[unit] for unit in shared]
            rotation, scale = orthogonal_procrustes(current[current_rows], aligned[0][reference_rows])
            aligned.append(current @ rotation)
            alignment_metadata.append(
                {
                    "time_index": int(idx),
                    "alignment": "orthogonal_procrustes_to_first_timepoint",
                    "shared_gene_rows": int(len(shared)),
                    "procrustes_scale": float(scale),
                }
            )
        else:
            aligned.append(current)
            alignment_metadata.append(
                {
                    "time_index": int(idx),
                    "alignment": "skipped_insufficient_shared_gene_rows",
                    "shared_gene_rows": int(len(shared)),
                }
            )
    return aligned, alignment_metadata


def _reduce_gene_expression_patterns_with_temporal_alignment(
    matrices: Sequence[np.ndarray],
    units_by_time: Sequence[Sequence[str]],
    cfg: TemporalRunConfig,
) -> tuple[List[np.ndarray], dict[str, object], List[str]]:
    requested = int(cfg.compare_gene_expression_pca_components)
    preprocessed = _preprocess_gene_expression_patterns(
        matrices,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )
    reduced: List[np.ndarray] = []
    pca_metadata: List[dict[str, object]] = []
    for idx, matrix in enumerate(preprocessed):
        transformed, meta = _fit_gene_pattern_pca(matrix, requested, cfg.nmf_seed)
        meta.update(
            {
                "time_index": int(idx),
                "input_shape": list(np.asarray(matrix).shape),
                "output_shape": list(transformed.shape),
            }
        )
        reduced.append(transformed)
        pca_metadata.append(meta)
    aligned, alignment_metadata = _align_gene_expression_pcs(reduced, units_by_time)
    names = [f"gene_expression_pc_{idx + 1}" for idx in range(requested)]
    return aligned, {
        "gene_expression_source": "virtual_from_spot_h5ad",
        "gene_expression_representation": "gene_by_spot_pca",
        "gene_expression_pca_components": requested,
        "gene_expression_temporal_alignment": "orthogonal_procrustes_to_first_timepoint",
        "no_spatial_feature_used": True,
        "normalization": {
            "row_sum_normalize": bool(cfg.pure_expression_normalize),
            "log1p": bool(cfg.pure_expression_log1p),
            "scale_factor": float(cfg.pure_expression_scale_factor),
        },
        "pca_by_time": pca_metadata,
        "temporal_alignment_by_time": alignment_metadata,
    }, names


def _build_regular_expression_side(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
) -> tuple[List[np.ndarray], List[str], dict[str, object]]:
    resolver = LayerDataResolver(cfg.data_root)
    raw: List[np.ndarray] = []
    source_metadata: List[dict[str, object]] = []
    genes = list(context.shared_genes)
    use_context_fallback = False
    for idx, stage in enumerate(map(str, context.time_points)):
        paths = _paths_for_side(resolver, context, side, stage)
        if not paths.h5ad.exists():
            use_context_fallback = True
            break
        expression = read_expression_h5ad(paths.h5ad)
        missing_genes = [gene for gene in genes if gene not in expression.expr.columns]
        if missing_genes:
            raise ValueError(f"Expression feature E is missing shared genes for {side} {stage}: missing={missing_genes[:5]}.")
        units = _native_units(context, side, idx)
        missing_units = [unit for unit in units if unit not in expression.expr.index]
        if missing_units:
            raise ValueError(
                f"Expression feature E cannot align h5ad rows to {side} context units for {stage}: "
                f"missing_units={missing_units[:5]}."
            )
        raw.append(expression.expr.loc[units, genes].to_numpy(dtype=float))
        source_metadata.append(
            {
                "stage": stage,
                "side": side,
                "h5ad": str(paths.h5ad),
                "genes": int(len(genes)),
                "units": int(len(units)),
            }
        )
    if use_context_fallback:
        raw = [
            np.asarray(matrix, dtype=float)
            for matrix in (context.lower_mats if side == "lower" else context.upper_mats)
        ]
        genes = [f"context_feature_{idx + 1}" for idx in range(raw[0].shape[1] if raw else 0)]
        source_metadata.append(
            {
                "side": side,
                "source": "network_context_mats_fallback",
                "reason": "one_or_more_h5ad_files_missing",
                "feature_dim": int(raw[0].shape[1] if raw else 0),
            }
        )

    pre = _preprocess_raw_mats(
        raw,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )
    gene_indices, selected_genes, gene_selection_metadata = _select_gene_indices(
        pre,
        genes=genes,
        max_genes=cfg.pure_expression_max_genes,
        mode=cfg.pure_expression_gene_selection,
    )
    selected = [matrix[:, gene_indices] for matrix in pre]
    scaled, _, scaler_metadata = _fit_transform_gene_scaler(selected, [], scaler_name=cfg.pure_expression_scaler)
    aligned, alignment = _align_side_features(scaled, context, side)
    reduced, _, reduction_metadata, feature_names = _reduce_aligned_features(
        aligned,
        [],
        n_components=cfg.pij_feature_components,
        seed=cfg.nmf_seed,
    )
    return reduced, [f"E:{name}" for name in feature_names], {
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
        "alignment": alignment,
    }


def _build_expression_base_with_gene_side(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower_meta: dict[str, object]
    upper_meta: dict[str, object]
    if _is_gene_side(context, "lower"):
        lower_raw, lower_source_metadata, lower_units = _build_gene_expression_pattern_mats(context, cfg, "lower")
        lower_features, lower_reduction_metadata, lower_names = _reduce_gene_expression_patterns_with_temporal_alignment(
            lower_raw,
            lower_units,
            cfg,
        )
        lower_meta = {
            **lower_reduction_metadata,
            "source_metadata": lower_source_metadata,
        }
        lower_names = [f"E:{name}" for name in lower_names]
    else:
        lower_features, lower_names, lower_meta = _build_regular_expression_side(context, cfg, "lower")

    if _is_gene_side(context, "upper"):
        upper_raw, upper_source_metadata, upper_units = _build_gene_expression_pattern_mats(context, cfg, "upper")
        upper_features, upper_reduction_metadata, upper_names = _reduce_gene_expression_patterns_with_temporal_alignment(
            upper_raw,
            upper_units,
            cfg,
        )
        upper_meta = {
            **upper_reduction_metadata,
            "source_metadata": upper_source_metadata,
        }
        upper_names = [f"E:{name}" for name in upper_names]
    else:
        upper_features, upper_names, upper_meta = _build_regular_expression_side(context, cfg, "upper")

    names = lower_names if _is_gene_side(context, "lower") else upper_names if _is_gene_side(context, "upper") else lower_names
    return _BaseFeatureResult(
        lower=lower_features,
        upper=upper_features,
        names=names,
        metadata={
            "feature_key": "E",
            "feature_source": "expression",
            "gene_pair_virtualization": True,
            "lower": lower_meta,
            "upper": upper_meta,
            "no_spatial_feature_used": True,
        },
    )


def _build_expression_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    if _has_gene_pair(context):
        return _build_expression_base_with_gene_side(context, cfg)

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


def _time_pair_label(context: NetworkContext, pair: tuple[int, int]) -> str:
    return f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"


def _nmf_model_type_for_layer(layer: str) -> str:
    return "spot_shared_core_directed_nmf" if layer == "spot" else "ordinary_pairwise_joint_nmf"


def _build_pairwise_nmf_side(
    matrices: List[sp.csr_matrix],
    context: NetworkContext,
    side: str,
    cfg: TemporalRunConfig,
    metadata: List[dict[str, object]],
    pairs: Sequence[tuple[int, int]],
) -> tuple[PairFeatures, dict[str, object]]:
    layer = _layer_for_side(context, side)
    model_type = _nmf_model_type_for_layer(layer)
    pairwise: PairFeatures = {}
    artifact_pairs: dict[str, dict[str, object]] = {}
    pair_summaries: List[dict[str, object]] = []

    for pair in pairs:
        source_index, target_index = pair
        source_matrix = matrices[source_index]
        target_matrix = matrices[target_index]
        source_dense = source_matrix.toarray().astype(float, copy=False)
        target_dense = target_matrix.toarray().astype(float, copy=False)
        pair_label = _time_pair_label(context, pair)

        if layer == "spot":
            u_source, v_source, u_target, v_target, core = pairwise_shared_core_directed_nmf(
                source_dense,
                target_dense,
                n_components=cfg.nmf_components,
                max_iter=cfg.nmf_max_iter,
                seed=cfg.nmf_seed + source_index * 1009 + target_index,
            )
            raw_source = np.hstack([u_source, v_source])
            raw_target = np.hstack([u_target, v_target])
            model_artifact: dict[str, object] = {
                "model_type": model_type,
                "B": core,
                "U_source": u_source,
                "V_source": v_source,
                "U_target": u_target,
                "V_target": v_target,
                "features_source": raw_source,
                "features_target": raw_target,
                "feature_definition": "concat(outgoing_U, incoming_V)",
            }
        else:
            w_source, w_target, h_matrix = pairwise_joint_nmf(
                source_dense,
                target_dense,
                n_components=cfg.nmf_components,
                max_iter=cfg.nmf_max_iter,
                seed=cfg.nmf_seed + source_index * 1009 + target_index,
            )
            raw_source = w_source
            raw_target = w_target
            model_artifact = {
                "model_type": model_type,
                "H": h_matrix,
                "W_source": w_source,
                "W_target": w_target,
                "feature_definition": "W row factor",
            }

        aligned_source, source_alignment = _align_side_feature_for_time(raw_source, context, side, source_index)
        aligned_target, target_alignment = _align_side_feature_for_time(raw_target, context, side, target_index)
        pairwise[pair] = (aligned_source, aligned_target)
        summary = {
            "time_pair": pair_label,
            "model_type": model_type,
            "source_stage": str(context.time_points[source_index]),
            "target_stage": str(context.time_points[target_index]),
            "source_shape": list(source_matrix.shape),
            "target_shape": list(target_matrix.shape),
            "source_nnz": int(source_matrix.nnz),
            "target_nnz": int(target_matrix.nnz),
            "raw_feature_source_shape": list(raw_source.shape),
            "raw_feature_target_shape": list(raw_target.shape),
            "feature_source_shape": list(aligned_source.shape),
            "feature_target_shape": list(aligned_target.shape),
            "source_alignment": source_alignment,
            "target_alignment": target_alignment,
            "uses_only_pair_timepoints": True,
            "uses_domain_anchor": False,
            "requires_equal_column_count": bool(layer != "spot"),
        }
        pair_summaries.append(summary)
        artifact_pairs[pair_label] = {
            **model_artifact,
            **summary,
            "diagnostics": {
                "side": side,
                "layer": layer,
                "n_components": int(cfg.nmf_components),
                "max_iter": int(cfg.nmf_max_iter),
                "seed": int(cfg.nmf_seed + source_index * 1009 + target_index),
                "source_adjacency": metadata[source_index],
                "target_adjacency": metadata[target_index],
                "dtype": str(raw_source.dtype),
            },
        }

    return pairwise, {
        "model_scope": "pairwise_time_pair",
        "side": side,
        "layer": layer,
        "model_type": model_type,
        "n_components": int(cfg.nmf_components),
        "max_iter": int(cfg.nmf_max_iter),
        "seed": int(cfg.nmf_seed),
        "uses_only_pair_timepoints": True,
        "uses_domain_anchor": False,
        "adjacency_sources": metadata,
        "pair_summaries": pair_summaries,
        "pairwise": artifact_pairs,
    }


def _build_nmf_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower_adj, lower_sources = _adjacency_lists_for_side(context, cfg, "lower")
    upper_adj, upper_sources = _adjacency_lists_for_side(context, cfg, "upper")
    pairs = TemporalMetricsEngine.build_time_pairs_all(context.time_points)
    lower_pairwise, lower_artifact = _build_pairwise_nmf_side(
        lower_adj,
        context,
        "lower",
        cfg,
        lower_sources,
        pairs,
    )
    upper_pairwise, upper_artifact = _build_pairwise_nmf_side(
        upper_adj,
        context,
        "upper",
        cfg,
        upper_sources,
        pairs,
    )
    lower_empty = [np.zeros((len(_context_units(context, "lower", idx)), 0), dtype=float) for idx in range(len(context.time_points))]
    upper_empty = [np.zeros((len(_context_units(context, "upper", idx)), 0), dtype=float) for idx in range(len(context.time_points))]
    max_dim = max(
        [features.shape[1] for pair_features in (*lower_pairwise.values(), *upper_pairwise.values()) for features in pair_features],
        default=0,
    )
    names = [f"N:pairwise_nmf_component_{idx + 1}" for idx in range(max_dim)]
    return _BaseFeatureResult(
        lower=lower_empty,
        upper=upper_empty,
        names=names,
        metadata={
            "feature_key": "N",
            "feature_source": "pairwise_nmf",
            "definition": (
                "N is built independently for each time-pair from adjacency matrices. "
                "Spot sides use shared-core directed NMF; fixed-node sides use ordinary pairwise joint NMF."
            ),
            "uses_only_pair_timepoints": True,
            "uses_domain_anchor": False,
            "lower_model_type": lower_artifact["model_type"],
            "upper_model_type": upper_artifact["model_type"],
            "lower_adjacency_sources": lower_sources,
            "upper_adjacency_sources": upper_sources,
            "lower_pair_summaries": lower_artifact["pair_summaries"],
            "upper_pair_summaries": upper_artifact["pair_summaries"],
        },
        artifacts={"lower": {"N": lower_artifact}, "upper": {"N": upper_artifact}},
        pairwise_lower=lower_pairwise,
        pairwise_upper=upper_pairwise,
        is_pairwise=True,
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


def _build_gene_sr_from_spot_expression(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    side: str,
    time_index: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if not _is_gene_side(context, side):
        raise ValueError("_build_gene_sr_from_spot_expression can only be used for a gene layer side.")
    if cfg.development_feature_root is None:
        raise ValueError(f"{cfg.effective_pij_method()} requires development_feature_root.")
    stage = str(context.time_points[time_index])
    resolver = LayerDataResolver(cfg.data_root)
    spot_paths = resolver.paths("spot", context.organ, stage)
    if not spot_paths.h5ad.exists():
        raise FileNotFoundError(f"Gene Sr feature requires spot h5ad: {spot_paths.h5ad}")
    spot_expr = read_expression_h5ad(spot_paths.h5ad)
    spot_table = load_developmental_features_for_layer(
        development_feature_root=cfg.development_feature_root,
        data_root=cfg.data_root,
        layer="spot",
        organ=context.organ,
        stage=stage,
        units=spot_expr.units,
        aggregation=cfg.pij_feature_aggregation,
        missing_policy=cfg.pij_missing_feature_policy,
        spot_domain_map=spot_paths.spot_domain_map,
    )
    sr_column = _select_sr_column(spot_table.values)
    sr_values = spot_table.values.loc[:, sr_column].to_numpy(dtype=float)
    fallback_value = float(np.nanmean(sr_values)) if np.isfinite(sr_values).any() else 0.0
    sr_values = np.nan_to_num(sr_values, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)

    gene_units = _native_units(context, side, time_index)
    expression = spot_expr.expr.reindex(columns=gene_units, fill_value=0.0).to_numpy(dtype=float)
    expression = np.maximum(expression, 0.0)
    totals = expression.sum(axis=0)
    weighted = expression.T @ sr_values
    eps = 1e-12
    gene_sr = np.divide(weighted, totals, out=np.full(len(gene_units), fallback_value, dtype=float), where=totals > eps)
    spot_gene_set = set(map(str, spot_expr.expr.columns))
    missing = [gene for gene in gene_units if gene not in spot_gene_set]
    zero_expression = [gene for gene, total in zip(gene_units, totals) if float(total) <= eps]
    return gene_sr.reshape(-1, 1), {
        "stage": stage,
        "side": side,
        "feature_column_used": sr_column,
        "feature_key": "Sr",
        "feature_source": "developmental_sr",
        "gene_sr_source": "expression_weighted_spot_sr",
        "formula": "sum_s X_sg * Sr_s / (sum_s X_sg + eps)",
        "spot_h5ad": str(spot_paths.h5ad),
        "spot_developmental_feature_metadata": spot_table.metadata,
        "spot_units": int(len(spot_expr.units)),
        "gene_nodes": int(len(gene_units)),
        "missing_gene_count": int(len(missing)),
        "missing_gene_examples": missing[:10],
        "zero_expression_gene_count": int(len(zero_expression)),
        "zero_expression_gene_examples": zero_expression[:10],
        "fallback_value": fallback_value,
        "fallback_policy": "spot_sr_mean_for_missing_or_zero_expression_gene",
        "no_sr_correlation_used": True,
        "no_spatial_feature_used": True,
    }


def _build_sr_base(context: NetworkContext, cfg: TemporalRunConfig) -> _BaseFeatureResult:
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    lower_meta: List[dict[str, object]] = []
    upper_meta: List[dict[str, object]] = []
    for idx, _stage in enumerate(map(str, context.time_points)):
        if _is_gene_side(context, "lower"):
            lower_values, lower_source = _build_gene_sr_from_spot_expression(context, cfg, "lower", idx)
            lower.append(lower_values)
            lower_meta.append(lower_source)
        else:
            lower_table = load_developmental_features_for_pij(context, cfg, idx, "lower")
            lower_col = _select_sr_column(lower_table.values)
            lower.append(lower_table.values.loc[:, [lower_col]].to_numpy(dtype=float))
            lower_meta.append({**lower_table.metadata, "feature_column_used": lower_col})

        if _is_gene_side(context, "upper"):
            upper_values, upper_source = _build_gene_sr_from_spot_expression(context, cfg, "upper", idx)
            upper.append(upper_values)
            upper_meta.append(upper_source)
        else:
            upper_table = load_developmental_features_for_pij(context, cfg, idx, "upper")
            upper_col = _select_sr_column(upper_table.values)
            upper.append(upper_table.values.loc[:, [upper_col]].to_numpy(dtype=float))
            upper_meta.append({**upper_table.metadata, "feature_column_used": upper_col})
    name = "Sr:expression_weighted_sr" if _has_gene_pair(context) else "Sr:sr"
    return _BaseFeatureResult(
        lower=lower,
        upper=upper,
        names=[name],
        metadata={
            "feature_key": "Sr",
            "feature_source": "developmental_sr",
            "gene_pair_virtualization": bool(_has_gene_pair(context)),
            "lower_sources": lower_meta,
            "upper_sources": upper_meta,
        },
    )


def _standardize_base(
    lower: Sequence[np.ndarray],
    upper: Sequence[np.ndarray],
    *,
    side_specific: bool = False,
) -> tuple[List[np.ndarray], List[np.ndarray], dict[str, object]]:
    if side_specific:
        def standardize_side(values: Sequence[np.ndarray]) -> tuple[List[np.ndarray], int]:
            all_values = np.vstack([np.asarray(matrix, dtype=float) for matrix in values])
            means = np.nanmean(all_values, axis=0, keepdims=True)
            stds = np.nanstd(all_values, axis=0, keepdims=True)
            out = [
                np.divide(
                    np.nan_to_num(matrix, nan=0.0) - means,
                    stds,
                    out=np.zeros_like(matrix, dtype=float),
                    where=stds > 0,
                )
                for matrix in values
            ]
            return out, int(np.count_nonzero(np.squeeze(stds, axis=0) <= 0))

        lower_out, lower_zero = standardize_side(lower)
        upper_out, upper_zero = standardize_side(upper)
        return lower_out, upper_out, {
            "standardization": "side_specific_zscore_for_gene_pair",
            "lower_zero_variance_columns": lower_zero,
            "upper_zero_variance_columns": upper_zero,
        }

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


def _standardize_pairwise_features(
    pairwise: PairFeatures,
    context: NetworkContext,
) -> tuple[PairFeatures, dict[str, object]]:
    out: PairFeatures = {}
    summaries: List[dict[str, object]] = []
    for pair, (source, target) in pairwise.items():
        source_arr = np.asarray(source, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        if source_arr.shape[1] != target_arr.shape[1]:
            raise ValueError(
                f"Pairwise feature dimensions differ for {_time_pair_label(context, pair)}: "
                f"{source_arr.shape[1]} vs {target_arr.shape[1]}."
            )
        if source_arr.shape[1] == 0 or source_arr.shape[0] + target_arr.shape[0] == 0:
            out[pair] = (np.zeros_like(source_arr, dtype=float), np.zeros_like(target_arr, dtype=float))
            summaries.append(
                {
                    "time_pair": _time_pair_label(context, pair),
                    "zero_variance_columns": int(source_arr.shape[1]),
                    "source_shape": list(source_arr.shape),
                    "target_shape": list(target_arr.shape),
                }
            )
            continue
        all_values = np.vstack([source_arr, target_arr])
        means = np.nanmean(all_values, axis=0, keepdims=True)
        stds = np.nanstd(all_values, axis=0, keepdims=True)
        source_out = np.divide(
            np.nan_to_num(source_arr, nan=0.0) - means,
            stds,
            out=np.zeros_like(source_arr, dtype=float),
            where=stds > 0,
        )
        target_out = np.divide(
            np.nan_to_num(target_arr, nan=0.0) - means,
            stds,
            out=np.zeros_like(target_arr, dtype=float),
            where=stds > 0,
        )
        out[pair] = (source_out, target_out)
        summaries.append(
            {
                "time_pair": _time_pair_label(context, pair),
                "zero_variance_columns": int(np.count_nonzero(np.squeeze(stds, axis=0) <= 0)),
                "source_shape": list(source_arr.shape),
                "target_shape": list(target_arr.shape),
            }
        )
    return out, {
        "standardization": "pairwise_side_specific_zscore",
        "fit_scope": "per_time_pair_per_side_on_concat_source_target",
        "pairwise_standardization": summaries,
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
    *,
    apply_feature_weights: bool = True,
) -> CompareFeatureSet:
    keys = tuple(feature_keys)
    if not keys:
        raise ValueError("feature_keys cannot be empty.")
    lower_parts_by_time: List[List[np.ndarray]] = [[] for _ in context.time_points]
    upper_parts_by_time: List[List[np.ndarray]] = [[] for _ in context.time_points]
    lower_pairwise_parts: dict[tuple[int, int], tuple[List[np.ndarray], List[np.ndarray]]] = {}
    upper_pairwise_parts: dict[tuple[int, int], tuple[List[np.ndarray], List[np.ndarray]]] = {}
    names: List[str] = []
    metadata: dict[str, object] = {
        "feature_keys": list(keys),
        "base_features": {},
        "combination_rule": (
            "standardize_each_base_then_concat_sqrt_weighted"
            if apply_feature_weights
            else "standardize_each_base_then_concat_unweighted"
        ),
        "feature_weight_applied": bool(apply_feature_weights),
        "feature_alignment_space": context.feature_alignment_space,
    }
    artifacts: dict[str, dict[str, dict[str, object]]] = {"lower": {}, "upper": {}}

    for key in keys:
        base = _build_base_feature(context, cfg, key)
        requested_weight = max(0.0, _feature_weight(key, cfg))
        weight = requested_weight if apply_feature_weights else 1.0
        scale = float(np.sqrt(weight))
        names.extend(base.names)
        base_metadata = dict(base.metadata)

        if base.is_pairwise:
            if base.pairwise_lower is None or base.pairwise_upper is None:
                raise ValueError(f"Base feature {key!r} marked pairwise but did not provide pairwise features.")
            lower_scaled_pairwise, lower_standardization = _standardize_pairwise_features(base.pairwise_lower, context)
            upper_scaled_pairwise, upper_standardization = _standardize_pairwise_features(base.pairwise_upper, context)
            for pair, (source, target) in lower_scaled_pairwise.items():
                source_parts, target_parts = lower_pairwise_parts.setdefault(pair, ([], []))
                source_parts.append(source * scale)
                target_parts.append(target * scale)
            for pair, (source, target) in upper_scaled_pairwise.items():
                source_parts, target_parts = upper_pairwise_parts.setdefault(pair, ([], []))
                source_parts.append(source * scale)
                target_parts.append(target * scale)
            base_metadata.update(
                {
                    "standardization": "pairwise_side_specific_zscore",
                    "lower_standardization": lower_standardization,
                    "upper_standardization": upper_standardization,
                }
            )
        else:
            lower_scaled, upper_scaled, standardization_metadata = _standardize_base(
                base.lower,
                base.upper,
                side_specific=_has_gene_pair(context),
            )
            for idx in range(len(context.time_points)):
                lower_parts_by_time[idx].append(lower_scaled[idx] * scale)
                upper_parts_by_time[idx].append(upper_scaled[idx] * scale)
            base_metadata.update(standardization_metadata)

        base_metadata.update(
            {
                "requested_weight": float(requested_weight),
                "feature_weight_applied": bool(apply_feature_weights),
                "weight": float(weight),
                "scale": float(scale),
            }
        )
        metadata["base_features"][key] = base_metadata
        for side in ("lower", "upper"):
            artifacts[side].update(base.artifacts.get(side, {}))

    lower_features = [
        np.hstack(parts) if parts else np.zeros((len(_context_units(context, "lower", idx)), 0), dtype=float)
        for idx, parts in enumerate(lower_parts_by_time)
    ]
    upper_features = [
        np.hstack(parts) if parts else np.zeros((len(_context_units(context, "upper", idx)), 0), dtype=float)
        for idx, parts in enumerate(upper_parts_by_time)
    ]

    pairwise_lower_features: PairFeatures | None = None
    pairwise_upper_features: PairFeatures | None = None
    if lower_pairwise_parts or upper_pairwise_parts:
        pairwise_lower_features = {}
        pairwise_upper_features = {}
        pair_keys = sorted(set(lower_pairwise_parts) | set(upper_pairwise_parts))
        for pair in pair_keys:
            source_index, target_index = pair
            lower_source_parts = [*lower_parts_by_time[source_index], *lower_pairwise_parts.get(pair, ([], []))[0]]
            lower_target_parts = [*lower_parts_by_time[target_index], *lower_pairwise_parts.get(pair, ([], []))[1]]
            upper_source_parts = [*upper_parts_by_time[source_index], *upper_pairwise_parts.get(pair, ([], []))[0]]
            upper_target_parts = [*upper_parts_by_time[target_index], *upper_pairwise_parts.get(pair, ([], []))[1]]
            pairwise_lower_features[pair] = (
                np.hstack(lower_source_parts)
                if lower_source_parts
                else np.zeros((len(_context_units(context, "lower", source_index)), 0), dtype=float),
                np.hstack(lower_target_parts)
                if lower_target_parts
                else np.zeros((len(_context_units(context, "lower", target_index)), 0), dtype=float),
            )
            pairwise_upper_features[pair] = (
                np.hstack(upper_source_parts)
                if upper_source_parts
                else np.zeros((len(_context_units(context, "upper", source_index)), 0), dtype=float),
                np.hstack(upper_target_parts)
                if upper_target_parts
                else np.zeros((len(_context_units(context, "upper", target_index)), 0), dtype=float),
            )

    metadata["feature_dim"] = int(lower_features[0].shape[1] if lower_features else 0)
    metadata["timewise_feature_dim"] = int(lower_features[0].shape[1] if lower_features else 0)
    if pairwise_lower_features:
        first_pair = sorted(pairwise_lower_features)[0]
        metadata["pairwise_feature_dim"] = int(pairwise_lower_features[first_pair][0].shape[1])
        metadata["pairwise_features"] = {
            "enabled": True,
            "time_pairs": [_time_pair_label(context, pair) for pair in sorted(pairwise_lower_features)],
            "source": "pairwise_base_features_preferred_for_kernel_and_metrics",
        }
    else:
        metadata["pairwise_features"] = {"enabled": False}
    metadata["feature_names"] = names
    return CompareFeatureSet(
        lower_features=lower_features,
        upper_features=upper_features,
        feature_names=names,
        metadata=metadata,
        artifacts=artifacts,
        pairwise_lower_features=pairwise_lower_features,
        pairwise_upper_features=pairwise_upper_features,
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
