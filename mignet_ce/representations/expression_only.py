from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mignet_ce.config import TemporalRunConfig
from mignet_ce.features import aggregate_lower_features_to_upper, align_upper_features
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult


def aggregate_lower_expression_to_upper(expression: np.ndarray, overlap) -> tuple[np.ndarray, np.ndarray]:
    return aggregate_lower_features_to_upper(np.asarray(expression, dtype=float), overlap)


def align_upper_expression_to_stable(
    expression: np.ndarray,
    current_units: Sequence[str],
    stable_upper_units: Sequence[str],
) -> np.ndarray:
    return align_upper_features(np.asarray(expression, dtype=float), current_units, stable_upper_units)


def _library_size_normalize(matrix: np.ndarray, scale_factor: float) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values, dtype=float), where=totals > 0) * float(scale_factor)


def _preprocess_raw_mats(
    matrices: Sequence[np.ndarray],
    normalize: bool,
    log1p: bool,
    scale_factor: float,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for matrix in matrices:
        values = np.asarray(matrix, dtype=float)
        if normalize:
            values = _library_size_normalize(values, scale_factor=scale_factor)
        if log1p:
            values = np.log1p(np.maximum(values, 0.0))
        out.append(values)
    return out


def _select_gene_indices(
    matrices: Sequence[np.ndarray],
    genes: Sequence[str],
    max_genes: int | None,
    mode: str,
) -> tuple[np.ndarray, List[str], dict[str, object]]:
    gene_count = len(genes)
    if mode == "all" or max_genes is None or max_genes <= 0 or max_genes >= gene_count:
        indices = np.arange(gene_count, dtype=int)
        return indices, list(map(str, genes)), {
            "mode": "all" if mode == "all" else "variance",
            "requested_max_genes": None if max_genes is None else int(max_genes),
            "selected_gene_count": int(gene_count),
            "reduced": False,
        }
    if mode != "variance":
        raise ValueError("pure_expression_gene_selection must be one of {'variance', 'all'}.")
    all_values = np.vstack([np.asarray(matrix, dtype=float) for matrix in matrices])
    variances = np.var(all_values, axis=0)
    top = np.argsort(-variances, kind="mergesort")[: int(max_genes)]
    indices = np.sort(top)
    selected = [str(genes[idx]) for idx in indices]
    return indices, selected, {
        "mode": "variance",
        "requested_max_genes": int(max_genes),
        "selected_gene_count": int(len(indices)),
        "reduced": True,
    }


def _fit_transform_gene_scaler(
    lower_mats: Sequence[np.ndarray],
    upper_mats: Sequence[np.ndarray],
    scaler_name: str,
) -> tuple[List[np.ndarray], List[np.ndarray], dict[str, object]]:
    if scaler_name == "none":
        return list(lower_mats), list(upper_mats), {"type": "none", "scaled": False}
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("pure_expression_scaler must be one of {'standard', 'minmax', 'none'}.")
    scaler.fit(np.vstack([*lower_mats, *upper_mats]))
    return (
        [scaler.transform(matrix) for matrix in lower_mats],
        [scaler.transform(matrix) for matrix in upper_mats],
        {"type": scaler_name, "scaled": True},
    )


def _reduce_aligned_features(
    lower_features: Sequence[np.ndarray],
    upper_features: Sequence[np.ndarray],
    n_components: int | None,
    seed: int,
) -> tuple[List[np.ndarray], List[np.ndarray], dict[str, object], List[str]]:
    if n_components is None or n_components <= 0:
        feature_dim = lower_features[0].shape[1] if lower_features else 0
        names = [f"pure_expression_gene_component_{idx + 1}" for idx in range(feature_dim)]
        return list(lower_features), list(upper_features), {"reduced": False}, names

    all_features = np.vstack([*lower_features, *upper_features])
    max_components = min(all_features.shape[0], all_features.shape[1])
    if max_components <= 1:
        feature_dim = all_features.shape[1] if all_features.ndim == 2 else 0
        names = [f"pure_expression_gene_component_{idx + 1}" for idx in range(feature_dim)]
        return list(lower_features), list(upper_features), {"reduced": False, "reason": "not_enough_rank"}, names

    actual = min(int(n_components), max_components)
    pca = PCA(n_components=actual, random_state=seed)
    pca.fit(all_features)
    names = [f"pure_expression_pc_{idx + 1}" for idx in range(actual)]
    return (
        [pca.transform(matrix) for matrix in lower_features],
        [pca.transform(matrix) for matrix in upper_features],
        {
            "reduced": True,
            "requested_components": int(n_components),
            "actual_components": int(actual),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
        names,
    )


def build_expression_only_feature_result(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    n_components: int | None = None,
    seed: int | None = None,
) -> MethodResult:
    if context.metadata.get("feature_source") != "pure_expression" or context.metadata.get("uses_legacy_graph"):
        raise ValueError("pure expression features require an expression_only NetworkContext.")

    lower_pre = _preprocess_raw_mats(
        context.lower_mats,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )
    upper_pre = _preprocess_raw_mats(
        context.upper_mats,
        normalize=cfg.pure_expression_normalize,
        log1p=cfg.pure_expression_log1p,
        scale_factor=cfg.pure_expression_scale_factor,
    )

    gene_indices, selected_genes, gene_selection_metadata = _select_gene_indices(
        [*lower_pre, *upper_pre],
        genes=context.shared_genes,
        max_genes=cfg.pure_expression_max_genes,
        mode=cfg.pure_expression_gene_selection,
    )
    lower_selected = [matrix[:, gene_indices] for matrix in lower_pre]
    upper_selected = [matrix[:, gene_indices] for matrix in upper_pre]
    lower_scaled_units, upper_scaled_units, scaler_metadata = _fit_transform_gene_scaler(
        lower_selected,
        upper_selected,
        scaler_name=cfg.pure_expression_scaler,
    )

    lower_aligned: List[np.ndarray] = []
    upper_aligned: List[np.ndarray] = []
    lower_coverage: List[np.ndarray] = []
    for t in range(len(context.time_points)):
        lower_feat, denom = aggregate_lower_expression_to_upper(lower_scaled_units[t], context.overlaps[t])
        upper_feat = align_upper_expression_to_stable(
            upper_scaled_units[t],
            context.upper_units_by_time[t],
            context.stable_upper_units,
        )
        lower_aligned.append(lower_feat)
        upper_aligned.append(upper_feat)
        lower_coverage.append(denom)

    requested_components = cfg.pure_expression_pca_components if n_components is None else n_components
    if requested_components is None:
        requested_components = cfg.pij_feature_components
    lower_reduced, upper_reduced, reduction_metadata, feature_names = _reduce_aligned_features(
        lower_aligned,
        upper_aligned,
        n_components=requested_components,
        seed=cfg.nmf_seed if seed is None else seed,
    )

    return MethodResult(
        lower_features=lower_reduced,
        upper_features=upper_reduced,
        lower_coords=context.upper_coords_by_time,
        upper_coords=context.upper_coords_by_time,
        method_metadata={
            "representation": "pure_expression",
            "feature_source": "pure_expression",
            "uses_grn": False,
            "uses_cci": False,
            "uses_legacy_graph": False,
            "normalization": {
                "library_size_normalize": bool(cfg.pure_expression_normalize),
                "log1p": bool(cfg.pure_expression_log1p),
                "scale_factor": float(cfg.pure_expression_scale_factor),
            },
            "gene_selection": gene_selection_metadata,
            "selected_genes": selected_genes,
            "selected_gene_count": int(len(selected_genes)),
            "gene_scaler": scaler_metadata,
            "feature_reduction": reduction_metadata,
            "feature_names": feature_names,
            "lower_aggregation": "overlap_weighted_average",
            "upper_alignment_missing_policy": "zero_fill",
            "lower_overlap_coverage": [coverage.astype(float).tolist() for coverage in lower_coverage],
        },
    )
