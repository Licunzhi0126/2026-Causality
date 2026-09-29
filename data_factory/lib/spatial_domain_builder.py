from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from factory_common import FACTORY_OUTPUT_ROOT


_BASE_MODULE = None


def _base():
    global _BASE_MODULE
    if _BASE_MODULE is None:
        import domain_builder_louvain as base_module

        _BASE_MODULE = base_module
    return _BASE_MODULE


@dataclass(frozen=True)
class SpatialDomainBuilderConfig:
    local_dir: Path = FACTORY_OUTPUT_ROOT / "spot"
    output_root: Path = FACTORY_OUTPUT_ROOT / "spatial_domain_k40"
    sample_names: Tuple[str, ...] = tuple()
    less_than_5_max_size: int = 4

    n_top_genes: int = 3000
    n_pcs: int = 30
    normalize_target_sum: float = 1e4
    expr_neighbors: int = 30
    spatial_neighbors: int = 12
    smooth_weight: float = 0.30
    expr_weight: float = 0.50
    spatial_weight: float = 0.50
    merge_spatial_weight: float = 0.25
    random_state: int = 2026
    spatial_algorithm: str = "ball_tree"


BuilderConfig = SpatialDomainBuilderConfig

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_spatial(adata: AnnData, file_path: Path) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError(f"{file_path} is missing obsm['spatial'].")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[0] != adata.n_obs:
        raise ValueError(f"{file_path} has invalid spatial shape {spatial.shape}.")
    return spatial


def choose_count_matrix(adata: AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def make_analysis_adata(spot_adata: AnnData, cfg: SpatialDomainBuilderConfig) -> tuple[AnnData, object]:
    return _base().make_analysis_adata(spot_adata, cfg)


def build_expression_connectivity(analysis_adata: AnnData, cfg: SpatialDomainBuilderConfig) -> tuple[sp.csr_matrix, np.ndarray]:
    return _base().build_expression_connectivity(analysis_adata, cfg)


def normalize_global_max(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    if matrix.nnz == 0:
        return matrix
    max_value = float(matrix.data.max())
    if max_value > 0:
        matrix.data /= max_value
    return matrix


def fit_exact_k_partition(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    target_k: int,
    cfg: SpatialDomainBuilderConfig,
) -> tuple[np.ndarray, Dict[str, object]]:
    return _base().fit_exact_k_partition(adjacency, features, target_k=target_k, cfg=cfg)


def fit_less_than_5_partition(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    cfg: SpatialDomainBuilderConfig,
) -> tuple[np.ndarray, Dict[str, object]]:
    return _base().fit_less_than_5_partition(adjacency, features, cfg=cfg)


def export_domain_result(
    spot_adata: AnnData,
    count_matrix,
    labels: np.ndarray,
    output_dir: Path,
    file_stem: str,
    build_info: Dict[str, object],
) -> None:
    _base().export_domain_result(
        spot_adata=spot_adata,
        count_matrix=count_matrix,
        labels=labels,
        output_dir=output_dir,
        file_stem=file_stem,
        build_info=build_info,
    )


def count_clusters(labels: np.ndarray) -> int:
    return int(np.unique(labels).size) if labels.size else 0


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    out = np.zeros(int(unique.max()) + 1, dtype=np.int32)
    out[unique] = counts.astype(np.int32)
    return out


def _validate_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    out = matrix.tocsr().astype(np.float32).copy()
    if out.nnz == 0:
        return out
    row_sums = np.asarray(out.sum(axis=1)).ravel().astype(np.float32)
    inv = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float32),
        where=row_sums > 0,
    )
    out.data *= np.repeat(inv, np.diff(out.indptr))
    return out


def build_spatial_connectivity(
    spatial: np.ndarray,
    n_neighbors: int,
    algorithm: str = "ball_tree",
) -> sp.csr_matrix:
    spatial = _validate_finite_array("spatial", spatial)
    n_spots = spatial.shape[0]
    if n_spots <= 1:
        return sp.csr_matrix((n_spots, n_spots), dtype=np.float32)

    effective_neighbors = min(max(1, int(n_neighbors)), n_spots - 1)
    knn = NearestNeighbors(n_neighbors=effective_neighbors + 1, algorithm=algorithm)
    knn.fit(spatial)
    distances, indices = knn.kneighbors(spatial)

    positive_distances = distances[distances > 0]
    sigma = float(np.median(positive_distances)) if positive_distances.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    rows = []
    cols = []
    data = []
    for row in range(n_spots):
        kept = 0
        for distance, col in zip(distances[row], indices[row]):
            col = int(col)
            if col == row:
                continue
            weight = np.exp(-float(distance) ** 2 / (2.0 * sigma**2))
            if np.isfinite(weight) and weight > 0:
                rows.append(row)
                cols.append(col)
                data.append(weight)
                kept += 1
            if kept >= effective_neighbors:
                break

    conn = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(n_spots, n_spots),
    )
    conn = conn.maximum(conn.T).tocsr()
    conn.setdiag(0)
    conn.eliminate_zeros()
    return normalize_global_max(conn)


def build_spatial_augmented_features(
    pca_features: np.ndarray,
    spatial_conn: sp.csr_matrix,
    smooth_weight: float,
) -> np.ndarray:
    pca_features = _validate_finite_array("pca_features", pca_features)
    if spatial_conn.shape[0] != pca_features.shape[0] or spatial_conn.shape[1] != pca_features.shape[0]:
        raise ValueError(
            "spatial_conn shape must match pca_features rows; "
            f"got {spatial_conn.shape} and {pca_features.shape}."
        )
    if not 0.0 <= float(smooth_weight) <= 1.0:
        raise ValueError("smooth_weight must be between 0 and 1.")

    weights = _row_normalize(spatial_conn)
    neighbor_features = weights @ pca_features
    augmented = (1.0 - float(smooth_weight)) * pca_features + float(smooth_weight) * neighbor_features
    return StandardScaler().fit_transform(augmented).astype(np.float32)


def fuse_expression_spatial_connectivity(
    expr_conn: sp.csr_matrix,
    spatial_conn: sp.csr_matrix,
    expr_weight: float = 0.5,
    spatial_weight: float = 0.5,
) -> sp.csr_matrix:
    if expr_conn.shape != spatial_conn.shape:
        raise ValueError(f"Connectivity shapes differ: {expr_conn.shape} != {spatial_conn.shape}.")
    if float(expr_weight) < 0 or float(spatial_weight) < 0:
        raise ValueError("Connectivity weights must be non-negative.")
    if float(expr_weight) + float(spatial_weight) <= 0:
        raise ValueError("At least one connectivity weight must be positive.")

    expr_conn = normalize_global_max(expr_conn)
    spatial_conn = normalize_global_max(spatial_conn)
    fused = float(expr_weight) * expr_conn + float(spatial_weight) * spatial_conn
    fused = fused.maximum(fused.T).tocsr()
    fused.setdiag(0)
    fused.eliminate_zeros()
    return normalize_global_max(fused)


def build_spatial_merge_features(
    x_aug: np.ndarray,
    spatial: np.ndarray,
    merge_spatial_weight: float = 0.25,
) -> np.ndarray:
    x_aug = _validate_finite_array("x_aug", x_aug)
    spatial = _validate_finite_array("spatial", spatial)
    if x_aug.shape[0] != spatial.shape[0]:
        raise ValueError(f"x_aug and spatial row counts differ: {x_aug.shape[0]} != {spatial.shape[0]}.")
    if float(merge_spatial_weight) < 0:
        raise ValueError("merge_spatial_weight must be non-negative.")

    spatial_scaled = StandardScaler().fit_transform(spatial)
    features = np.hstack([x_aug.astype(np.float32), float(merge_spatial_weight) * spatial_scaled.astype(np.float32)])
    return StandardScaler().fit_transform(features).astype(np.float32)


def build_spatial_domain_for_sample(
    spot_adata: AnnData,
    cfg: SpatialDomainBuilderConfig,
    mode: str,
    target_k: int | None,
) -> tuple[np.ndarray, Dict[str, object], object]:
    spatial = require_spatial(spot_adata, Path("<memory>"))
    analysis_adata, count_matrix = make_analysis_adata(spot_adata, cfg)
    expr_conn, expr_pca = build_expression_connectivity(analysis_adata, cfg)
    spatial_conn = build_spatial_connectivity(
        spatial,
        n_neighbors=cfg.spatial_neighbors,
        algorithm=cfg.spatial_algorithm,
    )
    x_aug = build_spatial_augmented_features(expr_pca, spatial_conn, smooth_weight=cfg.smooth_weight)
    fused_conn = fuse_expression_spatial_connectivity(
        expr_conn,
        spatial_conn,
        expr_weight=cfg.expr_weight,
        spatial_weight=cfg.spatial_weight,
    )
    merge_features = build_spatial_merge_features(
        x_aug,
        spatial,
        merge_spatial_weight=cfg.merge_spatial_weight,
    )

    if mode == "exact_k":
        if target_k is None:
            raise ValueError("target_k is required for exact_k mode.")
        labels, partition_info = fit_exact_k_partition(fused_conn, merge_features, target_k=int(target_k), cfg=cfg)
    elif mode == "less_than_5":
        labels, partition_info = fit_less_than_5_partition(fused_conn, merge_features, cfg=cfg)
    else:
        raise ValueError(f"Unsupported spatial domain mode: {mode!r}.")

    build_info: Dict[str, object] = {
        "method": "spatial_domain_core",
        "mode": mode,
        "target_k": None if target_k is None else int(target_k),
        "expr_neighbors": int(cfg.expr_neighbors),
        "spatial_neighbors": int(cfg.spatial_neighbors),
        "smooth_weight": float(cfg.smooth_weight),
        "expr_weight": float(cfg.expr_weight),
        "spatial_weight": float(cfg.spatial_weight),
        "merge_spatial_weight": float(cfg.merge_spatial_weight),
        "n_top_genes": int(cfg.n_top_genes),
        "n_pcs": int(cfg.n_pcs),
        "random_state": int(cfg.random_state),
        "spatial_algorithm": str(cfg.spatial_algorithm),
        "n_spots": int(spot_adata.n_obs),
        "n_domains": int(count_clusters(labels)),
        "partition_info": partition_info,
    }
    for key, value in partition_info.items():
        build_info.setdefault(key, value)
    return labels, build_info, count_matrix
