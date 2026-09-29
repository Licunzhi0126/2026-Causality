from __future__ import annotations

"""WYT network-only 80-dimensional node representation.

Migration source:
  reference/network_only_coarse_grain/scripts/build_network_only_features_v57.py

The 16 summary statistics, outgoing/incoming SVD definitions, SVD sign
canonicalization and joint z-score are kept intentionally close to v57.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


EPS = 1e-12


@dataclass(frozen=True)
class Network80Pair:
    features_t: np.ndarray
    features_tp: np.ndarray
    latent_t: np.ndarray
    latent_tp: np.ndarray
    feature_names: list[str]
    metadata: dict[str, object]


def dense_matrix(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return np.asarray(values, dtype=np.float32)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    values[~np.isfinite(values)] = 0.0
    values[values < 0.0] = 0.0
    row_sum = values.sum(axis=1, keepdims=True)
    return values / (row_sum + EPS)


def safe_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    width = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + EPS), axis=1)
    return entropy / (np.log(width + EPS) + EPS)


def topk_mass(probabilities: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), probabilities.shape[1])
    values = np.partition(probabilities, -k_eff, axis=1)[:, -k_eff:]
    return values.sum(axis=1)


def gini_like(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    ordered = np.sort(array, axis=1)
    width = array.shape[1]
    indices = np.arange(1, width + 1, dtype=np.float32)
    denominator = np.sum(ordered, axis=1) + EPS
    return (
        2.0 * np.sum(indices[None, :] * ordered, axis=1) / (width * denominator)
        - (width + 1.0) / width
    )


def summary_features(matrix: sp.spmatrix | np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = dense_matrix(matrix)
    row_prob = row_normalize(values)
    col_prob = row_normalize(values.T)
    row_sum = values.sum(axis=1)
    col_sum = values.sum(axis=0)
    row_nnz = (values > 0).sum(axis=1) / max(1, values.shape[1])
    col_nnz = (values > 0).sum(axis=0) / max(1, values.shape[0])
    diagonal = (
        np.diag(values)
        if values.shape[0] == values.shape[1]
        else np.zeros(values.shape[0], dtype=np.float32)
    )
    reciprocal = (
        np.minimum(values, values.T).sum(axis=1) / (row_sum + EPS)
        if values.shape[0] == values.shape[1]
        else np.zeros(values.shape[0], dtype=np.float32)
    )
    features = np.vstack(
        [
            np.log1p(row_sum),
            np.log1p(col_sum),
            row_nnz,
            col_nnz,
            safe_entropy(row_prob),
            safe_entropy(col_prob),
            row_prob.max(axis=1),
            col_prob.max(axis=1),
            topk_mass(row_prob, 5),
            topk_mass(row_prob, 10),
            topk_mass(col_prob, 5),
            topk_mass(col_prob, 10),
            gini_like(row_prob),
            gini_like(col_prob),
            diagonal,
            reciprocal,
        ]
    ).T.astype(np.float32)
    names = [
        "log_out_strength",
        "log_in_strength",
        "out_density",
        "in_density",
        "out_entropy",
        "in_entropy",
        "out_max_prob",
        "in_max_prob",
        "out_top5_mass",
        "out_top10_mass",
        "in_top5_mass",
        "in_top10_mass",
        "out_gini",
        "in_gini",
        "self_loop",
        "reciprocal_mass",
    ]
    return features, names


def svd_features(
    matrix: sp.spmatrix | np.ndarray,
    n_components: int = 32,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    values = dense_matrix(matrix)
    effective = min(int(n_components), min(values.shape) - 1)
    if effective <= 0:
        return np.zeros((values.shape[0], 0), dtype=np.float32), []
    outgoing = TruncatedSVD(n_components=effective, random_state=random_state).fit_transform(
        row_normalize(values)
    )
    incoming = TruncatedSVD(n_components=effective, random_state=random_state).fit_transform(
        row_normalize(values.T)
    )
    for embedding in (outgoing, incoming):
        for column in range(embedding.shape[1]):
            if embedding[:, column].sum() < 0:
                embedding[:, column] *= -1
    features = np.hstack([outgoing, incoming]).astype(np.float32)
    names = [f"svd_out_{idx}" for idx in range(effective)] + [
        f"svd_in_{idx}" for idx in range(effective)
    ]
    return features, names


def build_network80_features(
    matrix: sp.spmatrix | np.ndarray,
    *,
    svd_dim: int = 32,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    summary, summary_names = summary_features(matrix)
    svd, svd_names = svd_features(matrix, n_components=svd_dim, random_state=random_state)
    features = np.hstack([summary, svd]).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    return features, summary_names + svd_names


def joint_zscore(
    features_t: np.ndarray,
    features_tp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([features_t, features_tp]).astype(np.float32)
    mean = np.nanmean(combined, axis=0, keepdims=True)
    std = np.nanstd(combined, axis=0, keepdims=True) + 1e-6
    return (features_t - mean) / std, (features_tp - mean) / std


def joint_fixed_pca(
    features_t: np.ndarray,
    features_tp: np.ndarray,
    *,
    output_dim: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([features_t, features_tp]).astype(np.float64)
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    combined = (combined - mean) / std
    effective = min(combined.shape[0] - 1, combined.shape[1], int(output_dim))
    if effective <= 0:
        raise ValueError("Cannot build WYT fixed PCA features.")
    _, _, right_transpose = np.linalg.svd(combined, full_matrices=False)
    latent = combined @ right_transpose[:effective].T
    if effective < int(output_dim):
        latent = np.hstack(
            [latent, np.zeros((latent.shape[0], int(output_dim) - effective), dtype=np.float64)]
        )
    split = features_t.shape[0]
    return latent[:split].astype(np.float32), latent[split:].astype(np.float32)


def build_network80_pair(
    network_t: sp.spmatrix | np.ndarray,
    network_tp: sp.spmatrix | np.ndarray,
    *,
    svd_dim: int = 32,
    pca_dim: int = 32,
    random_state: int = 0,
) -> Network80Pair:
    features_t, names_t = build_network80_features(
        network_t,
        svd_dim=svd_dim,
        random_state=random_state,
    )
    features_tp, names_tp = build_network80_features(
        network_tp,
        svd_dim=svd_dim,
        random_state=random_state,
    )
    if names_t != names_tp:
        raise ValueError(
            "WYT network80 feature dimensions differ across the time pair; "
            f"t={len(names_t)}, tp={len(names_tp)}."
        )
    features_t, features_tp = joint_zscore(features_t, features_tp)
    latent_t, latent_tp = joint_fixed_pca(features_t, features_tp, output_dim=pca_dim)
    return Network80Pair(
        features_t=features_t,
        features_tp=features_tp,
        latent_t=latent_t,
        latent_tp=latent_tp,
        feature_names=names_t,
        metadata={
            "feature_type": "wyt_network80",
            "definition": "16 summary + outgoing SVD + incoming SVD",
            "svd_dim": int(svd_dim),
            "pca_dim": int(pca_dim),
            "feature_shape_t": list(features_t.shape),
            "feature_shape_tp": list(features_tp.shape),
            "latent_shape_t": list(latent_t.shape),
            "latent_shape_tp": list(latent_tp.shape),
            "joint_zscore": True,
            "joint_fixed_pca": True,
            "svd_sign_canonicalization": "flip_component_when_column_sum_is_negative",
        },
    )
