from __future__ import annotations

from typing import Tuple

import numpy as np

from mignet_ce.utils.coords import normalize_coords_pair
from mignet_ce.utils.matrix import safe_row_normalize


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms + eps, out=np.zeros_like(x, dtype=float), where=norms > 0)


def _support_from_spatial_and_similarity(
    similarity: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    sim_k: int,
    dist_k: int,
) -> np.ndarray:
    n_source, n_target = similarity.shape
    support = np.zeros((n_source, n_target), dtype=float)
    if n_source == 0 or n_target == 0:
        return support
    dist_k = n_target if dist_k <= 0 else min(int(dist_k), n_target)
    sim_k = dist_k if sim_k <= 0 else min(int(sim_k), dist_k)
    for row in range(n_source):
        distances = np.linalg.norm(target_coords - source_coords[row : row + 1], axis=1)
        dist_indices = np.argpartition(distances, dist_k - 1)[:dist_k]
        candidate_scores = similarity[row, dist_indices]
        if sim_k < len(dist_indices):
            top_local = np.argpartition(-candidate_scores, sim_k - 1)[:sim_k]
            keep = dist_indices[top_local]
        else:
            keep = dist_indices
        support[row, keep] = 1.0
    return support


def _unbalanced_sinkhorn(
    kernel: np.ndarray,
    epsilon: float,
    gamma: float,
    max_iter: int,
    eps: float = 1e-8,
) -> np.ndarray:
    if max_iter <= 0:
        return kernel
    n_source, n_target = kernel.shape
    if n_source == 0 or n_target == 0:
        return kernel.copy()
    power = gamma / (gamma + epsilon)
    a = np.full((n_source, 1), 1.0 / n_source, dtype=float)
    prob1 = np.full((n_source, 1), 1.0 / n_source, dtype=float)
    prob2 = np.full((n_target, 1), 1.0 / n_target, dtype=float)
    for _ in range(max_iter):
        k_t_a = kernel.T @ a
        b = np.power(prob2 / (k_t_a + eps), power)
        k_b = kernel @ b
        a = np.power(prob1 / (k_b + eps), power)
    return (a * kernel) * b.T


def build_3dot_transition_kernel(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    epsilon: float = 0.05,
    gamma: float = 1.0,
    max_iter: int = 100,
    sim_k: int = 10,
    dist_k: int = 50,
    return_similarity: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {source.shape} and {target.shape}.")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source.shape[1]} != {target.shape[1]}.")
    if source.shape[0] == 0 or target.shape[0] == 0:
        p = np.zeros((source.shape[0], target.shape[0]), dtype=float)
        return (p, p.copy()) if return_similarity else p

    source_coords_norm, target_coords_norm = normalize_coords_pair(source_coords, target_coords)
    source_norm = _l2_normalize(source)
    target_norm = _l2_normalize(target)
    similarity = source_norm @ target_norm.T
    support = _support_from_spatial_and_similarity(
        similarity=similarity,
        source_coords=source_coords_norm,
        target_coords=target_coords_norm,
        sim_k=sim_k,
        dist_k=dist_k,
    )
    cost = 1.0 - similarity
    kernel = np.exp(-cost / epsilon) * support
    transport = _unbalanced_sinkhorn(kernel, epsilon=epsilon, gamma=gamma, max_iter=max_iter)
    p = safe_row_normalize(transport)
    return (p, similarity) if return_similarity else p
