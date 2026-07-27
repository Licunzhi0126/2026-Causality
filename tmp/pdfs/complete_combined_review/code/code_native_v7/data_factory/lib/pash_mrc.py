"""Prospective Adaptive Spatial Hierarchy with Multi-Range Composition.

The implementation fits exactly one time point at a time. It receives only a
non-negative spot-by-gene expression matrix and a spot-by-2 coordinate matrix.
It cannot receive another time point, CCI, GRN, PIJ, EI, or reference labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler


METHOD_NAME = "pash_mrc_r"
METHOD_VERSION = "pash_mrc_r_v1_corrected_neighbors"


@dataclass(frozen=True)
class PASHMRCConfig:
    n_hvg: int = 1200
    n_pca: int = 30
    n_states: int = 16
    ring_ends: tuple[int, int, int] = (6, 18, 36)
    composition_dim: int = 24
    k40: int = 40
    k150: int = 150
    clustering_knn: int = 10
    diagnostic_knn: int = 6
    weight_expression: float = 0.75
    weight_composition: float = 1.00
    weight_neighborhood: float = 0.30
    weight_spatial: float = 0.16
    max_detached_piece: int = 2
    icm_lambda: float = 0.35
    icm_balance: float = 0.03
    icm_passes: int = 3
    min_domain_size_during_icm: int = 3
    random_state: int = 20260723


@dataclass(frozen=True)
class PASHMRCResult:
    labels_k40: np.ndarray
    labels_k150: np.ndarray
    features: np.ndarray
    metadata: dict[str, Any]


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"Expected a nonempty two-dimensional feature matrix, got {values.shape}.")
    return StandardScaler().fit_transform(values)


def _weighted_block(values: np.ndarray, weight: float) -> np.ndarray:
    """Standardize a block, preserve its weight, and remove dimension-count bias."""
    block = _standardize(values)
    return (float(weight) / np.sqrt(max(1, block.shape[1]))) * block


def _symmetric_knn(values: np.ndarray, k: int) -> sp.csr_matrix:
    n_obs = len(values)
    if n_obs <= 1:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float64)
    k = min(max(1, int(k)), n_obs - 1)
    graph = kneighbors_graph(
        values,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
    )
    return graph.maximum(graph.T).tocsr()


def _row_normalize(graph: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return (sp.diags(inv) @ graph).tocsr()


def _as_nonnegative_expression(expression) -> sp.csr_matrix:
    if sp.issparse(expression):
        matrix = expression.tocsr(copy=True).astype(np.float64)
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("Expression contains NaN or infinite sparse values.")
        if matrix.data.size and np.min(matrix.data) < 0:
            raise ValueError("Expression must be non-negative.")
        matrix.eliminate_zeros()
        return matrix

    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expression must be two-dimensional, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN or infinite values.")
    if values.size and np.min(values) < 0:
        raise ValueError("Expression must be non-negative.")
    return sp.csr_matrix(values)


def _log_cpm(expression) -> sp.csr_matrix:
    matrix = _as_nonnegative_expression(expression)
    library = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    normalized = (sp.diags(scale) @ matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def _sparse_variance(matrix: sp.csr_matrix) -> np.ndarray:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    return np.maximum(mean_square - mean * mean, 0.0)


def _expression_embedding(expression, cfg: PASHMRCConfig) -> np.ndarray:
    values = _log_cpm(expression)
    if values.shape[1] == 0:
        raise ValueError("Expression has no genes.")
    n_hvg = min(max(1, int(cfg.n_hvg)), values.shape[1])
    hvg = np.argsort(_sparse_variance(values), kind="stable")[-n_hvg:]
    scaled = _standardize(values[:, hvg].toarray())
    n_components = min(int(cfg.n_pca), values.shape[0] - 1, scaled.shape[1])
    if n_components < 1:
        raise ValueError("At least two spots and one expression component are required.")
    embedding = PCA(n_components=n_components, svd_solver="full").fit_transform(scaled)
    return _standardize(embedding)


def _ranked_neighbors_excluding_self(values: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Return actual neighbor ranks 1..max_neighbors without discarding rank 1.

    The report prototype called ``kneighbors(X=None)`` (which already excludes
    the fitted row) and then sliced away its first result. Here the query is
    explicit and the self index is removed by identity, so duplicate coordinates
    do not make self-removal depend on distance ordering.
    """
    values = np.asarray(values, dtype=float)
    n_obs = len(values)
    width = min(max(0, int(max_neighbors)), max(0, n_obs - 1))
    if width == 0:
        return np.empty((n_obs, 0), dtype=np.int32)

    query_width = min(n_obs, width + 1)
    raw = NearestNeighbors(n_neighbors=query_width).fit(values).kneighbors(
        values,
        return_distance=False,
    )
    result = np.empty((n_obs, width), dtype=np.int32)
    for row_index, row in enumerate(raw):
        candidates = [int(index) for index in row if int(index) != row_index]
        if len(candidates) < width:
            raise RuntimeError(
                f"Could not obtain {width} non-self neighbors for row {row_index}; "
                f"obtained {len(candidates)}."
            )
        result[row_index] = candidates[:width]
    return result


def _multirange_composition(
    expression_embedding: np.ndarray,
    coords: np.ndarray,
    cfg: PASHMRCConfig,
) -> np.ndarray:
    if int(cfg.n_states) < 1 or int(cfg.n_states) > len(coords):
        raise ValueError(f"n_states={cfg.n_states} must be in [1, {len(coords)}].")
    state_labels = KMeans(
        n_clusters=int(cfg.n_states),
        random_state=int(cfg.random_state),
        n_init=30,
        algorithm="lloyd",
    ).fit_predict(expression_embedding)
    one_hot = np.eye(int(cfg.n_states), dtype=float)[state_labels]

    if tuple(sorted(cfg.ring_ends)) != tuple(cfg.ring_ends) or min(cfg.ring_ends) < 1:
        raise ValueError(f"ring_ends must be positive and increasing, got {cfg.ring_ends}.")
    neighbor_index = _ranked_neighbors_excluding_self(coords, max(cfg.ring_ends))
    ring_parts: list[np.ndarray] = []
    start = 0
    for requested_stop in cfg.ring_ends:
        stop = min(int(requested_stop), neighbor_index.shape[1])
        if stop <= start:
            ring_parts.append(np.zeros((len(coords), int(cfg.n_states)), dtype=float))
        else:
            ring_parts.append(one_hot[neighbor_index[:, start:stop]].mean(axis=1))
        start = stop
    composition = _standardize(np.hstack(ring_parts))
    n_components = min(int(cfg.composition_dim), composition.shape[1], len(coords) - 1)
    if n_components < 1:
        raise ValueError("At least one multi-range composition component is required.")
    reduced = PCA(n_components=n_components, svd_solver="full").fit_transform(composition)
    return _standardize(reduced)


def _build_features(expression, coords: np.ndarray, cfg: PASHMRCConfig) -> np.ndarray:
    expr = _expression_embedding(expression, cfg)
    composition = _multirange_composition(expr, coords, cfg)
    local_expr = _row_normalize(_symmetric_knn(coords, cfg.diagnostic_knn)) @ expr
    spatial = _standardize(coords)
    return np.column_stack(
        [
            _weighted_block(expr, cfg.weight_expression),
            _weighted_block(composition, cfg.weight_composition),
            _weighted_block(local_expr, cfg.weight_neighborhood),
            _weighted_block(spatial, cfg.weight_spatial),
        ]
    )


def _repair_small_detached_components(
    labels: np.ndarray,
    features: np.ndarray,
    coords: np.ndarray,
    cfg: PASHMRCConfig,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    graph = _symmetric_knn(coords, cfg.diagnostic_knn)
    n_clusters = int(labels.max()) + 1

    for _ in range(4):
        changed = 0
        centroids = np.vstack([features[labels == k].mean(axis=0) for k in range(n_clusters)])
        for cluster in range(n_clusters):
            index = np.flatnonzero(labels == cluster)
            if len(index) <= 1:
                continue
            n_components, component = connected_components(
                graph[index][:, index],
                directed=False,
                return_labels=True,
            )
            if n_components <= 1:
                continue
            sizes = np.bincount(component)
            main_component = int(np.argmax(sizes))
            for component_id in range(n_components):
                if component_id == main_component or sizes[component_id] > cfg.max_detached_piece:
                    continue
                piece = index[component == component_id]
                neighbor_labels: set[int] = set()
                for spot in piece:
                    neighbors = graph.indices[graph.indptr[spot] : graph.indptr[spot + 1]]
                    neighbor_labels.update(labels[neighbors].tolist())
                neighbor_labels.discard(cluster)
                if not neighbor_labels:
                    continue
                target = min(
                    (
                        float(np.mean(np.sum((features[piece] - centroids[q]) ** 2, axis=1))),
                        q,
                    )
                    for q in sorted(neighbor_labels)
                )[1]
                labels[piece] = target
                changed += len(piece)
        if changed == 0:
            break
    return labels


def _boundary_icm(
    labels: np.ndarray,
    features: np.ndarray,
    coords: np.ndarray,
    cfg: PASHMRCConfig,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    graph = _symmetric_knn(coords, cfg.diagnostic_knn)
    n_clusters = int(labels.max()) + 1

    for _ in range(int(cfg.icm_passes)):
        sizes = np.bincount(labels, minlength=n_clusters).astype(int)
        centroids = np.vstack([features[labels == k].mean(axis=0) for k in range(n_clusters)])
        own_cost = np.sum((features - centroids[labels]) ** 2, axis=1)
        feature_scale = float(np.median(own_cost) + 1e-9)
        target_size = len(labels) / n_clusters
        changed = 0

        for spot in range(len(labels)):
            current = int(labels[spot])
            neighbors = graph.indices[graph.indptr[spot] : graph.indptr[spot + 1]]
            candidates = sorted(set(labels[neighbors].tolist() + [current]))

            def score(candidate: int) -> float:
                feature_cost = float(
                    np.sum((features[spot] - centroids[candidate]) ** 2) / feature_scale
                )
                boundary_cost = (
                    float(np.mean(labels[neighbors] != candidate)) if len(neighbors) else 0.0
                )
                balance_cost = abs(np.log((sizes[candidate] + 1) / target_size))
                return (
                    feature_cost
                    + float(cfg.icm_lambda) * boundary_cost
                    + float(cfg.icm_balance) * balance_cost
                )

            best_score, best = min((score(candidate), candidate) for candidate in candidates)
            if (
                best != current
                and best_score + 1e-10 < score(current)
                and sizes[current] > int(cfg.min_domain_size_during_icm)
            ):
                labels[spot] = best
                sizes[current] -= 1
                sizes[best] += 1
                changed += 1
        if changed == 0:
            break
    return labels


def _allocate_nested_counts(parent: np.ndarray, target: int) -> dict[int, int]:
    parent_ids = np.unique(parent)
    sizes = {int(parent_id): int(np.sum(parent == parent_id)) for parent_id in parent_ids}
    counts = {int(parent_id): 1 for parent_id in parent_ids}
    remaining = int(target) - len(parent_ids)
    while remaining > 0:
        eligible = [parent_id for parent_id in counts if counts[parent_id] < sizes[parent_id]]
        if not eligible:
            raise ValueError("Cannot allocate requested nested domains without empty clusters.")
        parent_id = max(eligible, key=lambda value: (sizes[value] / counts[value], -value))
        counts[parent_id] += 1
        remaining -= 1
    return counts


def _nested_k150(labels_k40: np.ndarray, coords: np.ndarray, cfg: PASHMRCConfig) -> np.ndarray:
    counts = _allocate_nested_counts(labels_k40, cfg.k150)
    spatial = _standardize(coords)
    labels_k150 = np.empty(len(labels_k40), dtype=int)
    offset = 0
    for parent in sorted(counts):
        index = np.flatnonzero(labels_k40 == parent)
        child_count = counts[parent]
        if child_count == 1:
            child = np.zeros(len(index), dtype=int)
        elif child_count == len(index):
            child = np.arange(len(index), dtype=int)
        else:
            connectivity = _symmetric_knn(coords[index], min(6, len(index) - 1))
            child = AgglomerativeClustering(
                n_clusters=child_count,
                linkage="ward",
                connectivity=connectivity,
            ).fit_predict(spatial[index])
        labels_k150[index] = offset + child
        offset += child_count
    if offset != cfg.k150:
        raise RuntimeError(f"Expected {cfg.k150} nested domains, obtained {offset}.")
    return labels_k150


def _validate_inputs(expression, coords: np.ndarray, cfg: PASHMRCConfig) -> tuple[object, np.ndarray]:
    if not hasattr(expression, "shape") or len(expression.shape) != 2:
        raise ValueError("Expected expression[n_spot,n_gene].")
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Expected coords[n_spot,2].")
    if expression.shape[0] != len(coords):
        raise ValueError("Expression and coordinate rows must match.")
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or infinite values.")
    if not (1 <= int(cfg.k40) <= int(cfg.k150) <= len(coords)):
        raise ValueError(
            f"Expected 1 <= k40 <= k150 <= n_spots, got "
            f"{cfg.k40}, {cfg.k150}, {len(coords)}."
        )
    return expression, coords


def fit_single_timepoint(
    expression,
    coords: np.ndarray,
    *,
    config: PASHMRCConfig = PASHMRCConfig(),
) -> PASHMRCResult:
    """Fit prospective, strictly nested K40/K150 labels for one time point."""
    expression, coords = _validate_inputs(expression, coords, config)
    features = _build_features(expression, coords, config)
    labels_k40 = AgglomerativeClustering(
        n_clusters=int(config.k40),
        linkage="ward",
        connectivity=_symmetric_knn(coords, config.clustering_knn),
    ).fit_predict(features)
    labels_k40 = _repair_small_detached_components(labels_k40, features, coords, config)
    labels_k40 = _boundary_icm(labels_k40, features, coords, config)
    if len(np.unique(labels_k40)) != int(config.k40):
        raise RuntimeError("Refinement removed a K40 domain.")

    labels_k150 = _nested_k150(labels_k40, coords, config)
    parent_sets = [
        np.unique(labels_k40[labels_k150 == child])
        for child in np.unique(labels_k150)
    ]
    if any(len(parent) != 1 for parent in parent_sets):
        raise RuntimeError("K150 is not strictly nested inside K40.")

    return PASHMRCResult(
        labels_k40=np.asarray(labels_k40, dtype=np.int32),
        labels_k150=np.asarray(labels_k150, dtype=np.int32),
        features=np.asarray(features, dtype=np.float32),
        metadata={
            "method": METHOD_NAME,
            "method_version": METHOD_VERSION,
            "prospective_single_timepoint": True,
            "uses_future_time": False,
            "uses_cci": False,
            "uses_grn": False,
            "uses_pgr": False,
            "uses_pij_or_ei": False,
            "strict_k150_in_k40": True,
            "neighbor_ranks": ["1-6", "7-18", "19-36"],
            "prototype_neighbor_bug_corrected": True,
            "config": asdict(config),
        },
    )
