from __future__ import annotations

from math import ceil
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from .config import Settings, TOPOLOGIES, Task, seed_for, write_json


def load_adjacency(path: Path) -> sparse.csr_matrix:
    raw = sparse.load_npz(path).tocoo()
    if raw.shape[0] != raw.shape[1] or not raw.shape[0]:
        raise ValueError(f"CCI must be a nonempty square matrix: {path}")
    if not np.isfinite(raw.data).all() or np.any(raw.data < 0):
        raise ValueError(f"CCI weights must be finite and nonnegative: {path}")
    matrix = raw.tocsr().astype(np.float64)
    if matrix.nnz != raw.nnz:
        raise ValueError(f"CCI contains duplicate edges: {path}")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def edge_arrays(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    return rows, matrix.indices, matrix.data


def without_loops(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    result = matrix.copy()
    result.setdiag(0)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _rank(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.lexsort((rng.permutation(len(scores)), -scores))


def _communities(matrix: sparse.csr_matrix) -> np.ndarray:
    graph = nx.from_scipy_sparse_array(matrix, create_using=nx.DiGraph).to_undirected()
    groups = nx.community.greedy_modularity_communities(graph, weight="weight") if graph.number_of_edges() else []
    labels = np.arange(matrix.shape[0], dtype=int)
    for label, group in enumerate(sorted(groups, key=min)):
        labels[list(group)] = label
    return labels


def _common_neighbors(support: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    counts = np.empty(len(rows), dtype=float)
    for start in range(0, support.shape[0], 128):
        stop = min(start + 128, support.shape[0])
        left, right = np.searchsorted(rows, (start, stop))
        if left < right:
            overlap = support[start:stop] @ support
            counts[left:right] = np.asarray(overlap[rows[left:right] - start, cols[left:right]]).ravel()
    return counts


def discover_regions(matrix: sparse.csr_matrix, settings: Settings) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    graph = without_loops(matrix)
    n = graph.shape[0]
    rows, cols, weights = edge_arrays(graph)
    rng = np.random.default_rng(seed_for(settings.perturbation_seed, "cci-backbone"))
    keep = _rank(weights, rng)[:ceil(settings.backbone_fraction * len(weights))]
    backbone = sparse.csr_matrix((weights[keep], (rows[keep], cols[keep])), shape=matrix.shape)
    labels = _communities(backbone)
    rows, cols, weights = edge_arrays(backbone)
    if not len(weights):
        regions = {name: np.empty(0, dtype=np.int64) for name in TOPOLOGIES}
        return regions, labels, {"full_off_diagonal_edges": 0, "backbone_edges": 0}

    directed = backbone.astype(bool)
    support = directed.maximum(directed.T).astype(np.int32).tocsr()
    support.sort_indices()
    degree = 2 * np.diff(support.indptr)
    degree_deviation = np.abs(degree - np.median(degree))
    sr, sc, _ = edge_arrays(support)
    inter = np.bincount(sr, weights=labels[sr] != labels[sc], minlength=n)
    common = _common_neighbors(support, sr, sc)
    triangle_counts = np.bincount(sr, weights=common / 2, minlength=n)
    possible = degree * np.maximum(degree - 1, 1) / 2
    clustering = triangle_counts / np.maximum(possible, 1)
    out_degree = np.diff(directed.indptr)
    in_degree = np.asarray(directed.sum(axis=0)).ravel()
    reciprocal = np.asarray(directed[cols, rows]).ravel().astype(bool)
    reciprocal_degree = (np.bincount(rows[reciprocal], minlength=n)
                         + np.bincount(cols[reciprocal], minlength=n))
    keys = rows.astype(np.int64) * n + cols
    measures = (
        (degree_deviation, degree_deviation[rows] + degree_deviation[cols]),
        (inter / np.maximum(degree, 1), 1 + (labels[rows] != labels[cols]) + weights / weights.max()),
        (clustering, common[np.searchsorted(sr.astype(np.int64) * n + sc, keys)]),
        (degree_deviation, np.abs(out_degree[rows] - in_degree[cols])),
        (reciprocal_degree, reciprocal.astype(float)),
    )
    regions = {}
    for name, (node_scores, edge_scores) in zip(TOPOLOGIES, measures, strict=True):
        rng = np.random.default_rng(seed_for(settings.perturbation_seed, "cci-region", name))
        chosen_nodes = _rank(node_scores, rng)[:ceil(settings.region_node_fraction * n)]
        in_region = np.zeros(n, dtype=bool)
        in_region[chosen_nodes] = True
        eligible = in_region[rows] | in_region[cols]
        if name == "P5":
            eligible &= reciprocal
        ranked = _rank(edge_scores[eligible], rng)
        regions[name] = keys[eligible][ranked[:ceil(settings.region_edge_fraction * len(ranked))]]

    sets = {name: set(keys.tolist()) for name, keys in regions.items()}
    overlaps = {}
    for i, left in enumerate(TOPOLOGIES):
        for right in TOPOLOGIES[i + 1:]:
            union = sets[left] | sets[right]
            overlaps[f"{left}-{right}"] = len(sets[left] & sets[right]) / len(union) if union else None
    return regions, labels, {
        "full_off_diagonal_edges": graph.nnz,
        "backbone_edges": backbone.nnz,
        "region_edge_counts": {name: len(edges) for name, edges in regions.items()},
        "region_jaccard": overlaps,
        "selection": "top-weight directed baseline support, excluding self-loops",
    }


def mirror_condition(matrix: sparse.csr_matrix, region: np.ndarray, task: Task,
                     settings: Settings) -> tuple[sparse.csr_matrix, dict]:
    selected = region[:ceil(task.level / 100 * len(region))]
    if not len(selected):
        raise ValueError(f"{task.topology} has no eligible baseline edges")
    rows, cols, values = edge_arrays(matrix)
    keys = rows.astype(np.int64) * matrix.shape[0] + cols
    indices = np.searchsorted(keys, selected)
    if np.any(indices >= len(keys)) or np.any(keys[indices] != selected) or np.any(selected // matrix.shape[0] == selected % matrix.shape[0]):
        raise ValueError("CCI region contains an absent edge or a self-loop")
    multiplier = 1 + (1 if task.variant == "positive" else -1) * settings.mirror_fraction
    altered = matrix.copy()
    original_weight = float(values[indices].sum())
    altered.data[indices] *= multiplier
    altered.eliminate_zeros()
    return altered, {
        "selected_edges": len(selected), "selected_baseline_weight": original_weight,
        "weight_l1_change": original_weight * settings.mirror_fraction,
        "selected_edge_fraction": len(selected) / np.count_nonzero(rows != cols),
        "selected_weight_fraction": original_weight / values[rows != cols].sum().item(),
        "edge_weight_multiplier": multiplier,
    }


def _gini(values: np.ndarray) -> float:
    ordered = np.sort(values)
    total = float(ordered.sum())
    if not total:
        return 0.0
    n = len(ordered)
    return float(2 * np.dot(np.arange(1, n + 1), ordered) / (n * total) - (n + 1) / n)


def _correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    dx = x - np.average(x, weights=weights)
    dy = y - np.average(y, weights=weights)
    denominator = np.sqrt(np.dot(weights, dx ** 2) * np.dot(weights, dy ** 2))
    return float(np.dot(weights, dx * dy) / denominator) if denominator else 0.0


def _directed_clustering(matrix: sparse.csr_matrix, denominator: np.ndarray) -> float:
    weights = matrix.astype(np.float32)
    weights.data = np.cbrt(weights.data)
    symmetric = (weights + weights.T).tocsr()
    n = matrix.shape[0]
    if n <= 2500 and matrix.nnz > 0.12 * n * n:
        dense = symmetric.toarray()
        traces = np.einsum("ij,ij->i", dense @ dense, dense, dtype=np.float64)
    else:
        traces = np.zeros(n)
        for start in range(0, n, 128):
            block = symmetric[start:start + 128]
            traces[start:start + block.shape[0]] = np.asarray((block @ symmetric).multiply(block).sum(axis=1)).ravel()
    return float(np.mean(traces / np.maximum(denominator, 1)))


def topology_metrics(matrix: sparse.csr_matrix, labels: np.ndarray, baseline_max: float) -> dict:
    graph = without_loops(matrix)
    rows, cols, weights = edge_arrays(graph)
    n = graph.shape[0]
    names = ("degree_gini", "degree_cv", "centralization", "modularity", "clustering",
             "assortativity", "reciprocity", "strength_gini", "clustering_fixed_weight_scale",
             "strength_assortativity_weighted", "reciprocity_weighted")
    if not len(weights):
        return {**dict.fromkeys(names, 0.0), "edge_count": 0, "node_count": n}
    out = np.bincount(rows, minlength=n)
    incoming = np.bincount(cols, minlength=n)
    out_weight = np.bincount(rows, weights=weights, minlength=n)
    in_weight = np.bincount(cols, weights=weights, minlength=n)
    reverse = np.asarray(graph[cols, rows]).ravel()
    reciprocal_degree = np.bincount(rows, weights=reverse > 0, minlength=n)
    degree = out + incoming
    total = float(weights.sum())
    internal = float(weights[labels[rows] == labels[cols]].sum())
    group_out = np.bincount(labels, weights=out_weight)
    group_in = np.bincount(labels, weights=in_weight)
    max_weight = float(weights.max())
    clustering = _directed_clustering(graph / max_weight, 2 * (degree * (degree - 1) - 2 * reciprocal_degree))
    return {
        "degree_gini": _gini(out), "degree_cv": float(out.std() / out.mean()),
        "centralization": float((n * out.max() - out.sum()) / max((n - 1) ** 2, 1)),
        "modularity": internal / total - float(np.dot(group_out, group_in)) / total ** 2,
        "clustering": clustering,
        "assortativity": _correlation(out[rows], incoming[cols], np.ones_like(weights)),
        "reciprocity": float(np.mean(reverse > 0)), "edge_count": graph.nnz, "node_count": n,
        "strength_gini": _gini(out_weight),
        "clustering_fixed_weight_scale": clustering * max_weight / baseline_max,
        "strength_assortativity_weighted": _correlation(out_weight[rows], in_weight[cols], weights),
        "reciprocity_weighted": float(np.minimum(weights, reverse).sum() / total),
    }


def plan_cci(path: Path, settings: Settings, directory: Path) -> dict:
    matrix = load_adjacency(path)
    regions, labels, audit = discover_regions(matrix, settings)
    nonloop = without_loops(matrix)
    baseline_max = float(nonloop.data.max()) if nonloop.nnz else 1.0
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "regions.npz", labels=labels, baseline_max=baseline_max, **regions)
    audit["baseline_topology"] = topology_metrics(matrix, labels, baseline_max)
    write_json(directory / "cci_preflight.json", audit)
    return audit


def stage_cci(path: Path, region_path: Path, task: Task, settings: Settings,
              directory: Path) -> dict:
    matrix = load_adjacency(path)
    with np.load(region_path, allow_pickle=False) as regions:
        altered, audit = mirror_condition(matrix, regions[task.topology], task, settings)
        audit.update(topology_metrics(altered, regions["labels"], float(regions["baseline_max"])))
    target = directory / "cci_source.npz"
    sparse.save_npz(target, altered, compressed=False)
    return {"input": str(target), "audit": audit}
