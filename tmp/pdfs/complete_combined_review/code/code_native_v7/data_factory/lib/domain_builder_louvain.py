from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import networkx as nx
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData, read_h5ad
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


SERVER_WORK_ROOT = Path("/home/jovyan/work/2026 Causality")
DATA_ROOT = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory")
LOCAL_DIR = DATA_ROOT / "spot"
LOCAL_COMMOT_DIR = DATA_ROOT / "cci" / "spot"
OUTPUT_ROOT = DATA_ROOT / "louvain"

DEFAULT_K_VALUES: Tuple[int, ...] = tuple()
LESS_THAN_5_MAX_SIZE = 4

SEARCH_RESOLUTIONS = (
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
    6.00,
    8.00,
    10.00,
    12.00,
    16.00,
    20.00,
    24.00,
    32.00,
    40.00,
    50.00,
    64.00,
    80.00,
    100.00,
)
SPLIT_RESOLUTIONS = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0)


@dataclass(frozen=True)
class BuilderConfig:
    local_dir: Path = LOCAL_DIR
    local_commot_dir: Path = LOCAL_COMMOT_DIR
    output_root: Path = OUTPUT_ROOT
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES
    sample_names: Tuple[str, ...] = tuple()
    less_than_5_max_size: int = LESS_THAN_5_MAX_SIZE
    expr_neighbors: int = 30
    cci_topk: int = 30
    n_top_genes: int = 3000
    n_pcs: int = 30
    normalize_target_sum: float = 1e4
    expr_weight: float = 0.5
    cci_weight: float = 0.5
    random_state: int = 2026


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_local_files(local_dir: Path, sample_names: Sequence[str] = ()) -> List[Path]:
    files = sorted(local_dir.glob("*.h5ad"))
    if not sample_names:
        return files
    allowed = set(map(str, sample_names))
    return [path for path in files if path.stem in allowed]


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


def make_analysis_adata(spot_adata: AnnData, cfg: BuilderConfig) -> Tuple[AnnData, object]:
    count_matrix = choose_count_matrix(spot_adata)
    analysis = AnnData(
        X=count_matrix.copy() if hasattr(count_matrix, "copy") else np.array(count_matrix, copy=True),
        obs=spot_adata.obs.copy(),
        var=spot_adata.var.copy(),
    )
    analysis.obs_names = spot_adata.obs_names.astype(str)
    analysis.var_names = spot_adata.var_names.astype(str)
    analysis.obsm["spatial"] = require_spatial(spot_adata, Path("<memory>")).copy()

    sc.pp.normalize_total(analysis, target_sum=cfg.normalize_target_sum)
    sc.pp.log1p(analysis)
    analysis.var_names_make_unique()

    if cfg.n_top_genes < analysis.n_vars:
        sc.pp.highly_variable_genes(analysis, n_top_genes=cfg.n_top_genes, flavor="seurat")
        analysis = analysis[:, analysis.var["highly_variable"]].copy()

    n_comps = max(2, min(cfg.n_pcs, analysis.n_obs - 1, analysis.n_vars - 1))
    sc.pp.scale(analysis, zero_center=True, max_value=10)
    sc.tl.pca(analysis, n_comps=n_comps, svd_solver="arpack")
    return analysis, count_matrix


def load_cci_total(sample_name: str, obs_names: Sequence[str], cfg: BuilderConfig) -> sp.csr_matrix:
    matrix_path = cfg.local_commot_dir / f"{sample_name}_CCI_total.npz"
    index_path = cfg.local_commot_dir / f"{sample_name}_index.tsv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing COMMOT total matrix: {matrix_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing COMMOT index file: {index_path}")

    cci = sp.load_npz(matrix_path).tocsr().astype(np.float32)
    names = pd.read_csv(index_path, sep="\t").iloc[:, 0].astype(str).tolist()
    if len(names) != cci.shape[0]:
        raise ValueError(
            f"COMMOT index length {len(names)} does not match matrix shape {cci.shape} for {sample_name}."
        )

    obs_names = list(map(str, obs_names))
    if names == obs_names:
        return cci

    lookup = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in obs_names if name not in lookup]
    if missing:
        raise ValueError(f"COMMOT index is missing {len(missing)} spots for {sample_name}.")
    order = np.array([lookup[name] for name in obs_names], dtype=np.int64)
    return cci[order][:, order].tocsr()


def normalize_global_max(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    if matrix.nnz == 0:
        return matrix
    max_value = float(matrix.data.max())
    if max_value > 0:
        matrix.data /= max_value
    return matrix


def normalize_rows_by_max(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32).copy()
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        if start == end:
            continue
        row_max = float(matrix.data[start:end].max())
        if row_max > 0:
            matrix.data[start:end] /= row_max
    return matrix


def keep_topk_per_row(matrix: sp.csr_matrix, topk: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    data: List[float] = []
    indices: List[int] = []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        if row_data.size > topk:
            keep = np.argpartition(row_data, -topk)[-topk:]
            keep = keep[np.argsort(row_data[keep])[::-1]]
            row_data = row_data[keep]
            row_indices = row_indices[keep]
        data.extend(row_data.tolist())
        indices.extend(row_indices.tolist())
        indptr.append(len(data))
    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=matrix.shape,
    )


def build_expression_connectivity(analysis_adata: AnnData, cfg: BuilderConfig) -> Tuple[sp.csr_matrix, np.ndarray]:
    n_neighbors = min(cfg.expr_neighbors, max(1, analysis_adata.n_obs - 1))
    n_pcs = min(cfg.n_pcs, analysis_adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(analysis_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    connectivities = analysis_adata.obsp["connectivities"].tocsr().astype(np.float32)
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    return normalize_global_max(connectivities), np.asarray(analysis_adata.obsm["X_pca"][:, :n_pcs], dtype=np.float32)


def build_cci_connectivity(cci_total: sp.csr_matrix, cfg: BuilderConfig) -> sp.csr_matrix:
    directed = cci_total.tocsr().astype(np.float32)
    undirected = (directed + directed.T).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    undirected = normalize_rows_by_max(undirected)
    undirected = keep_topk_per_row(undirected, topk=cfg.cci_topk)
    undirected = undirected.maximum(undirected.T).tocsr()
    return normalize_global_max(undirected)


def build_merge_features(expr_pca: np.ndarray, cci_total: sp.csr_matrix) -> np.ndarray:
    sender = np.asarray(cci_total.sum(axis=1)).ravel()
    receiver = np.asarray(cci_total.sum(axis=0)).ravel()
    activity = np.log1p(np.column_stack([sender, receiver])).astype(np.float32)
    features = np.hstack([expr_pca.astype(np.float32), activity])
    return StandardScaler().fit_transform(features).astype(np.float32)


def fuse_connectivities(expr_conn: sp.csr_matrix, cci_conn: sp.csr_matrix, cfg: BuilderConfig) -> sp.csr_matrix:
    fused = cfg.expr_weight * expr_conn + cfg.cci_weight * cci_conn
    fused = fused.maximum(fused.T).tocsr()
    fused.setdiag(0)
    fused.eliminate_zeros()
    return normalize_global_max(fused)


def build_feature_knn_connectivity(features: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    if features.shape[0] <= 1:
        return sp.csr_matrix((features.shape[0], features.shape[0]), dtype=np.float32)
    adata = AnnData(X=np.asarray(features, dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, features.shape[0] - 1), use_rep="X")
    conn = adata.obsp["connectivities"].tocsr().astype(np.float32)
    conn.setdiag(0)
    conn.eliminate_zeros()
    return normalize_global_max(conn)


def relabel_partition(labels: Sequence[object]) -> np.ndarray:
    ordered = pd.Index(labels).astype(str).unique().tolist()
    mapping = {label: idx for idx, label in enumerate(ordered)}
    return np.array([mapping[str(label)] for label in labels], dtype=np.int32)


def count_clusters(labels: np.ndarray) -> int:
    return int(np.unique(labels).size) if labels.size else 0


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    out = np.zeros(int(unique.max()) + 1, dtype=np.int32)
    out[unique] = counts.astype(np.int32)
    return out


def run_louvain(adjacency: sp.csr_matrix, resolution: float, random_state: int) -> np.ndarray:
    if adjacency.shape[0] <= 1:
        return np.zeros(adjacency.shape[0], dtype=np.int32)
    graph = nx.from_scipy_sparse_array(adjacency, edge_attribute="weight")
    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        resolution=float(resolution),
        seed=int(random_state),
    )
    labels = np.zeros(adjacency.shape[0], dtype=np.int32)
    for community_id, nodes in enumerate(communities):
        labels[np.fromiter(nodes, dtype=np.int32)] = community_id
    return relabel_partition(labels.tolist())


def search_initial_partition(adjacency: sp.csr_matrix, target_k: int, cfg: BuilderConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    best_labels: Optional[np.ndarray] = None
    best_score: Optional[Tuple[int, int]] = None
    best_resolution = None
    search_rows = []

    for resolution in tqdm(
        SEARCH_RESOLUTIONS,
        desc=f"Init search K={target_k}",
        unit="res",
        leave=False,
    ):
        labels = run_louvain(adjacency, resolution=resolution, random_state=cfg.random_state)
        n_clusters = count_clusters(labels)
        search_rows.append({"resolution": resolution, "n_clusters": n_clusters})
        score = (abs(n_clusters - target_k), 0 if n_clusters >= target_k else 1)
        if best_score is None or score < best_score:
            best_labels = labels
            best_score = score
            best_resolution = resolution
        if n_clusters == target_k:
            break

    if best_labels is None or best_resolution is None:
        raise RuntimeError("Louvain search failed to produce an initial partition.")

    return best_labels, {
        "resolution": float(best_resolution),
        "search_table": search_rows,
        "n_clusters": int(count_clusters(best_labels)),
    }


def choose_best_split(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    indices: np.ndarray,
    current_n_clusters: int,
    target_k: int,
    cfg: BuilderConfig,
    random_state: int,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    if indices.size <= 1:
        return None

    candidates: List[Tuple[Tuple[int, int, int], np.ndarray, Dict[str, object]]] = []
    sub_adj = adjacency[indices][:, indices].tocsr()
    for resolution in SPLIT_RESOLUTIONS:
        labels = run_louvain(sub_adj, resolution=resolution, random_state=random_state)
        n_sub = count_clusters(labels)
        if n_sub <= 1:
            continue
        projected = current_n_clusters - 1 + n_sub
        score = (abs(projected - target_k), 0 if projected <= target_k else 1, -n_sub)
        candidates.append(
            (
                score,
                labels,
                {
                    "resolution": float(resolution),
                    "n_subclusters": int(n_sub),
                    "mode": "adjacency",
                    "projected_n_clusters": int(projected),
                },
            )
        )

    if not candidates:
        sub_features = np.asarray(features[indices], dtype=np.float32)
        feature_conn = build_feature_knn_connectivity(sub_features, n_neighbors=cfg.expr_neighbors)
        for resolution in SPLIT_RESOLUTIONS:
            labels = run_louvain(feature_conn, resolution=resolution, random_state=random_state)
            n_sub = count_clusters(labels)
            if n_sub <= 1:
                continue
            projected = current_n_clusters - 1 + n_sub
            score = (abs(projected - target_k), 0 if projected <= target_k else 1, -n_sub)
            candidates.append(
                (
                    score,
                    labels,
                    {
                        "resolution": float(resolution),
                        "n_subclusters": int(n_sub),
                        "mode": "feature_knn",
                        "projected_n_clusters": int(projected),
                    },
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, labels, info = candidates[0]
    return labels, info


def replace_cluster(labels: np.ndarray, cluster_id: int, indices: np.ndarray, sub_labels: np.ndarray) -> np.ndarray:
    new_labels = labels.copy()
    unique_sub = pd.Index(sub_labels).unique().tolist()
    next_label = int(labels.max()) + 1
    mapping: Dict[int, int] = {}
    for pos, value in enumerate(unique_sub):
        mapping[int(value)] = cluster_id if pos == 0 else next_label + pos - 1
    new_labels[indices] = np.array([mapping[int(value)] for value in sub_labels], dtype=np.int32)
    return relabel_partition(new_labels.tolist())


def split_partition_to_target(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    target_k: int,
    cfg: BuilderConfig,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    labels = relabel_partition(labels.tolist())
    split_log: List[Dict[str, object]] = []
    attempts = 0
    pbar = tqdm(
        total=max(0, target_k - count_clusters(labels)),
        desc=f"Split to K={target_k}",
        unit="split",
        leave=False,
    )

    while count_clusters(labels) < target_k:
        sizes = cluster_sizes(labels)
        order = np.argsort(-sizes)
        progress = False
        for cluster_id in order:
            if sizes[cluster_id] <= 1:
                continue
            idx = np.where(labels == cluster_id)[0]
            result = choose_best_split(
                adjacency=adjacency,
                features=features,
                indices=idx,
                current_n_clusters=count_clusters(labels),
                target_k=target_k,
                cfg=cfg,
                random_state=cfg.random_state + attempts + int(cluster_id),
            )
            if result is None:
                continue
            sub_labels, info = result
            prev_n_clusters = count_clusters(labels)
            labels = replace_cluster(labels, cluster_id=cluster_id, indices=idx, sub_labels=sub_labels)
            info["cluster_id"] = int(cluster_id)
            info["cluster_size_before"] = int(idx.size)
            info["n_clusters_after"] = int(count_clusters(labels))
            split_log.append(info)
            progress = True
            attempts += 1
            pbar.update(max(0, int(info["n_clusters_after"]) - int(prev_n_clusters)))
            break
        if not progress:
            pbar.close()
            raise RuntimeError(f"Unable to split partition from {count_clusters(labels)} to target {target_k}.")

    pbar.close()
    return relabel_partition(labels.tolist()), split_log


def merge_partition_to_target(labels: np.ndarray, features: np.ndarray, target_k: int) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    labels = relabel_partition(labels.tolist())
    merge_log: List[Dict[str, object]] = []
    pbar = tqdm(
        total=max(0, count_clusters(labels) - target_k),
        desc=f"Merge to K={target_k}",
        unit="merge",
        leave=False,
    )

    while count_clusters(labels) > target_k:
        unique = np.unique(labels)
        centroids = np.vstack([features[labels == cluster].mean(axis=0) for cluster in unique])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(norms == 0, 1.0, norms)
        sim = centroids @ centroids.T
        np.fill_diagonal(sim, -np.inf)
        row, col = np.unravel_index(np.argmax(sim), sim.shape)
        keep_cluster = int(unique[row])
        drop_cluster = int(unique[col])
        labels[labels == drop_cluster] = keep_cluster
        labels = relabel_partition(labels.tolist())
        merge_log.append(
            {
                "keep_cluster": keep_cluster,
                "drop_cluster": drop_cluster,
                "similarity": float(sim[row, col]),
                "n_clusters_after": int(count_clusters(labels)),
            }
        )
        pbar.update(1)

    pbar.close()
    return labels, merge_log


def fit_exact_k_partition(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    target_k: int,
    cfg: BuilderConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    n_obs = adjacency.shape[0]
    if target_k < 1 or target_k > n_obs:
        raise ValueError(f"Target K={target_k} is out of range for n_obs={n_obs}.")
    if target_k == n_obs:
        return np.arange(n_obs, dtype=np.int32), {"mode": "singleton", "n_clusters": int(target_k)}

    initial_labels, initial_info = search_initial_partition(adjacency, target_k=target_k, cfg=cfg)
    labels = initial_labels

    split_log: List[Dict[str, object]] = []
    merge_log: List[Dict[str, object]] = []
    if count_clusters(labels) < target_k:
        labels, split_log = split_partition_to_target(adjacency, features, labels, target_k=target_k, cfg=cfg)
    if count_clusters(labels) > target_k:
        labels, merge_log = merge_partition_to_target(labels, features, target_k=target_k)
    if count_clusters(labels) < target_k:
        labels, extra_split_log = split_partition_to_target(adjacency, features, labels, target_k=target_k, cfg=cfg)
        split_log.extend(extra_split_log)
    if count_clusters(labels) != target_k:
        raise RuntimeError(f"Failed to reach exact K={target_k}; current cluster count is {count_clusters(labels)}.")

    return relabel_partition(labels.tolist()), {
        "mode": "exact_k",
        "target_k": int(target_k),
        "initial": initial_info,
        "split_log": split_log,
        "merge_log": merge_log,
        "n_clusters": int(count_clusters(labels)),
    }


def enforce_max_cluster_size(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    cfg: BuilderConfig,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    labels = relabel_partition(labels.tolist())
    enforcement_log: List[Dict[str, object]] = []
    iteration = 0
    pbar = tqdm(
        desc=f"Enforce size<={max_size}",
        unit="split",
        leave=False,
    )

    while True:
        sizes = cluster_sizes(labels)
        oversized = [cluster for cluster, size in enumerate(sizes.tolist()) if size > max_size]
        if not oversized:
            break

        cluster_id = max(oversized, key=lambda value: sizes[value])
        idx = np.where(labels == cluster_id)[0]
        target_subclusters = int(math.ceil(idx.size / max_size))
        sub_adj = adjacency[idx][:, idx].tocsr()
        sub_features = np.asarray(features[idx], dtype=np.float32)
        sub_labels, sub_info = fit_exact_k_partition(sub_adj, sub_features, target_k=target_subclusters, cfg=cfg)
        labels = replace_cluster(labels, cluster_id=cluster_id, indices=idx, sub_labels=sub_labels)
        enforcement_log.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size_before": int(idx.size),
                "target_subclusters": int(target_subclusters),
                "n_clusters_after": int(count_clusters(labels)),
                "sub_info": sub_info,
            }
        )
        pbar.update(1)
        iteration += 1
        if iteration > adjacency.shape[0]:
            pbar.close()
            raise RuntimeError("Exceeded maximum iterations while enforcing max cluster size.")

    pbar.close()
    return labels, enforcement_log


def fit_less_than_5_partition(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    cfg: BuilderConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    initial_target = min(500, adjacency.shape[0])
    initial_labels, exact_info = fit_exact_k_partition(adjacency, features, target_k=initial_target, cfg=cfg)
    final_labels, enforcement_log = enforce_max_cluster_size(
        adjacency=adjacency,
        features=features,
        labels=initial_labels,
        max_size=cfg.less_than_5_max_size,
        cfg=cfg,
    )
    return final_labels, {
        "mode": "less_than_5",
        "seed_target_k": int(initial_target),
        "less_than_5_max_size": int(cfg.less_than_5_max_size),
        "exact_seed_info": exact_info,
        "enforcement_log": enforcement_log,
        "n_clusters": int(count_clusters(final_labels)),
    }


def format_cluster_names(labels: np.ndarray) -> Tuple[pd.DataFrame, List[int]]:
    ordered_clusters = pd.Index(labels).unique().tolist()
    width = max(3, len(str(len(ordered_clusters))))
    mapping_rows = []
    for idx, cluster_id in enumerate(ordered_clusters, start=1):
        mapping_rows.append(
            {
                "cluster_id": int(cluster_id),
                "domain_label": f"cluster_{idx:0{width}d}",
                "domain_id": f"domain_{idx:0{width}d}",
            }
        )
    return pd.DataFrame(mapping_rows), ordered_clusters


def aggregate_by_cluster(count_matrix, labels: np.ndarray, ordered_clusters: Sequence[int]) -> Tuple[object, np.ndarray]:
    n_obs = len(labels)
    cluster_to_pos = {cluster: idx for idx, cluster in enumerate(ordered_clusters)}
    group_index = np.array([cluster_to_pos[int(label)] for label in labels], dtype=np.int32)
    membership = sp.csr_matrix(
        (np.ones(n_obs, dtype=np.float32), (np.arange(n_obs), group_index)),
        shape=(n_obs, len(ordered_clusters)),
    )
    aggregated = membership.T @ count_matrix
    spot_count = np.asarray(membership.sum(axis=0)).ravel().astype(np.int32)
    return aggregated, spot_count


def aggregate_coordinates(spatial: np.ndarray, labels: np.ndarray, ordered_clusters: Sequence[int]) -> np.ndarray:
    n_obs = len(labels)
    cluster_to_pos = {cluster: idx for idx, cluster in enumerate(ordered_clusters)}
    group_index = np.array([cluster_to_pos[int(label)] for label in labels], dtype=np.int32)
    membership = sp.csr_matrix(
        (np.ones(n_obs, dtype=np.float32), (np.arange(n_obs), group_index)),
        shape=(n_obs, len(ordered_clusters)),
    )
    coord_sum = membership.T @ spatial
    counts = np.asarray(membership.sum(axis=0)).ravel()
    counts = np.where(counts == 0, 1.0, counts)
    return coord_sum / counts[:, None]


def build_assignment_table(
    spot_adata: AnnData,
    labels: np.ndarray,
    cluster_map: pd.DataFrame,
) -> pd.DataFrame:
    cluster_to_domain = dict(zip(cluster_map["cluster_id"], cluster_map["domain_id"]))
    cluster_to_label = dict(zip(cluster_map["cluster_id"], cluster_map["domain_label"]))
    spatial = require_spatial(spot_adata, Path("<memory>"))
    assignments = pd.DataFrame(
        {
            "spot_id": spot_adata.obs_names.astype(str),
            "domain_label": [cluster_to_label[int(label)] for label in labels],
            "domain_id": [cluster_to_domain[int(label)] for label in labels],
        }
    )
    if "annotation" in spot_adata.obs.columns:
        assignments["annotation"] = spot_adata.obs["annotation"].astype(str).to_numpy()
    assignments["x"] = spatial[:, 0]
    assignments["y"] = spatial[:, 1]
    return assignments


def export_domain_result(
    spot_adata: AnnData,
    count_matrix,
    labels: np.ndarray,
    output_dir: Path,
    file_stem: str,
    build_info: Dict[str, object],
) -> None:
    ensure_dir(output_dir)
    cluster_map, ordered_clusters = format_cluster_names(labels)
    assignments = build_assignment_table(spot_adata, labels, cluster_map)

    aggregated_counts, spot_count = aggregate_by_cluster(count_matrix, labels, ordered_clusters)
    domain_coords = aggregate_coordinates(require_spatial(spot_adata, Path("<memory>")), labels, ordered_clusters)

    obs = cluster_map.loc[:, ["domain_id", "domain_label"]].copy()
    obs["spot_count"] = spot_count.astype(float)
    obs.index = pd.Index(obs["domain_id"].tolist(), name="domain_id")
    domain_adata = AnnData(X=aggregated_counts, obs=obs, var=spot_adata.var.copy())
    domain_adata.obsm["spatial"] = np.asarray(domain_coords, dtype=np.float32)
    domain_adata.layers["count"] = aggregated_counts.copy() if hasattr(aggregated_counts, "copy") else aggregated_counts
    domain_adata.uns["X_name"] = "counts"
    domain_adata.uns["domain_label"] = obs["domain_label"].astype(str).to_numpy()
    domain_adata.uns["cluster_labels_spot_level"] = assignments["domain_label"].tolist()
    domain_adata.uns["builder_info_json"] = json.dumps(build_info, ensure_ascii=False)

    domain_adata.write_h5ad(output_dir / f"{file_stem}.h5ad")
    assignments.to_csv(output_dir / f"{file_stem}_spot_domain_map.csv", index=False)

    if "annotation" in assignments.columns:
        organ_counts = pd.crosstab(assignments["domain_id"], assignments["annotation"]).reset_index()
        organ_counts.to_csv(output_dir / f"{file_stem}_domain_organ_counts.csv", index=False)

    spots_with_domain = spot_adata.copy()
    spots_with_domain.obs["domain_label"] = assignments["domain_label"].to_numpy()
    spots_with_domain.obs["domain_id"] = assignments["domain_id"].to_numpy()
    spots_with_domain.write_h5ad(output_dir / f"{file_stem}_spots_with_domain.h5ad")

    cluster_sizes_series = assignments.groupby("domain_id")["spot_id"].count().sort_values()
    cluster_sizes_series.to_csv(output_dir / f"{file_stem}_cluster_sizes.csv", header=["spot_count"])

    summary = {
        "file_stem": file_stem,
        "n_spots": int(spot_adata.n_obs),
        "n_domains": int(len(ordered_clusters)),
        "min_cluster_size": int(cluster_sizes_series.min()),
        "median_cluster_size": float(cluster_sizes_series.median()),
        "max_cluster_size": int(cluster_sizes_series.max()),
        "build_info": build_info,
    }
    with (output_dir / f"{file_stem}_build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def process_sample(sample_path: Path, cfg: BuilderConfig) -> None:
    sample_name = sample_path.stem
    print(f"[Build] Processing {sample_name}")
    spot_adata = read_h5ad(sample_path)
    require_spatial(spot_adata, sample_path)

    analysis_adata, count_matrix = make_analysis_adata(spot_adata, cfg)
    cci_total = load_cci_total(sample_name, spot_adata.obs_names.astype(str), cfg)
    expr_conn, expr_pca = build_expression_connectivity(analysis_adata, cfg)
    cci_conn = build_cci_connectivity(cci_total, cfg)
    fused_conn = fuse_connectivities(expr_conn, cci_conn, cfg)
    merge_features = build_merge_features(expr_pca, cci_total)

    for target_k in tqdm(cfg.k_values, desc=f"{sample_name} K", unit="k", leave=False):
        labels, build_info = fit_exact_k_partition(fused_conn, merge_features, target_k=target_k, cfg=cfg)
        output_dir = cfg.output_root / str(target_k)
        file_stem = f"{sample_name}_domainK{target_k}"
        export_domain_result(
            spot_adata=spot_adata,
            count_matrix=count_matrix,
            labels=labels,
            output_dir=output_dir,
            file_stem=file_stem,
            build_info=build_info,
        )
        print(f"[Build] {sample_name} -> K={target_k} done")

    if not cfg.k_values:
        print(f"[Build] {sample_name} -> fixed K skipped")

    labels, build_info = fit_less_than_5_partition(fused_conn, merge_features, cfg=cfg)
    output_dir = cfg.output_root / "less_than_5"
    file_stem = f"{sample_name}_domainLessThan5"
    export_domain_result(
        spot_adata=spot_adata,
        count_matrix=count_matrix,
        labels=labels,
        output_dir=output_dir,
        file_stem=file_stem,
        build_info=build_info,
    )
    print(f"[Build] {sample_name} -> less_than_5 done")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build domain-level h5ad files with Louvain on expression + COMMOT CCI."
    )
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DIR)
    parser.add_argument("--local-commot-dir", type=Path, default=LOCAL_COMMOT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Optional exact-K outputs. Default is empty, so the script only builds less_than_5 results for all samples.",
    )
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--expr-neighbors", type=int, default=30)
    parser.add_argument("--cci-topk", type=int, default=30)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--expr-weight", type=float, default=0.5)
    parser.add_argument("--cci-weight", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser


def build_config_from_args(args: argparse.Namespace) -> BuilderConfig:
    return BuilderConfig(
        local_dir=args.local_dir,
        local_commot_dir=args.local_commot_dir,
        output_root=args.output_root,
        k_values=tuple(args.k_values or ()),
        sample_names=tuple(args.sample_names),
        expr_neighbors=int(args.expr_neighbors),
        cci_topk=int(args.cci_topk),
        n_top_genes=int(args.n_top_genes),
        n_pcs=int(args.n_pcs),
        expr_weight=float(args.expr_weight),
        cci_weight=float(args.cci_weight),
        random_state=int(args.random_state),
    )


def main() -> None:
    args = build_argparser().parse_args()
    cfg = build_config_from_args(args)

    ensure_dir(cfg.output_root)
    with (cfg.output_root / "domain_builder_louvain_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2, default=str)

    sample_files = list_local_files(cfg.local_dir, cfg.sample_names)
    if not sample_files:
        raise FileNotFoundError(f"No local h5ad files found under {cfg.local_dir}.")

    print(f"[Build] Found {len(sample_files)} local samples")
    for sample_path in tqdm(sample_files, desc="Samples", unit="sample"):
        process_sample(sample_path, cfg)
    print("[Build] All Louvain domain files finished")


if __name__ == "__main__":
    main()
