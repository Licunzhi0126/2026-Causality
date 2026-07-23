"""Lightweight domain output contract shared by builders without Scanpy imports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from factory_common import ensure_dir


def require_spatial(adata: AnnData, file_path: Path) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError(f"{file_path} is missing obsm['spatial'].")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape != (adata.n_obs, 2):
        raise ValueError(f"{file_path} has invalid spatial shape {spatial.shape}; expected ({adata.n_obs}, 2).")
    if not np.isfinite(spatial).all():
        raise ValueError(f"{file_path} contains non-finite spatial coordinates.")
    return spatial


def choose_count_matrix(adata: AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def format_cluster_names(labels: np.ndarray) -> Tuple[pd.DataFrame, list[int]]:
    ordered_clusters = [int(value) for value in pd.Index(labels).unique().tolist()]
    width = max(3, len(str(len(ordered_clusters))))
    rows = [
        {
            "cluster_id": cluster_id,
            "domain_label": f"cluster_{index:0{width}d}",
            "domain_id": f"domain_{index:0{width}d}",
        }
        for index, cluster_id in enumerate(ordered_clusters, start=1)
    ]
    return pd.DataFrame(rows), ordered_clusters


def _membership(labels: np.ndarray, ordered_clusters: Sequence[int]) -> sp.csr_matrix:
    cluster_to_pos = {int(cluster): index for index, cluster in enumerate(ordered_clusters)}
    group_index = np.array([cluster_to_pos[int(label)] for label in labels], dtype=np.int32)
    return sp.csr_matrix(
        (
            np.ones(len(labels), dtype=np.float32),
            (np.arange(len(labels)), group_index),
        ),
        shape=(len(labels), len(ordered_clusters)),
    )


def aggregate_by_cluster(
    count_matrix,
    labels: np.ndarray,
    ordered_clusters: Sequence[int],
) -> Tuple[object, np.ndarray]:
    membership = _membership(labels, ordered_clusters)
    aggregated = membership.T @ count_matrix
    spot_count = np.asarray(membership.sum(axis=0)).ravel().astype(np.int32)
    return aggregated, spot_count


def aggregate_coordinates(
    spatial: np.ndarray,
    labels: np.ndarray,
    ordered_clusters: Sequence[int],
) -> np.ndarray:
    membership = _membership(labels, ordered_clusters)
    coordinate_sum = membership.T @ spatial
    counts = np.asarray(membership.sum(axis=0)).ravel()
    counts = np.where(counts == 0, 1.0, counts)
    return np.asarray(coordinate_sum / counts[:, None])


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
    """Write the same six-artifact domain contract used by existing builders."""
    ensure_dir(output_dir)
    labels = np.asarray(labels, dtype=np.int32)
    cluster_map, ordered_clusters = format_cluster_names(labels)
    assignments = build_assignment_table(spot_adata, labels, cluster_map)

    aggregated_counts, spot_count = aggregate_by_cluster(
        count_matrix,
        labels,
        ordered_clusters,
    )
    domain_coords = aggregate_coordinates(
        require_spatial(spot_adata, Path("<memory>")),
        labels,
        ordered_clusters,
    )

    obs = cluster_map.loc[:, ["domain_id", "domain_label"]].copy()
    obs["spot_count"] = spot_count.astype(float)
    obs.index = pd.Index(obs["domain_id"].tolist(), name="domain_id")
    domain_adata = AnnData(X=aggregated_counts, obs=obs, var=spot_adata.var.copy())
    domain_adata.obsm["spatial"] = np.asarray(domain_coords, dtype=np.float32)
    domain_adata.layers["count"] = (
        aggregated_counts.copy()
        if hasattr(aggregated_counts, "copy")
        else aggregated_counts
    )
    domain_adata.uns["X_name"] = "counts"
    domain_adata.uns["domain_label"] = obs["domain_label"].astype(str).to_numpy()
    domain_adata.uns["cluster_labels_spot_level"] = assignments["domain_label"].tolist()
    domain_adata.uns["builder_info_json"] = json.dumps(build_info, ensure_ascii=False)

    domain_adata.write_h5ad(output_dir / f"{file_stem}.h5ad")
    assignments.to_csv(output_dir / f"{file_stem}_spot_domain_map.csv", index=False)

    if "annotation" in assignments.columns:
        organ_counts = pd.crosstab(
            assignments["domain_id"],
            assignments["annotation"],
        ).reset_index()
        organ_counts.to_csv(
            output_dir / f"{file_stem}_domain_organ_counts.csv",
            index=False,
        )

    spots_with_domain = spot_adata.copy()
    spots_with_domain.obs["domain_label"] = assignments["domain_label"].to_numpy()
    spots_with_domain.obs["domain_id"] = assignments["domain_id"].to_numpy()
    spots_with_domain.write_h5ad(output_dir / f"{file_stem}_spots_with_domain.h5ad")

    cluster_sizes = assignments.groupby("domain_id")["spot_id"].count().sort_values()
    cluster_sizes.to_csv(
        output_dir / f"{file_stem}_cluster_sizes.csv",
        header=["spot_count"],
    )
    summary = {
        "file_stem": file_stem,
        "n_spots": int(spot_adata.n_obs),
        "n_domains": int(len(ordered_clusters)),
        "min_cluster_size": int(cluster_sizes.min()),
        "median_cluster_size": float(cluster_sizes.median()),
        "max_cluster_size": int(cluster_sizes.max()),
        "build_info": build_info,
    }
    with (output_dir / f"{file_stem}_build_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
