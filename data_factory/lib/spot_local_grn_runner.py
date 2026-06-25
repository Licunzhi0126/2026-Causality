from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad
from sklearn.neighbors import NearestNeighbors

from factory_common import ensure_dir
from grn_layer_runner import (
    DEFAULT_N_TREES,
    DEFAULT_THREADS,
    DEFAULT_TOP_EDGE_COUNT,
    DEFAULT_TOP_HVG,
    configure_grn_runtime,
    infer_grn_edges_from_adata,
)
from unit_grn_layer_runner import normalize_edge_weights


def spatial_coordinates(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        return coords[:, :2]
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs.loc[:, ["x", "y"]].to_numpy(dtype=float)
    raise ValueError("Spot h5ad requires obsm['spatial'] or obs columns ['x', 'y'].")


def select_spatial_neighbors(
    unit_ids: Sequence[str],
    coords: np.ndarray,
    selected_units: Sequence[str],
    *,
    k_neighbors: int = 50,
) -> dict[str, pd.DataFrame]:
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive.")
    ids = list(map(str, unit_ids))
    id_to_index = {unit: idx for idx, unit in enumerate(ids)}
    if len(ids) < 2:
        raise ValueError("At least two spots/cells are required.")
    actual_k = min(int(k_neighbors), len(ids) - 1)
    model = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean").fit(
        np.asarray(coords, dtype=float)[:, :2]
    )
    result: dict[str, pd.DataFrame] = {}
    for center in map(str, selected_units):
        if center not in id_to_index:
            raise ValueError(f"Selected center unit {center!r} is not present in the h5ad.")
        center_idx = id_to_index[center]
        distances, indices = model.kneighbors(
            np.asarray(coords, dtype=float)[center_idx : center_idx + 1, :2]
        )
        rows = []
        rank = 0
        for distance, neighbor_idx in zip(distances[0], indices[0]):
            neighbor = ids[int(neighbor_idx)]
            if neighbor == center:
                continue
            rank += 1
            rows.append(
                {
                    "center_unit_id": center,
                    "neighbor_unit_id": neighbor,
                    "spatial_distance": float(distance),
                    "neighbor_rank": rank,
                    "used_in_grn": True,
                }
            )
        result[center] = pd.DataFrame(rows)
    return result


def infer_spot_local_grn_tables(
    adata,
    selected_units: Sequence[str],
    *,
    k_neighbors: int = 50,
    include_center: bool = True,
    min_cells: int = 30,
    infer_fn: Callable = infer_grn_edges_from_adata,
    grn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = adata.obs_names.astype(str).tolist()
    id_to_index = {unit: idx for idx, unit in enumerate(ids)}
    neighbor_tables = select_spatial_neighbors(
        ids,
        spatial_coordinates(adata),
        selected_units,
        k_neighbors=k_neighbors,
    )
    edge_parts: List[pd.DataFrame] = []
    summary_rows: List[dict[str, object]] = []
    all_neighbor_rows: List[pd.DataFrame] = []
    for center, neighbors in neighbor_tables.items():
        all_neighbor_rows.append(neighbors)
        local_ids = neighbors["neighbor_unit_id"].astype(str).tolist()
        if include_center:
            local_ids = [center, *local_ids]
        if len(local_ids) < min_cells:
            summary_rows.append(
                {
                    "center_unit_id": center,
                    "n_neighbors": int(len(neighbors)),
                    "include_center": bool(include_center),
                    "neighbor_mode": "spatial",
                    "status": "skipped",
                    "reason": f"local_cells={len(local_ids)} < min_cells={min_cells}",
                }
            )
            continue
        try:
            local_indices = [id_to_index[unit] for unit in local_ids]
            local_adata = adata[local_indices].copy()
            edges, metadata = infer_fn(local_adata, grn)
            edges = normalize_edge_weights(edges)
            edges.insert(0, "center_unit_id", center)
            edges["n_neighbors"] = int(len(neighbors))
            edges["include_center"] = bool(include_center)
            edges["neighbor_mode"] = "spatial"
            edges["status"] = "written"
            edge_parts.append(edges)
            summary_rows.append(
                {
                    "center_unit_id": center,
                    "n_neighbors": int(len(neighbors)),
                    "include_center": bool(include_center),
                    "neighbor_mode": "spatial",
                    **metadata,
                    "status": "written",
                    "reason": "",
                }
            )
            del local_adata, edges
            gc.collect()
        except Exception as exc:
            summary_rows.append(
                {
                    "center_unit_id": center,
                    "n_neighbors": int(len(neighbors)),
                    "include_center": bool(include_center),
                    "neighbor_mode": "spatial",
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    edge_columns = [
        "center_unit_id",
        "regulator",
        "target",
        "weight",
        "weight_norm",
        "n_neighbors",
        "include_center",
        "neighbor_mode",
        "status",
    ]
    edges = (
        pd.concat(edge_parts, ignore_index=True).reindex(columns=edge_columns)
        if edge_parts
        else pd.DataFrame(columns=edge_columns)
    )
    neighbors = (
        pd.concat(all_neighbor_rows, ignore_index=True)
        if all_neighbor_rows
        else pd.DataFrame(
            columns=[
                "center_unit_id",
                "neighbor_unit_id",
                "spatial_distance",
                "neighbor_rank",
                "used_in_grn",
            ]
        )
    )
    return edges, neighbors, pd.DataFrame(summary_rows)


def run_spot_local_grn_pilot(
    *,
    spot_h5ad: Path,
    selected_units: Sequence[str],
    output_root: Path,
    k_neighbors: int = 50,
    include_center: bool = True,
    min_cells: int = 30,
    threads: int = DEFAULT_THREADS,
    n_trees: int = DEFAULT_N_TREES,
    top_hvg: int = DEFAULT_TOP_HVG,
    top_edge_count: int = DEFAULT_TOP_EDGE_COUNT,
    tf_list: Path | None = None,
) -> None:
    import GRN_global as grn

    configure_grn_runtime(
        grn,
        threads=threads,
        n_trees=n_trees,
        top_hvg=top_hvg,
        top_edge_count=top_edge_count,
        tf_list=tf_list,
    )
    ensure_dir(output_root)
    adata = read_h5ad(spot_h5ad)
    edges, neighbors, summary = infer_spot_local_grn_tables(
        adata,
        selected_units,
        k_neighbors=k_neighbors,
        include_center=include_center,
        min_cells=min_cells,
        grn=grn,
    )
    edges.to_csv(output_root / "spot_local_grn_edges.csv", index=False)
    neighbors.to_csv(output_root / "spot_local_neighbors.csv", index=False)
    summary.to_csv(output_root / "spot_local_grn_summary.csv", index=False)
