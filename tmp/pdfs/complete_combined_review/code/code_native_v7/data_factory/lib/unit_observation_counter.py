from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


UNIT_AUXILIARY_SUFFIX = "_spots_with_domain.h5ad"


def _layer_input_root(data_root: Path, layer: str) -> Path:
    root = Path(data_root)
    return root if root.name == layer else root / layer


def discover_unit_grn_input_files(data_root: Path, layer: str) -> list[Path]:
    layer_root = _layer_input_root(data_root, layer)
    if layer == "spot":
        return sorted(
            path
            for path in layer_root.rglob("spot_*_*.h5ad")
            if path.is_file()
            and not path.name.endswith(UNIT_AUXILIARY_SUFFIX)
            and not path.name.endswith("_COMMOT.h5ad")
        )
    return sorted(
        path
        for path in layer_root.rglob(f"*{UNIT_AUXILIARY_SUFFIX}")
        if path.is_file()
    )


def sample_name_from_unit_grn_input(path: Path, layer: str) -> str:
    if layer == "spot":
        return path.stem
    if not path.name.endswith(UNIT_AUXILIARY_SUFFIX):
        raise ValueError(
            f"Domain unit GRN input must end with {UNIT_AUXILIARY_SUFFIX!r}: {path}"
        )
    return path.name.removesuffix(UNIT_AUXILIARY_SUFFIX)


def spatial_coordinates(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("obsm['spatial'] must be a two-dimensional array with at least two columns.")
        return coords[:, :2]
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs.loc[:, ["x", "y"]].to_numpy(dtype=float)
    raise ValueError("Spot h5ad requires obsm['spatial'] or obs columns ['x', 'y'].")


def build_spatial_neighbor_tables(
    unit_ids: Sequence[str],
    coords: np.ndarray,
    *,
    k_neighbors: int = 50,
) -> Dict[str, pd.DataFrame]:
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive.")
    ids = list(map(str, unit_ids))
    if len(ids) < 2:
        raise ValueError("At least two spots/cells are required.")
    coordinates = np.asarray(coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(ids) or coordinates.shape[1] < 2:
        raise ValueError(
            f"Coordinate shape {coordinates.shape} does not match {len(ids)} unit IDs."
        )
    if not np.all(np.isfinite(coordinates[:, :2])):
        raise ValueError("Spatial coordinates contain non-finite values.")

    actual_k = min(int(k_neighbors), len(ids) - 1)
    model = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean").fit(
        coordinates[:, :2]
    )
    distances, indices = model.kneighbors(coordinates[:, :2])
    result: Dict[str, pd.DataFrame] = {}
    for center_index, center in enumerate(ids):
        rows = []
        rank = 0
        for distance, neighbor_index in zip(distances[center_index], indices[center_index]):
            if int(neighbor_index) == center_index:
                continue
            rank += 1
            rows.append(
                {
                    "unit_id": center,
                    "neighbor_unit_id": ids[int(neighbor_index)],
                    "spatial_distance": float(distance),
                    "neighbor_rank": rank,
                    "used_in_grn": True,
                }
            )
            if rank >= actual_k:
                break
        result[center] = pd.DataFrame(
            rows,
            columns=[
                "unit_id",
                "neighbor_unit_id",
                "spatial_distance",
                "neighbor_rank",
                "used_in_grn",
            ],
        )
    return result


def count_domain_unit_observations(
    adata,
    *,
    unit_column: str = "domain_id",
    threshold: int = 30,
) -> pd.DataFrame:
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")
    if unit_column not in adata.obs.columns:
        raise ValueError(f"Input h5ad is missing obs[{unit_column!r}].")
    counts = adata.obs[unit_column].astype(str).value_counts().sort_index()
    return pd.DataFrame(
        {
            "unit_id": counts.index.astype(str),
            "n_observations": counts.to_numpy(dtype=int),
            "below_threshold": counts.to_numpy(dtype=int) < int(threshold),
            "threshold": int(threshold),
            "unit_source": f"obs[{unit_column!r}]",
        }
    )


def count_spot_unit_observations(
    adata,
    *,
    spot_k_neighbors: int = 50,
    include_center: bool = True,
    threshold: int = 30,
) -> pd.DataFrame:
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")
    unit_ids = adata.obs_names.astype(str).tolist()
    neighbor_tables = build_spatial_neighbor_tables(
        unit_ids,
        spatial_coordinates(adata),
        k_neighbors=spot_k_neighbors,
    )
    n_observations = np.asarray(
        [
            len(neighbor_tables[unit_id]) + (1 if include_center else 0)
            for unit_id in unit_ids
        ],
        dtype=int,
    )
    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "n_observations": n_observations,
            "below_threshold": n_observations < int(threshold),
            "threshold": int(threshold),
            "unit_source": "obs_names_with_spatial_neighbors",
        }
    )


def summarize_unit_observation_counts(counts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "layer",
        "sample",
        "n_observations",
        "below_threshold",
        "threshold",
        "input_file",
    }
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"Observation count table is missing columns {sorted(missing)}.")
    rows = []
    for (layer, sample, input_file), sub in counts.groupby(
        ["layer", "sample", "input_file"],
        sort=True,
        dropna=False,
    ):
        values = sub["n_observations"].to_numpy(dtype=float)
        below = sub["below_threshold"].astype(bool).to_numpy()
        rows.append(
            {
                "layer": str(layer),
                "sample": str(sample),
                "n_units": int(len(sub)),
                "n_units_below_threshold": int(below.sum()),
                "below_threshold_ratio": float(below.mean()) if len(sub) else 0.0,
                "min_observations": float(np.min(values)) if len(values) else np.nan,
                "median_observations": float(np.median(values)) if len(values) else np.nan,
                "mean_observations": float(np.mean(values)) if len(values) else np.nan,
                "max_observations": float(np.max(values)) if len(values) else np.nan,
                "threshold": int(sub["threshold"].iloc[0]),
                "input_file": str(input_file),
            }
        )
    return pd.DataFrame(rows)
