"""Align natural and learned domain labels to the original Spot coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import pandas as pd

from mignet_ce.io.h5ad_h5py import read_axis_index, read_h5ad_spatial
from mignet_ce.io.organ_paths import resolve_organ_paths, sample_stem


ORGAN = "heart"
INPUT_LABELS = {"spot": "Spot", "seurat_k150": "Seurat K150", "seurat_k40": "Seurat K40"}
TARGET_K = {"spot": 40, "seurat_k150": 40, "seurat_k40": 10}
METHOD_ORDER = (
    "maturity_cci_grn_two_stage",
    "complete_combined_coarse_maturity_cci_grn",
    "two_stage_noKeff",
    "two_stage_noKeff_HardEI",
)


@dataclass(frozen=True)
class StageMaps:
    stage: str
    coordinates: pd.DataFrame
    seurat: dict[str, pd.Series]


@dataclass(frozen=True)
class LearnedDomains:
    labels: pd.Series
    keff: float


def _check_ids(ids: pd.Index, path: Path) -> None:
    if ids.empty or ids.hasnans or ids.has_duplicates or (ids.str.strip() == "").any():
        raise ValueError(f"Empty, missing or duplicate IDs in {path}")


def _unit_ids(path: Path) -> pd.Index:
    with h5py.File(path, "r") as handle:
        ids = pd.Index(read_axis_index(handle, "obs", path), dtype=str)
    _check_ids(ids, path)
    return ids


def load_stage(data_root: Path, stage: str) -> StageMaps:
    spot_path = resolve_organ_paths(data_root, "spot", ORGAN, stage)["h5ad"]
    ids = _unit_ids(spot_path)
    spatial = read_h5ad_spatial(spot_path)
    if spatial is None or spatial.shape != (len(ids), 2) or not np.isfinite(spatial).all():
        raise ValueError(f"Missing or invalid spatial coordinates in {spot_path}")
    coordinates = pd.DataFrame(spatial, index=ids, columns=["x", "y"])

    seurat = {}
    for scale, expected_domains in (("seurat_k150", 150), ("seurat_k40", 40)):
        path = resolve_organ_paths(data_root, scale, ORGAN, stage)["map"]
        frame = pd.read_csv(path, dtype={"spot_id": str, "domain_id": str}).set_index("spot_id")
        _check_ids(frame.index, path)
        if set(frame.index) != set(ids):
            raise ValueError(f"Spot IDs differ between {path} and {spot_path}")
        frame = frame.reindex(ids)
        # H5AD stores coordinates as float32 while the CSV preserves extra precision.
        if not np.allclose(frame[["x", "y"]], coordinates, atol=5e-5, rtol=0):
            raise ValueError(f"Spot coordinates differ between {path} and {spot_path}")
        labels = frame["domain_id"]
        if labels.isna().any() or labels.str.strip().eq("").any() or labels.nunique() != expected_domains:
            raise ValueError(f"Expected {expected_domains} populated domains in {path}")
        seurat[scale] = labels
    return StageMaps(stage, coordinates, seurat)


def _load_projection(
    run_dir: Path, side: str, input_ids: pd.Index, spot_to_unit: pd.Series, k: int, keff: float,
) -> LearnedDomains:
    path = run_dir / f"assignments_{side}.csv"
    assignments = pd.read_csv(path, dtype={"spot_id": str}).set_index("spot_id")
    _check_ids(assignments.index, path)
    if not assignments.index.equals(input_ids):
        raise ValueError(f"Assignment ID order differs from the input H5AD in {path}")

    soft = np.load(run_dir / f"S_{side}.npy", allow_pickle=False)
    if soft.shape != (len(input_ids), k) or (soft < 0).any():
        raise ValueError(f"Invalid soft assignment for {path}")
    hard = soft.argmax(axis=1)
    if not np.array_equal(hard, assignments["hard_cluster"].to_numpy()):
        raise ValueError(f"Hard cluster labels disagree with soft assignments in {path}")
    if set(spot_to_unit) != set(input_ids):
        raise ValueError(f"Spatial mapping and trained input IDs differ for {path}")
    return LearnedDomains(spot_to_unit.map(pd.Series(hard, index=input_ids)), keff)


def load_comparisons(
    data_root: Path, optimal_root: Path, stages: tuple[StageMaps, StageMaps],
) -> dict[str, dict[str, tuple[LearnedDomains, LearnedDomains]]]:
    """Load exactly the selected 12 formal runs, with no checkpoint reselection."""
    from mignet_ce.downstream.formal_core_analysis.tables import inspect_optimal_run

    pair = f"{stages[0].stage}->{stages[1].stage}"
    pair_dir = pair.replace("->", "_to_")
    paths = {
        (scale, method): optimal_root / method / scale / f"K{k}" / pair_dir
        for scale, k in TARGET_K.items() for method in METHOD_ORDER
    }
    missing = [str(path) for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing formal runs:\n  " + "\n  ".join(missing))

    results = {}
    for scale, k in TARGET_K.items():
        input_ids = tuple(
            stage.coordinates.index if scale == "spot" else
            _unit_ids(resolve_organ_paths(data_root, scale, ORGAN, stage.stage)["h5ad"])
            for stage in stages
        )
        spot_to_unit = tuple(
            stage.seurat[scale] if scale != "spot" else
            pd.Series(stage.coordinates.index, index=stage.coordinates.index)
            for stage in stages
        )
        results[scale] = {}
        for method in METHOD_ORDER:
            run = paths[scale, method]
            audit = inspect_optimal_run(run)  # Also verifies Keff from the persisted soft matrices.
            if (audit["method"], audit["input_scale"], audit["time_pair"], audit["K"]) != (method, scale, pair, k):
                raise ValueError(f"Recorded method, scale, time pair or K disagrees with {run}")
            checkpoint = (
                "best_model.pt" if method == "complete_combined_coarse_maturity_cci_grn" else "best_joint.pt"
            )
            if not (run / checkpoint).is_file():
                raise FileNotFoundError(run / checkpoint)
            feature = json.loads((run / "feature_manifest.json").read_text(encoding="utf-8"))
            provenance = feature["provenance"]
            for side, stage in zip(("t", "tp"), stages):
                expected = f"{sample_stem(scale, ORGAN, stage.stage)}.h5ad"
                if PureWindowsPath(provenance.get(f"h5ad_{side}", "")).name != expected:
                    raise ValueError(f"Recorded {side} input sample differs from {expected} in {run}")
            results[scale][method] = (
                _load_projection(run, "t", input_ids[0], spot_to_unit[0], k, audit["Keff_t"]),
                _load_projection(run, "tp", input_ids[1], spot_to_unit[1], k, audit["Keff_tp"]),
            )
    return results
