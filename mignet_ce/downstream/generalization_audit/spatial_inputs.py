"""Read-only spatial inputs for separate Seurat and two-stage figures.

Preview groups are geometric placeholders, never a replacement for either fit.
Real learned figures use hard labels only for display; saved soft assignments
and EI values are checked without changing the scientific artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
import re

import h5py
import numpy as np
import pandas as pd

from mignet_ce.io.h5ad_h5py import decode_values, read_axis_index, read_h5ad_spatial
from mignet_ce.io.organ_paths import resolve_organ_paths, sample_stem
from .runner import HIERARCHIES

TWO_STAGE = "maturity_cci_grn_two_stage"


def checked_frame(frame: pd.DataFrame, id_column: str, name: str) -> pd.DataFrame:
    columns = [id_column, "domain_id", "x", "y"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    out = frame[columns].copy()
    if out.empty or out.isna().any().any():
        raise ValueError(f"{name}: empty data or missing IDs, labels or coordinates")
    for column in (id_column, "domain_id"):
        out[column] = out[column].astype(str)
        if out[column].str.strip().eq("").any():
            raise ValueError(f"{name}: blank {column}")
    if out[id_column].duplicated().any():
        raise ValueError(f"{name}: duplicate {id_column}")
    for column in ("x", "y"):
        out[column] = pd.to_numeric(out[column], errors="raise")
    if not np.isfinite(out[["x", "y"]].to_numpy(float)).all():
        raise ValueError(f"{name}: non-finite coordinates")
    return out


@dataclass
class SpatialStage:
    # Original spots retain the organ outline and define upper-domain centroids.
    spots: pd.DataFrame
    # One row per lower input unit: Spot, or a Seurat-domain centroid.
    nodes: pd.DataFrame
    audit: dict

    def __post_init__(self) -> None:
        self.spots = checked_frame(self.spots, "spot_id", "spatial spots")
        self.nodes = checked_frame(self.nodes, "unit_id", "lower nodes")
        if not set(self.nodes["domain_id"]) <= set(self.spots["domain_id"]):
            raise ValueError("A lower node refers to an absent upper domain")


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.replace("->", ":").split(":")
    if (len(parts) != 2 or not all(re.fullmatch(r"\d+(?:\.\d+)?", s) for s in parts)
            or not float(parts[0]) < float(parts[1])):
        raise ValueError("Expected a forward time pair, e.g. 12.5:13.5")
    return parts[0], parts[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _same_geometry(a: pd.DataFrame, b: pd.DataFrame, name: str) -> None:
    if set(a["spot_id"]) != set(b["spot_id"]):
        raise ValueError(f"{name}: Spot ID sets differ (no spots may be dropped)")
    aa, bb = a.set_index("spot_id"), b.set_index("spot_id")
    if not np.allclose(aa.loc[bb.index, ["x", "y"]].to_numpy(float),
                       bb[["x", "y"]].to_numpy(float), atol=1e-5, rtol=0):
        raise ValueError(f"{name}: coordinates disagree for matched Spot IDs")


def load_raw_spots(raw_root: Path, organ: str, stage: str) -> tuple[pd.DataFrame, dict]:
    """Read coordinates and annotation only, including old MOSTA categoricals."""
    path = Path(raw_root) / f"E{stage}_E1S1.MOSTA.h5ad"
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        node = obs["annotation"]
        if isinstance(node, h5py.Group):
            categories, codes = decode_values(node["categories"][()]), node["codes"][()]
        elif "__categories" in obs and "annotation" in obs["__categories"]:
            categories = decode_values(obs["__categories"]["annotation"][()])
            codes = node[()]
        elif "categories" in node.attrs:
            categories = decode_values(handle[node.attrs["categories"]][()])
            codes = node[()]
        else:
            categories, codes = None, None
        if categories is None:
            labels = np.asarray(decode_values(node[()]), dtype=object)
        else:
            codes = np.asarray(codes)
            if np.any(codes < -1) or np.any(codes >= len(categories)):
                raise ValueError(f"Invalid annotation category codes: {path}")
            labels = np.asarray([categories[int(c)] if c >= 0 else "" for c in codes], dtype=object)
        aliases = {organ.casefold(), f"{organ.casefold()} primordium"}
        mask = np.asarray([str(v).casefold() in aliases for v in labels])
        ids = np.asarray(read_axis_index(handle, "obs", path), dtype=object)
    coords = read_h5ad_spatial(path)
    if coords is None or len(ids) != len(labels) or len(coords) != len(ids):
        raise ValueError(f"Spatial coordinates, annotation and index disagree: {path}")
    frame = pd.DataFrame({"spot_id": ids[mask], "x": coords[mask, 0], "y": coords[mask, 1]})
    checked_frame(frame.assign(domain_id="raw"), "spot_id", str(path))
    return frame, {"path": str(path.resolve()), "bytes": path.stat().st_size,
                   "organ": organ, "stage": stage, "annotation_labels": sorted(set(labels[mask])),
                   "slice_spots": len(ids), "organ_spots": len(frame), "subsampled": False}


def _map(data_root: Path, layer: str, organ: str, stage: str) -> tuple[pd.DataFrame, Path]:
    path = resolve_organ_paths(data_root, layer, organ, stage)["map"]
    frame = pd.read_csv(path, dtype={"spot_id": str, "domain_id": str})
    return checked_frame(frame, "spot_id", str(path)), path


def _coverage(spots: pd.DataFrame, raw_root: Path | None, organ: str, stage: str) -> dict:
    if raw_root is None:
        return {"full_local_organ": None, "server_identity_verified": False}
    raw, info = load_raw_spots(raw_root, organ, stage)
    _same_geometry(spots, raw, f"{organ} E{stage} raw-organ coverage")
    return {"full_local_organ": True, "server_identity_verified": False, "raw": info}


def load_seurat_stage(data_root: Path, organ: str, stage: str, hierarchy: str,
                      raw_root: Path | None = None) -> SpatialStage:
    _, lower, k = HIERARCHIES[hierarchy]
    upper, upper_path = _map(data_root, f"seurat_k{k}", organ, stage)
    paths = [str(upper_path.resolve())]
    if upper["domain_id"].nunique() != k:
        raise ValueError(f"{upper_path}: expected {k} occupied Seurat domains")
    if lower == "spot":
        nodes = upper.rename(columns={"spot_id": "unit_id"})
        rule = "Each Spot -> its Seurat-domain centroid"
    else:
        low, lower_path = _map(data_root, lower, organ, stage)
        if low["domain_id"].nunique() != 150:
            raise ValueError(f"{lower_path}: expected 150 occupied lower Seurat domains")
        _same_geometry(low, upper, f"{organ} E{stage} lower/upper maps")
        joined = low.rename(columns={"domain_id": "unit_id"}).merge(
            upper[["spot_id", "domain_id"]], on="spot_id", validate="one_to_one")
        centers = joined.groupby("unit_id", sort=True)[["x", "y"]].mean()
        dominant = pd.crosstab(joined["unit_id"], joined["domain_id"]).idxmax(axis=1)
        nodes = centers.assign(domain_id=dominant).reset_index()
        paths.append(str(lower_path.resolve()))
        rule = "Each K150 centroid -> maximum-Spot-overlap K40 centroid (independent Seurat fits)"
    return SpatialStage(upper, nodes, {"method": "seurat", "illustrative": False,
                        "mapping_rule": rule, "input_files": paths,
                        "coverage": _coverage(upper, raw_root, organ, stage)})


def resolve_two_stage_run(root: Path, organ: str, pair: str, hierarchy: str) -> Path:
    """Select one successful audit job; relocated server run folders are supported."""
    root = Path(root)
    if not (root / "manifest.json").is_file():
        root = root / "two_stage"
    state = _json(root / "manifest.json")
    source, target = parse_pair(pair)
    if state.get("protocol") != "organ_independent_trained_deltaei_v1" or state.get("method") != TWO_STAGE:
        raise ValueError("Expected a two-stage generalization-audit manifest")
    matches = [job for job in state["jobs"] if job.get("organ") == organ
               and job.get("time_pair") == f"{source}->{target}"
               and job.get("hierarchy") == hierarchy and job.get("method") == TWO_STAGE]
    if len(matches) != 1 or matches[0].get("status") != "success":
        raise ValueError("Expected exactly one successful two-stage job for this selection")
    attempt = PureWindowsPath(matches[0]["run_dir"]).name
    if not re.fullmatch(r"attempt_\d+", attempt):
        raise ValueError("Unexpected generalization-audit attempt directory")
    return root / "runs" / organ / hierarchy / f"{source}_to_{target}" / attempt


def load_two_stage_pair(data_root: Path, run_dir: Path, organ: str, pair: str,
                        hierarchy: str, raw_root: Path | None = None
                        ) -> tuple[SpatialStage, SpatialStage, dict]:
    from .tables import _checked_matrix, _matrix_stats

    run_dir = Path(run_dir)
    source, target = parse_pair(pair)
    _, scale, k = HIERARCHIES[hierarchy]
    summary, config = _json(run_dir / "summary.json"), _json(run_dir / "config.json")
    feature = _json(run_dir / "feature_manifest.json")
    if summary.get("method") != TWO_STAGE or feature.get("method") != TWO_STAGE:
        raise ValueError("Saved run is not maturity_cci_grn_two_stage")
    if summary.get("K") != k or config.get("k") != k:
        raise ValueError("Saved K disagrees with the selected hierarchy")
    if not (run_dir / "best_joint.pt").is_file():
        raise FileNotFoundError(run_dir / "best_joint.pt")
    stages = []
    inputs = []
    counts = []
    for side, stage in (("t", source), ("tp", target)):
        expected = sample_stem(scale, organ, stage) + ".h5ad"
        recorded = feature.get("provenance", {}).get(f"h5ad_{side}", "")
        if PureWindowsPath(recorded).name != expected:
            raise ValueError(f"Wrong organ, time or input scale in saved h5ad_{side}: {recorded}")
        path = resolve_organ_paths(data_root, scale, organ, stage)["h5ad"]
        with h5py.File(path, "r") as handle:
            unit_ids = read_axis_index(handle, "obs", path)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError(f"Duplicate input unit IDs: {path}")
        assignment_path = run_dir / f"assignments_{side}.csv"
        labels = pd.read_csv(assignment_path, dtype={"spot_id": str})
        if not {"spot_id", "hard_cluster"} <= set(labels) or labels[["spot_id", "hard_cluster"]].isna().any().any():
            raise ValueError(f"Missing assignment IDs or labels: {assignment_path}")
        if labels["spot_id"].tolist() != unit_ids:
            raise ValueError("Assignment IDs/order differ from the selected organ/time H5AD")
        soft_path = run_dir / f"S_{side}.npy"
        soft = _checked_matrix(soft_path, (len(unit_ids), k))
        _matrix_stats(soft)  # Validate finite, nonnegative, row-stochastic assignments.
        hard = pd.to_numeric(labels["hard_cluster"], errors="raise").to_numpy(float)
        if not np.array_equal(hard, np.asarray(soft).argmax(axis=1)):
            raise ValueError("Display labels disagree with saved soft-assignment argmax")
        if scale == "spot":
            coords = read_h5ad_spatial(path)
            if coords is None or coords.shape != (len(unit_ids), 2):
                raise ValueError(f"Missing/misaligned input coordinates: {path}")
            spots = pd.DataFrame({"spot_id": unit_ids, "domain_id": hard.astype(int).astype(str),
                                  "x": coords[:, 0], "y": coords[:, 1]})
            nodes = spots.rename(columns={"spot_id": "unit_id"})
        else:
            low, map_path = _map(data_root, scale, organ, stage)
            if set(low["domain_id"]) != set(unit_ids):
                raise ValueError("Lower-domain map IDs differ from the trained input units")
            mapping = pd.Series(hard.astype(int).astype(str), index=unit_ids)
            nodes = low.groupby("domain_id", sort=False)[["x", "y"]].mean().loc[unit_ids]
            nodes = nodes.rename_axis("unit_id").assign(domain_id=mapping).reset_index()
            spots = low.assign(domain_id=low["domain_id"].map(mapping))
            inputs.append(str(map_path.resolve()))
        stages.append(SpatialStage(spots, nodes, {
            "method": TWO_STAGE, "illustrative": False,
            "mapping_rule": "Each lower unit -> centroid of its learned hard domain; hard labels are display-only",
            "coverage": _coverage(spots, raw_root, organ, stage)}))
        counts.append(len(unit_ids))
        inputs.extend(str(p.resolve()) for p in (path, assignment_path, soft_path))
    if k >= min(counts):
        raise ValueError("K must be smaller than both lower-state counts")
    micro = _checked_matrix(run_dir / "PIJ_micro_train.npy", tuple(counts))
    macro = _checked_matrix(run_dir / "PIJ_macro_train.npy", (k, k))
    ei_micro, _ = _matrix_stats(micro)
    ei_macro, _ = _matrix_stats(macro)
    delta = ei_macro - ei_micro
    for actual, key in ((ei_micro, "EI_micro_fixed"), (ei_macro, "EI_macro_best_checkpoint"),
                        (delta, "delta_EI_best_checkpoint")):
        value = summary.get(key)
        if value is None or not np.isfinite(float(value)) or not np.isclose(actual, float(value), atol=2e-4, rtol=0):
            raise ValueError(f"Persisted PIJ disagrees with summary {key}")
    if "trained_delta_EI" in summary and not np.isclose(delta, float(summary["trained_delta_EI"]), atol=2e-4, rtol=0):
        raise ValueError("Persisted PIJ disagrees with trained_delta_EI")
    metadata = {"run_dir": str(run_dir.resolve()), "method": TWO_STAGE,
                "delta_ei": delta, "epochs": config.get("epochs"), "best_epoch": summary.get("best_epoch"),
                "input_files": inputs + [str((run_dir / name).resolve()) for name in
                    ("summary.json", "config.json", "feature_manifest.json", "best_joint.pt",
                     "PIJ_micro_train.npy", "PIJ_macro_train.npy")],
                "metric_note": "EI uses saved soft assignments; displayed memberships use argmax"}
    return stages[0], stages[1], metadata


def load_seurat_delta(path: Path, organ: str, pair: str, hierarchy: str) -> float:
    frame = pd.read_csv(path)
    required = {"organ", "time_pair", "lower_layer", "upper_layer", "EI_gain"}
    if not required <= set(frame):
        raise ValueError(f"Seurat EI CSV requires {sorted(required)}")
    _, lower, k = HIERARCHIES[hierarchy]
    source, target = parse_pair(pair)
    normalized = frame["time_pair"].astype(str).str.replace(":", "->", regex=False)
    rows = frame.loc[frame["organ"].eq(organ) & normalized.eq(f"{source}->{target}")
                     & frame["lower_layer"].eq(lower) & frame["upper_layer"].eq(f"seurat_k{k}")]
    if len(rows) != 1 or not np.isfinite(float(rows.iloc[0]["EI_gain"])):
        raise ValueError("Select an EI CSV with exactly one measured row for this organ/time/hierarchy")
    return float(rows.iloc[0]["EI_gain"])


def _illustrative_groups(coords: np.ndarray, k: int, rotation: float) -> np.ndarray:
    """Deterministic median partitions for layout preview; no expression/fit used."""
    if not 1 <= k <= len(coords):
        raise ValueError("Preview K must not exceed the available point count")
    angle = np.deg2rad(rotation)
    xy = coords @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    groups = [np.arange(len(coords))]
    while len(groups) < k:
        candidates = [i for i, group in enumerate(groups) if len(group) > 1]
        index = max(candidates, key=lambda i: (len(groups[i]), float(np.ptp(xy[groups[i]], axis=0).max())))
        group = groups.pop(index)
        axis = int(np.ptp(xy[group], axis=0).argmax())
        ordered = group[np.argsort(xy[group, axis], kind="stable")]
        midpoint = len(ordered) // 2
        groups.extend((ordered[:midpoint], ordered[midpoint:]))
    labels = np.empty(len(coords), dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    return labels


def preview_stage(raw_root: Path, organ: str, stage: str, hierarchy: str, method: str) -> SpatialStage:
    if method not in {"seurat", "two_stage"}:
        raise ValueError(f"Unknown preview method: {method}")
    spots, raw = load_raw_spots(raw_root, organ, stage)
    _, lower, k = HIERARCHIES[hierarchy]
    coords = spots[["x", "y"]].to_numpy(float)
    if lower == "spot":
        lower_ids = spots["spot_id"].to_numpy()
    else:
        lower_ids = _illustrative_groups(coords, 150, 0).astype(str)
    lower_spots = spots.assign(unit_id=lower_ids)
    centers = lower_spots.groupby("unit_id", sort=False)[["x", "y"]].mean()
    groups = _illustrative_groups(centers.to_numpy(float), k, 0 if method == "seurat" else 31)
    mapping = pd.Series(groups.astype(str), index=centers.index)
    nodes = centers.assign(domain_id=mapping).reset_index()
    upper_spots = spots.assign(domain_id=lower_spots["unit_id"].map(mapping))
    return SpatialStage(upper_spots, nodes, {"method": method, "illustrative": True,
                        "mapping_rule": "Geometric median partitions for style only; neither method was fitted",
                        "coverage": {"full_local_organ": True, "server_identity_verified": False, "raw": raw}})
