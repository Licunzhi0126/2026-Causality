from __future__ import annotations

import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch

from .common import (
    apply_spatial_orientation,
    clean_2d_axis,
    draw_axis_brace,
    draw_figure_bracket,
    format_signed,
    normalized_xy,
    set_equal_spatial_limits,
    spatial_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1_domain_factory"
DEFAULT_SLICE_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "output" / "mignet_vertical_ablation_seurat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "paper_asset" / "ei_existence_figures_slat_style"

DEFAULT_NETWORK_METHOD = "legacy_mixed_grn_cci"
DEFAULT_PIJ_METHOD = "pseudotime_expression_ot"
DEFAULT_ORGAN = "heart"
DEFAULT_TIME_POINTS = ("11.5", "12.5", "13.5", "14.5")
DEFAULT_LEVEL_PAIRS = (
    "spot:seurat_k150",
    "seurat_k150:seurat_k40",
    "spot:seurat_k40",
)

LAYER_LABELS: Dict[str, str] = {
    "spot": "Spot",
    "seurat_k150": "K150",
    "seurat_k40": "K40",
    "seurat_less_than5": "Seurat <5",
    "louvain_k150": "Louvain K150",
    "louvain_k40": "Louvain K40",
    "louvain_less_than5": "Louvain <5",
    "organ": "Organ",
}

LAYER_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "spot": ("spot",),
    "seurat_less_than5": ("seuratLessThan5",),
    "seurat_k150": ("seurat150",),
    "seurat_k40": ("seurat", "seurat40"),
    "louvain_less_than5": ("louvainLessThan5",),
    "louvain_k150": ("louvain150",),
    "louvain_k40": ("louvain40",),
    "organ": ("organ",),
}

LOCAL_LAYER_ALIASES: Dict[str, str] = {
    "seurat_less_than5": "louvain_less_than5",
    "seurat_k150": "louvain_k150",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue_white_red",
    ["#2b6cb0", "#f7f7f7", "#b2182b"],
)

STACK_Z = {"spot": 0.0, "seurat_k150": 0.55, "seurat_k40": 1.10}


@dataclass(frozen=True)
class LevelPair:
    lower: str
    upper: str

    @classmethod
    def parse(cls, raw: str) -> "LevelPair":
        if ":" in raw:
            lower, upper = raw.split(":", 1)
        elif "->" in raw:
            lower, upper = raw.split("->", 1)
        else:
            raise ValueError(f"Cannot parse level pair {raw!r}; use lower:upper.")
        return cls(lower.strip(), upper.strip())

    @property
    def key(self) -> str:
        return f"{self.lower}->{self.upper}"

    @property
    def csv_key(self) -> str:
        return f"{self.lower}_to_{self.upper}"

    @property
    def label(self) -> str:
        return f"{layer_label(self.lower)} -> {layer_label(self.upper)}"


@dataclass(frozen=True)
class SpatialOrientation:
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False


@dataclass(frozen=True)
class FigurePaths:
    fig1: Path
    fig2: Path
    fig3: Path
    fig2_table: Path
    fig3_table: Path


def layer_label(layer: str) -> str:
    return LAYER_LABELS.get(layer, layer)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_level_pairs(values: Sequence[str]) -> List[LevelPair]:
    return [LevelPair.parse(value) for value in values]


def time_pair_order(time_points: Sequence[str]) -> List[str]:
    return [f"{left}->{right}" for left, right in itertools.combinations(time_points, 2)]


def adjacent_time_pairs(time_points: Sequence[str]) -> List[str]:
    return [f"{left}->{right}" for left, right in zip(time_points[:-1], time_points[1:])]


def resolve_metrics_path(result_root: Path, network_method: str, pij_method: str) -> Path:
    direct = result_root / "metrics.csv"
    if direct.exists():
        return direct
    nested = result_root / f"network={network_method}" / f"pij={pij_method}" / "metrics.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Could not find metrics.csv at either {direct} or {nested}.")


def load_metrics(
    result_root: Path,
    network_method: str,
    pij_method: str,
    organ: str,
) -> pd.DataFrame:
    metrics_path = resolve_metrics_path(result_root, network_method, pij_method)
    metrics = pd.read_csv(metrics_path)
    rename = {}
    if "EI_local" in metrics.columns and "EI_lower" not in metrics.columns:
        rename["EI_local"] = "EI_lower"
    if "EI_global" in metrics.columns and "EI_upper" not in metrics.columns:
        rename["EI_global"] = "EI_upper"
    metrics = metrics.rename(columns=rename)

    required = {"organ", "lower_layer", "upper_layer", "time_pair"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")

    if "EI_gain" not in metrics.columns:
        if {"EI_lower", "EI_upper"}.issubset(metrics.columns):
            metrics["EI_gain"] = pd.to_numeric(metrics["EI_upper"], errors="coerce") - pd.to_numeric(
                metrics["EI_lower"], errors="coerce"
            )
        else:
            raise ValueError(f"{metrics_path} must contain EI_gain or both EI_lower/EI_upper.")

    work = metrics.copy()
    for column in ("network_method", "pij_method", "organ", "lower_layer", "upper_layer", "time_pair"):
        if column in work.columns:
            work[column] = work[column].astype(str)

    if "network_method" in work.columns:
        work = work[work["network_method"] == network_method]
    if "pij_method" in work.columns:
        work = work[work["pij_method"] == pij_method]
    work = work[work["organ"] == organ].copy()
    work["EI_gain"] = pd.to_numeric(work["EI_gain"], errors="coerce")
    work["level_pair"] = work["lower_layer"] + "->" + work["upper_layer"]

    if work.empty:
        raise ValueError(
            "No metric rows remain after filtering by "
            f"organ={organ!r}, network_method={network_method!r}, pij_method={pij_method!r}."
        )
    return work


def metric_value(metrics: pd.DataFrame, time_pair: str, pair: LevelPair) -> float:
    match = metrics[
        (metrics["time_pair"] == time_pair)
        & (metrics["lower_layer"] == pair.lower)
        & (metrics["upper_layer"] == pair.upper)
    ]
    if match.empty:
        return float("nan")
    if len(match) > 1:
        warnings.warn(f"Multiple rows for {time_pair} {pair.key}; using mean EI_gain.", RuntimeWarning)
    return float(match["EI_gain"].mean())


def build_heatmap_table(
    metrics: pd.DataFrame,
    ordered_time_pairs: Sequence[str],
    level_pairs: Sequence[LevelPair],
) -> pd.DataFrame:
    rows = []
    for pair_time in ordered_time_pairs:
        row = {"time_pair": pair_time}
        for pair in level_pairs:
            row[pair.csv_key] = metric_value(metrics, pair_time, pair)
        rows.append(row)
    return pd.DataFrame(rows)


def build_time_pair_mean_table(
    metrics: pd.DataFrame,
    ordered_time_pairs: Sequence[str],
    level_pairs: Sequence[LevelPair],
) -> pd.DataFrame:
    rows = []
    for pair_time in ordered_time_pairs:
        values = [metric_value(metrics, pair_time, pair) for pair in level_pairs]
        finite = [value for value in values if np.isfinite(value)]
        rows.append(
            {
                "time_pair": pair_time,
                "mean_EI_gain": float(np.mean(finite)) if finite else float("nan"),
                "n_level_pairs": len(finite),
                "level_pairs": ";".join(pair.key for pair in level_pairs),
            }
        )
    return pd.DataFrame(rows)


def find_domain_map(data_root: Path, layer: str, organ: str, stage: str) -> Path | None:
    actual_layer = LOCAL_LAYER_ALIASES.get(layer, layer)
    layer_dir = data_root / actual_layer / organ
    if not layer_dir.exists():
        return None
    prefixes = LAYER_PREFIXES.get(actual_layer, (actual_layer,))
    for prefix in prefixes:
        candidate = layer_dir / f"{prefix}_{organ}_{stage}_spot_domain_map.csv"
        if candidate.exists():
            return candidate
    matches = sorted(layer_dir.glob(f"*_{organ}_{stage}_spot_domain_map.csv"))
    return matches[0] if matches else None


def load_domain_map(data_root: Path, layer: str, organ: str, stage: str) -> pd.DataFrame | None:
    path = find_domain_map(data_root, layer, organ, stage)
    if path is None:
        warnings.warn(f"Missing domain map for layer={layer}, organ={organ}, stage={stage}.", RuntimeWarning)
        return None
    domain_map = pd.read_csv(path)
    required = {"x", "y", "domain_id"}
    missing = required - set(domain_map.columns)
    if missing:
        warnings.warn(f"{path} is missing columns {sorted(missing)}.", RuntimeWarning)
        return None
    domain_map = domain_map.copy()
    domain_map["x"] = pd.to_numeric(domain_map["x"], errors="coerce")
    domain_map["y"] = pd.to_numeric(domain_map["y"], errors="coerce")
    domain_map["domain_id"] = domain_map["domain_id"].astype(str)
    domain_map = domain_map.dropna(subset=["x", "y"])
    if domain_map.empty:
        warnings.warn(f"{path} has no usable x/y rows.", RuntimeWarning)
        return None
    return domain_map


def decode_h5_values(values: Iterable[object]) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_full_slice(slice_root: Path, stage: str, organ: str) -> pd.DataFrame:
    path = slice_root / f"E{stage}_E1S1.MOSTA.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"Missing full slice h5ad: {path}")
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if "cell_name" not in obs:
            raise ValueError(f"{path} is missing obs/cell_name.")
        if "spatial" not in handle["obsm"]:
            raise ValueError(f"{path} is missing obsm/spatial.")
        spot_ids = decode_h5_values(obs["cell_name"][:])
        spatial = handle["obsm"]["spatial"][:]
        frame = pd.DataFrame(
            {
                "spot_id": spot_ids,
                "x": spatial[:, 0].astype(float),
                "y": spatial[:, 1].astype(float),
            }
        )
        if "annotation" in obs:
            annotation = obs["annotation"]
            categories_ref = annotation.attrs.get("categories")
            if categories_ref is not None:
                categories = decode_h5_values(handle[categories_ref][:])
                codes = annotation[:].astype(int)
                labels = [categories[code] if 0 <= code < len(categories) else "" for code in codes]
            else:
                labels = decode_h5_values(annotation[:])
            frame["annotation"] = labels
        else:
            frame["annotation"] = ""
    frame["is_target_organ"] = frame["annotation"].astype(str).str.lower() == organ.lower()
    return frame


def orient_frame(frame: pd.DataFrame, orientation: SpatialOrientation) -> pd.DataFrame:
    return apply_spatial_orientation(
        frame,
        swap_xy=orientation.swap_xy,
        invert_x=orientation.invert_x,
        invert_y=orientation.invert_y,
    )


def merge_slice_domain(slice_frame: pd.DataFrame, domain_map: pd.DataFrame | None) -> pd.DataFrame:
    if domain_map is None:
        return slice_frame.iloc[0:0].copy()
    if "spot_id" not in domain_map.columns:
        warnings.warn("Domain map is missing spot_id; cannot align with full slice.", RuntimeWarning)
        return slice_frame.iloc[0:0].copy()
    domain_part = domain_map.loc[:, [column for column in ("spot_id", "domain_id") if column in domain_map.columns]].copy()
    domain_part["spot_id"] = domain_part["spot_id"].astype(str)
    domain_part["domain_id"] = domain_part["domain_id"].astype(str)
    merged = slice_frame.merge(domain_part, on="spot_id", how="inner")
    if merged.empty:
        warnings.warn("Domain map did not match any full-slice spot_id values.", RuntimeWarning)
    return merged


def domain_color_lookup(domain_ids: Iterable[str], cmap_name: str = "tab20") -> Dict[str, Tuple[float, float, float, float]]:
    ordered = sorted(set(map(str, domain_ids)))
    cmap = plt.get_cmap(cmap_name)
    return {domain_id: cmap(idx % cmap.N) for idx, domain_id in enumerate(ordered)}


def plot_full_slice_base(
    ax: plt.Axes,
    slice_frame: pd.DataFrame,
    point_size: float = 0.55,
    alpha: float = 0.34,
    color: str = "#b7c0cb",
) -> None:
    ax.scatter(
        slice_frame["x"],
        slice_frame["y"],
        s=point_size,
        color=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    set_equal_spatial_limits(ax, slice_frame, pad_fraction=0.025)
    clean_2d_axis(ax)


def plot_heart_overlay(
    ax: plt.Axes,
    heart_frame: pd.DataFrame,
    color_lookup: Dict[str, Tuple[float, float, float, float]] | None = None,
    point_size: float = 2.1,
    alpha: float = 0.92,
) -> None:
    if heart_frame.empty:
        return
    if color_lookup is None or "domain_id" not in heart_frame.columns:
        colors = "#c53d3d"
    else:
        colors = heart_frame["domain_id"].astype(str).map(color_lookup).tolist()
    ax.scatter(
        heart_frame["x"],
        heart_frame["y"],
        c=colors,
        s=point_size,
        linewidths=0,
        alpha=alpha,
        rasterized=True,
    )


def domain_centroids(frame: pd.DataFrame, x_column: str = "x", y_column: str = "y") -> pd.DataFrame:
    if frame is None or frame.empty or "domain_id" not in frame.columns:
        return pd.DataFrame(columns=["domain_id", "x", "y"])
    work = frame.copy()
    work["domain_id"] = work["domain_id"].astype(str)
    return work.groupby("domain_id", as_index=False)[[x_column, y_column]].mean().rename(
        columns={x_column: "x", y_column: "y"}
    )


def compute_domain_parent_by_overlap(
    lower_domain_map: pd.DataFrame,
    upper_domain_map: pd.DataFrame,
) -> pd.DataFrame:
    required = {"spot_id", "domain_id"}
    if not required.issubset(lower_domain_map.columns) or not required.issubset(upper_domain_map.columns):
        raise ValueError("Both domain maps must contain spot_id and domain_id.")
    joined = lower_domain_map[["spot_id", "domain_id"]].merge(
        upper_domain_map[["spot_id", "domain_id"]],
        on="spot_id",
        how="inner",
        suffixes=("_lower", "_upper"),
    )
    if joined.empty:
        return pd.DataFrame(columns=["lower_domain_id", "upper_domain_id", "n_overlap"])
    counts = (
        joined.groupby(["domain_id_lower", "domain_id_upper"], as_index=False)
        .size()
        .rename(
            columns={
                "domain_id_lower": "lower_domain_id",
                "domain_id_upper": "upper_domain_id",
                "size": "n_overlap",
            }
        )
    )
    return (
        counts.sort_values(["lower_domain_id", "n_overlap", "upper_domain_id"], ascending=[True, False, True])
        .drop_duplicates("lower_domain_id")
        .reset_index(drop=True)
    )


def sample_membership_edges(
    edges: pd.DataFrame,
    *,
    group_col: str,
    max_per_group: int = 8,
    random_state: int = 7,
) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    if group_col not in edges.columns:
        raise ValueError(f"Missing group column: {group_col}")
    parts = []
    for _, group in edges.groupby(group_col, sort=True):
        if len(group) > max_per_group:
            parts.append(group.sample(n=max_per_group, random_state=random_state))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else edges.iloc[0:0].copy()


def _sample_frame(frame: pd.DataFrame, max_points: int, random_state: int = 7) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=random_state)


def draw_slat_like_layer_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    z_value: float,
    color="#c53d3d",
    color_col: str | None = None,
    color_lookup: Dict[str, Tuple[float, float, float, float]] | None = None,
    point_size: float = 1.7,
    alpha: float = 0.82,
    max_points: int | None = None,
) -> None:
    if frame.empty:
        return
    work = _sample_frame(frame, max_points) if max_points is not None else frame
    if color_col is not None and color_lookup is not None:
        colors = work[color_col].astype(str).map(color_lookup).tolist()
    else:
        colors = color
    ax.scatter(
        work["nx"],
        work["ny"],
        np.full(len(work), z_value),
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
    )


def draw_slat_like_wires(
    ax: plt.Axes,
    edges: pd.DataFrame,
    *,
    start_x: str = "start_x",
    start_y: str = "start_y",
    start_z: str = "start_z",
    end_x: str = "end_x",
    end_y: str = "end_y",
    end_z: str = "end_z",
    line_alpha: float = 0.08,
    line_width: float = 0.25,
    color: str = "#6b7280",
    dashed: bool = False,
) -> None:
    if edges.empty:
        return
    linestyle = "dashed" if dashed else "-"
    for row in edges.itertuples(index=False):
        ax.plot(
            [getattr(row, start_x), getattr(row, end_x)],
            [getattr(row, start_y), getattr(row, end_y)],
            [getattr(row, start_z), getattr(row, end_z)],
            color=color,
            linestyle=linestyle,
            linewidth=line_width,
            alpha=line_alpha,
        )


def _spot_to_domain_wire_edges(domain_frame: pd.DataFrame, *, max_per_group: int, random_state: int) -> pd.DataFrame:
    if domain_frame.empty:
        return pd.DataFrame()
    centroids = domain_centroids(domain_frame, "nx", "ny").set_index("domain_id")
    sampled = sample_membership_edges(
        domain_frame[["domain_id", "nx", "ny"]].copy(),
        group_col="domain_id",
        max_per_group=max_per_group,
        random_state=random_state,
    )
    rows = []
    for row in sampled.itertuples(index=False):
        centroid = centroids.loc[str(row.domain_id)]
        rows.append(
            {
                "start_x": row.nx,
                "start_y": row.ny,
                "start_z": STACK_Z["spot"],
                "end_x": float(centroid.x),
                "end_y": float(centroid.y),
                "end_z": STACK_Z["seurat_k150"],
                "domain_id": str(row.domain_id),
            }
        )
    return pd.DataFrame(rows)


def _domain_to_domain_wire_edges(
    lower_frame: pd.DataFrame,
    upper_frame: pd.DataFrame,
    *,
    max_per_group: int,
    random_state: int,
) -> pd.DataFrame:
    if lower_frame.empty or upper_frame.empty:
        return pd.DataFrame()
    parents = compute_domain_parent_by_overlap(lower_frame, upper_frame)
    if parents.empty:
        return pd.DataFrame()
    parents = sample_membership_edges(
        parents,
        group_col="upper_domain_id",
        max_per_group=max_per_group,
        random_state=random_state,
    )
    lower_centroids = domain_centroids(lower_frame, "nx", "ny").set_index("domain_id")
    upper_centroids = domain_centroids(upper_frame, "nx", "ny").set_index("domain_id")
    rows = []
    for row in parents.itertuples(index=False):
        lower_domain = str(row.lower_domain_id)
        upper_domain = str(row.upper_domain_id)
        if lower_domain not in lower_centroids.index or upper_domain not in upper_centroids.index:
            continue
        lower = lower_centroids.loc[lower_domain]
        upper = upper_centroids.loc[upper_domain]
        rows.append(
            {
                "start_x": float(lower.x),
                "start_y": float(lower.y),
                "start_z": STACK_Z["seurat_k150"],
                "end_x": float(upper.x),
                "end_y": float(upper.y),
                "end_z": STACK_Z["seurat_k40"],
                "upper_domain_id": upper_domain,
                "n_overlap": int(row.n_overlap),
            }
        )
    return pd.DataFrame(rows)


def _prepare_stage_stack_data(
    slice_root: Path,
    data_root: Path,
    organ: str,
    stage: str,
    orientation: SpatialOrientation,
) -> dict[str, pd.DataFrame]:
    slice_frame = orient_frame(load_full_slice(slice_root, stage, organ), orientation)
    bounds = spatial_bounds(slice_frame)
    slice_frame = normalized_xy(slice_frame, bounds)
    k150_map = load_domain_map(data_root, "seurat_k150", organ, stage)
    k40_map = load_domain_map(data_root, "seurat_k40", organ, stage)
    k150 = normalized_xy(merge_slice_domain(slice_frame, k150_map), bounds)
    k40 = normalized_xy(merge_slice_domain(slice_frame, k40_map), bounds)
    heart = slice_frame[slice_frame["is_target_organ"]].copy()
    return {"slice": slice_frame, "spot": heart, "seurat_k150": k150, "seurat_k40": k40}


def _configure_3d_stack_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.54, 0.54)
    ax.set_ylim(-0.54, 0.54)
    ax.set_zlim(-0.08, 1.22)
    ax.set_box_aspect((1.0, 1.0, 0.78))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()


def _plot_stage_stack(
    ax: plt.Axes,
    stage_data: dict[str, pd.DataFrame],
    *,
    stage: str,
    k150_colors: Dict[str, Tuple[float, float, float, float]],
    k40_colors: Dict[str, Tuple[float, float, float, float]],
    random_state: int,
    show_layer_labels: bool = True,
) -> None:
    full_slice = stage_data["slice"]
    background = _sample_frame(full_slice, 22000, random_state=random_state)
    for z_value in STACK_Z.values():
        ax.scatter(
            background["nx"],
            background["ny"],
            np.full(len(background), z_value),
            s=0.34,
            color="#aeb8c4",
            alpha=0.14,
            linewidths=0,
            depthshade=False,
        )

    spot_to_k150 = _spot_to_domain_wire_edges(stage_data["seurat_k150"], max_per_group=6, random_state=random_state)
    k150_to_k40 = _domain_to_domain_wire_edges(
        stage_data["seurat_k150"],
        stage_data["seurat_k40"],
        max_per_group=6,
        random_state=random_state,
    )
    draw_slat_like_wires(ax, spot_to_k150, line_alpha=0.045, line_width=0.18, color="#6b7280")
    draw_slat_like_wires(ax, k150_to_k40, line_alpha=0.13, line_width=0.34, color="#6b7280")

    draw_slat_like_layer_points(
        ax,
        stage_data["spot"],
        z_value=STACK_Z["spot"],
        color="#c53d3d",
        point_size=2.05,
        alpha=0.90,
        max_points=18000,
    )
    draw_slat_like_layer_points(
        ax,
        stage_data["seurat_k150"],
        z_value=STACK_Z["seurat_k150"],
        color_col="domain_id",
        color_lookup=k150_colors,
        point_size=1.85,
        alpha=0.78,
        max_points=18000,
    )
    draw_slat_like_layer_points(
        ax,
        stage_data["seurat_k40"],
        z_value=STACK_Z["seurat_k40"],
        color_col="domain_id",
        color_lookup=k40_colors,
        point_size=1.90,
        alpha=0.78,
        max_points=18000,
    )
    if show_layer_labels:
        for y, label in ((0.62, "K40"), (0.47, "K150"), (0.31, "Spot / Local")):
            ax.text2D(0.03, y, label, transform=ax.transAxes, fontsize=9.4, ha="left", va="center", color="#1f2937")
    ax.text2D(0.50, 0.96, f"E{stage}", transform=ax.transAxes, ha="center", va="top", fontsize=12, weight="semibold")
    _configure_3d_stack_axis(ax)


def draw_vertical_ei_brackets(ax: plt.Axes, values_by_pair: Dict[str, float]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_axis_brace(
        ax,
        0.13,
        0.58,
        0.82,
        f"EI gain = {format_signed(values_by_pair.get('seurat_k150->seurat_k40', float('nan')))}",
        fontsize=9.2,
    )
    draw_axis_brace(
        ax,
        0.13,
        0.30,
        0.54,
        f"EI gain = {format_signed(values_by_pair.get('spot->seurat_k150', float('nan')))}",
        fontsize=9.2,
    )
    draw_axis_brace(
        ax,
        0.62,
        0.22,
        0.84,
        f"EI gain = {format_signed(values_by_pair.get('spot->seurat_k40', float('nan')))}",
        fontsize=9.2,
    )


def plot_fig1_hierarchy(
    metrics: pd.DataFrame,
    slice_root: Path,
    data_root: Path,
    organ: str,
    time_points: Sequence[str],
    level_pairs: Sequence[LevelPair],
    output_path: Path,
    dpi: int,
    orientation: SpatialOrientation,
) -> None:
    if len(time_points) < 2:
        raise ValueError("Figure 1 requires at least two time points.")
    pair_time = f"{time_points[0]}->{time_points[1]}"
    value_by_pair = {pair.key: metric_value(metrics, pair_time, pair) for pair in level_pairs}
    stages = [time_points[0], time_points[1]]
    stage_data = {
        stage: _prepare_stage_stack_data(slice_root, data_root, organ, stage, orientation)
        for stage in stages
    }

    k150_ids: List[str] = []
    k40_ids: List[str] = []
    for stage in stages:
        k150_ids.extend(stage_data[stage]["seurat_k150"]["domain_id"].astype(str).tolist())
        k40_ids.extend(stage_data[stage]["seurat_k40"]["domain_id"].astype(str).tolist())
    k150_colors = domain_color_lookup(k150_ids or ["missing"], cmap_name="tab20")
    k40_colors = domain_color_lookup(k40_ids or ["missing"], cmap_name="tab20")

    fig = plt.figure(figsize=(14.8, 6.2), dpi=dpi)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.55, 0.13, 1.55, 0.90], wspace=-0.08)
    left_ax = fig.add_subplot(grid[0, 0], projection="3d")
    arrow_ax = fig.add_subplot(grid[0, 1])
    right_ax = fig.add_subplot(grid[0, 2], projection="3d")
    brace_ax = fig.add_subplot(grid[0, 3])

    _plot_stage_stack(
        left_ax,
        stage_data[stages[0]],
        stage=stages[0],
        k150_colors=k150_colors,
        k40_colors=k40_colors,
        random_state=11,
        show_layer_labels=True,
    )
    _plot_stage_stack(
        right_ax,
        stage_data[stages[1]],
        stage=stages[1],
        k150_colors=k150_colors,
        k40_colors=k40_colors,
        random_state=12,
        show_layer_labels=False,
    )

    arrow_ax.set_axis_off()
    for y in (0.27, 0.50, 0.73):
        arrow_ax.annotate(
            "",
            xy=(0.92, y),
            xytext=(0.08, y),
            arrowprops={"arrowstyle": "->", "lw": 1.35, "color": "#2d3742"},
            xycoords="axes fraction",
        )
        arrow_ax.text(0.50, y + 0.035, "Pij", ha="center", va="bottom", fontsize=10.0, weight="semibold")

    draw_vertical_ei_brackets(brace_ax, value_by_pair)
    fig.patches.append(
        FancyArrowPatch(
            (0.040, 0.23),
            (0.040, 0.79),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=12,
            lw=1.1,
            color="#30343b",
        )
    )
    fig.text(0.018, 0.51, "coarse-graining direction", rotation=90, ha="center", va="center", fontsize=9.8, color="#30343b")
    fig.suptitle("EI existence hierarchy", fontsize=14, weight="semibold", y=0.98)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.92, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_heatmap(
    heatmap_table: pd.DataFrame,
    level_pairs: Sequence[LevelPair],
    output_path: Path,
    dpi: int,
) -> None:
    values = heatmap_table[[pair.csv_key for pair in level_pairs]].to_numpy(dtype=float)
    labels_x = [pair.label for pair in level_pairs]
    labels_y = heatmap_table["time_pair"].astype(str).tolist()

    finite = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=dpi)
    image = ax.imshow(values, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Level pair")
    ax.set_ylabel("Time pair")
    ax.set_title("EI gain across developmental transitions", fontsize=13, weight="semibold", pad=12)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text = "NA" if not np.isfinite(value) else f"{value:.3f}"
            if np.isfinite(value):
                scaled = norm(value)
                text_color = "#ffffff" if scaled < 0.28 or scaled > 0.72 else "#1f2937"
            else:
                text_color = "#1f2937"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=9.2)

    ax.set_xticks(np.arange(-0.5, len(labels_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_y), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.035)
    cbar.set_label("EI gain")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _time_pair_lookup(table: pd.DataFrame) -> Dict[str, float]:
    return dict(zip(table["time_pair"].astype(str), pd.to_numeric(table["mean_EI_gain"], errors="coerce")))


def draw_time_bracket_lanes(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    time_points: Sequence[str],
    mean_by_time_pair: Dict[str, float],
) -> None:
    top = max(ax.get_position().y1 for ax in axes)
    lane_y = {1: top + 0.025, 2: top + 0.085, 3: top + 0.145}
    for left_idx in range(len(time_points)):
        for right_idx in range(left_idx + 1, len(time_points)):
            lag = right_idx - left_idx
            pair_time = f"{time_points[left_idx]}->{time_points[right_idx]}"
            left = axes[left_idx].get_position()
            right = axes[right_idx].get_position()
            x0 = left.x0 + 0.62 * (left.x1 - left.x0)
            x1 = right.x0 + 0.38 * (right.x1 - right.x0)
            value = mean_by_time_pair.get(pair_time, float("nan"))
            label = f"{pair_time}  mean EI = {format_signed(value)}"
            draw_figure_bracket(fig, x0, x1, lane_y[lag], label, fontsize=7.8 if lag == 1 else 8.2)


def plot_fig3_timeline(
    slice_root: Path,
    organ: str,
    time_points: Sequence[str],
    time_pair_means: pd.DataFrame,
    output_path: Path,
    dpi: int,
    orientation: SpatialOrientation,
) -> None:
    slice_frames = [orient_frame(load_full_slice(slice_root, stage, organ), orientation) for stage in time_points]

    fig, axes = plt.subplots(1, len(time_points), figsize=(12.4, 4.4), dpi=dpi)
    if len(time_points) == 1:
        axes = [axes]

    for ax, stage, frame in zip(axes, time_points, slice_frames):
        plot_full_slice_base(ax, frame, point_size=0.62, alpha=0.42)
        heart = frame[frame["is_target_organ"]].copy()
        plot_heart_overlay(ax, heart, color_lookup=None, point_size=2.2, alpha=0.94)
        ax.text(0.5, -0.075, f"E{stage}", ha="center", va="top", transform=ax.transAxes, fontsize=12, weight="semibold")

    fig.suptitle(f"{organ.title()} over developmental time", fontsize=13.5, weight="semibold", y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 0.76), w_pad=0.8)
    fig.canvas.draw()

    mean_by_time = _time_pair_lookup(time_pair_means)
    draw_time_bracket_lanes(fig, axes, time_points, mean_by_time)
    for idx in range(len(time_points) - 1):
        left = axes[idx].get_position()
        right = axes[idx + 1].get_position()
        y_mid = (left.y0 + left.y1) / 2.0
        fig.patches.append(
            FancyArrowPatch(
                (left.x1 + 0.004, y_mid),
                (right.x0 - 0.004, y_mid),
                transform=fig.transFigure,
                arrowstyle="->",
                mutation_scale=10,
                lw=1.0,
                color="#3f4752",
            )
        )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_tables(output_dir: Path, heatmap_table: pd.DataFrame, time_pair_means: pd.DataFrame) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    ensure_dir(table_dir)
    fig2_table = table_dir / "fig2_heatmap_values.csv"
    fig3_table = table_dir / "fig3_time_pair_mean_ei_gain.csv"
    heatmap_table.to_csv(fig2_table, index=False)
    time_pair_means.to_csv(fig3_table, index=False)
    return fig2_table, fig3_table


def generate_ei_existence_figures(
    *,
    data_root: Path,
    slice_root: Path,
    result_root: Path,
    output_dir: Path,
    organ: str = DEFAULT_ORGAN,
    network_method: str = DEFAULT_NETWORK_METHOD,
    pij_method: str = DEFAULT_PIJ_METHOD,
    time_points: Sequence[str] = DEFAULT_TIME_POINTS,
    level_pairs: Sequence[LevelPair] | None = None,
    orientation: SpatialOrientation = SpatialOrientation(),
    dpi: int = 300,
) -> FigurePaths:
    parsed_pairs = list(level_pairs) if level_pairs is not None else parse_level_pairs(DEFAULT_LEVEL_PAIRS)
    ordered_time_pairs = time_pair_order(time_points)

    ensure_dir(output_dir)
    metrics = load_metrics(result_root, network_method, pij_method, organ)
    heatmap_table = build_heatmap_table(metrics, ordered_time_pairs, parsed_pairs)
    time_pair_means = build_time_pair_mean_table(metrics, ordered_time_pairs, parsed_pairs)
    fig2_table, fig3_table = write_tables(output_dir, heatmap_table, time_pair_means)

    fig1_path = output_dir / "fig1_hierarchy_t0_t1_brackets.png"
    fig2_path = output_dir / "fig2_time_pair_level_pair_heatmap.png"
    fig3_path = output_dir / "fig3_four_timepoint_mean_ei_timeline.png"
    plot_fig2_heatmap(heatmap_table, parsed_pairs, fig2_path, dpi)
    plot_fig3_timeline(slice_root, organ, time_points, time_pair_means, fig3_path, dpi, orientation)
    plot_fig1_hierarchy(metrics, slice_root, data_root, organ, time_points, parsed_pairs, fig1_path, dpi, orientation)
    return FigurePaths(fig1=fig1_path, fig2=fig2_path, fig3=fig3_path, fig2_table=fig2_table, fig3_table=fig3_table)


def print_paths(paths: FigurePaths, stream=sys.stdout) -> None:
    for path in (paths.fig1, paths.fig2, paths.fig3, paths.fig2_table, paths.fig3_table):
        print(f"Wrote {path}", file=stream)
