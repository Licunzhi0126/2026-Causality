#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1_domain_factory"
DEFAULT_SLICE_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "output" / "mignet_vertical_ablation_seurat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "paper_asset" / "ei_existence_figures"

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
    raise FileNotFoundError(
        "Could not find metrics.csv at either "
        f"{direct} or {nested}."
    )


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
        warnings.warn(
            f"Multiple rows for {time_pair} {pair.key}; using mean EI_gain.",
            RuntimeWarning,
        )
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


def build_adjacent_mean_table(
    metrics: pd.DataFrame,
    ordered_adjacent_pairs: Sequence[str],
    level_pairs: Sequence[LevelPair],
) -> pd.DataFrame:
    rows = []
    for pair_time in ordered_adjacent_pairs:
        selected = metrics[metrics["time_pair"] == pair_time].copy()
        selected_pairs = sorted(selected["level_pair"].dropna().astype(str).unique())
        rows.append(
            {
                "time_pair": pair_time,
                "mean_EI_gain": float(selected["EI_gain"].mean()) if not selected.empty else float("nan"),
                "n_level_pairs": int(selected["level_pair"].nunique()),
                "level_pairs": ";".join(selected_pairs),
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
        warnings.warn(
            f"Missing domain map for layer={layer}, organ={organ}, stage={stage}.",
            RuntimeWarning,
        )
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


def merge_slice_domain(slice_frame: pd.DataFrame, domain_map: pd.DataFrame | None) -> pd.DataFrame:
    if domain_map is None:
        return slice_frame.iloc[0:0].copy()
    required = ["spot_id", "domain_id", "x", "y"]
    if "spot_id" not in domain_map.columns:
        warnings.warn("Domain map is missing spot_id; cannot align with full slice.", RuntimeWarning)
        return slice_frame.iloc[0:0].copy()
    domain_part = domain_map.loc[:, [column for column in required if column in domain_map.columns]].copy()
    domain_part["spot_id"] = domain_part["spot_id"].astype(str)
    merged = slice_frame.merge(domain_part[["spot_id", "domain_id"]], on="spot_id", how="inner")
    if merged.empty:
        warnings.warn("Domain map did not match any full-slice spot_id values.", RuntimeWarning)
    return merged


def set_equal_spatial_limits(ax: plt.Axes, frame: pd.DataFrame, pad_fraction: float = 0.04) -> None:
    x_min, x_max = float(frame["x"].min()), float(frame["x"].max())
    y_min, y_max = float(frame["y"].min()), float(frame["y"].max())
    span = max(x_max - x_min, y_max - y_min, 1.0)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    pad = span * pad_fraction
    half = span / 2.0 + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_aspect("equal", adjustable="box")


def domain_color_lookup(domain_ids: Iterable[str], cmap_name: str = "tab20") -> Dict[str, Tuple[float, float, float, float]]:
    ordered = sorted(set(map(str, domain_ids)))
    cmap = plt.get_cmap(cmap_name)
    return {domain_id: cmap(idx % cmap.N) for idx, domain_id in enumerate(ordered)}


def plot_spatial_domain(
    ax: plt.Axes,
    frame: pd.DataFrame,
    color_lookup: Dict[str, Tuple[float, float, float, float]],
    point_size: float = 2.2,
    alpha: float = 0.92,
) -> None:
    colors = frame["domain_id"].map(color_lookup).tolist()
    ax.scatter(
        frame["x"],
        frame["y"],
        c=colors,
        s=point_size,
        linewidths=0,
        alpha=alpha,
    )
    set_equal_spatial_limits(ax, frame)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_full_slice_base(
    ax: plt.Axes,
    slice_frame: pd.DataFrame,
    point_size: float = 0.55,
    alpha: float = 0.34,
) -> None:
    ax.scatter(
        slice_frame["x"],
        slice_frame["y"],
        s=point_size,
        color="#cfd5dd",
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    set_equal_spatial_limits(ax, slice_frame, pad_fraction=0.025)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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


def draw_spot_to_domain_links(
    ax: plt.Axes,
    domain_frame: pd.DataFrame,
    max_links: int = 90,
    seed: int = 7,
) -> None:
    if domain_frame is None or domain_frame.empty:
        return
    centroids = domain_centroids(domain_frame).set_index("domain_id")
    work = domain_frame[domain_frame["domain_id"].astype(str).isin(centroids.index)].copy()
    if work.empty:
        return
    if len(work) > max_links:
        work = work.sample(n=max_links, random_state=seed)
    for row in work.itertuples(index=False):
        centroid = centroids.loc[str(row.domain_id)]
        ax.plot(
            [row.x, centroid.x],
            [row.y, centroid.y],
            color="#374151",
            alpha=0.10,
            lw=0.30,
            zorder=1,
        )
    ax.scatter(centroids["x"], centroids["y"], s=7, color="#1f2937", marker="x", linewidths=0.45, alpha=0.62, zorder=4)


def draw_domain_to_domain_links(
    ax: plt.Axes,
    lower_frame: pd.DataFrame,
    upper_frame: pd.DataFrame,
    max_links: int = 110,
) -> None:
    if lower_frame is None or upper_frame is None or lower_frame.empty or upper_frame.empty:
        return
    joined = lower_frame[["spot_id", "domain_id"]].merge(
        upper_frame[["spot_id", "domain_id"]],
        on="spot_id",
        how="inner",
        suffixes=("_lower", "_upper"),
    )
    if joined.empty:
        return
    lower_centroids = domain_centroids(lower_frame).set_index("domain_id")
    upper_centroids = domain_centroids(upper_frame).set_index("domain_id")
    strongest = (
        joined.groupby(["domain_id_lower", "domain_id_upper"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .drop_duplicates("domain_id_lower")
        .head(max_links)
    )
    for row in strongest.itertuples(index=False):
        lower_domain = str(row.domain_id_lower)
        upper_domain = str(row.domain_id_upper)
        if lower_domain not in lower_centroids.index or upper_domain not in upper_centroids.index:
            continue
        lower = lower_centroids.loc[lower_domain]
        upper = upper_centroids.loc[upper_domain]
        ax.plot(
            [lower.x, upper.x],
            [lower.y, upper.y],
            color="#111827",
            alpha=0.12,
            lw=0.42,
            zorder=1,
        )
    ax.scatter(lower_centroids["x"], lower_centroids["y"], s=6, color="#1f2937", marker="x", linewidths=0.42, alpha=0.60, zorder=4)
    ax.scatter(upper_centroids["x"], upper_centroids["y"], s=12, facecolors="none", edgecolors="#1f2937", linewidths=0.45, alpha=0.60, zorder=4)


def draw_figure_bracket(fig: plt.Figure, x0: float, x1: float, y: float, text: str) -> None:
    tick = 0.012
    line_kwargs = {"transform": fig.transFigure, "color": "#222222", "lw": 1.25, "clip_on": False}
    fig.lines.append(Line2D([x0, x1], [y, y], **line_kwargs))
    fig.lines.append(Line2D([x0, x0], [y, y - tick], **line_kwargs))
    fig.lines.append(Line2D([x1, x1], [y, y - tick], **line_kwargs))
    fig.text((x0 + x1) / 2.0, y + 0.008, text, ha="center", va="bottom", fontsize=9.2, color="#222222")


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


def plot_fig3_timeline(
    slice_root: Path,
    organ: str,
    time_points: Sequence[str],
    adjacent_means: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    slice_frames = [load_full_slice(slice_root, stage, organ) for stage in time_points]

    fig, axes = plt.subplots(1, len(time_points), figsize=(12.0, 3.9), dpi=dpi)
    if len(time_points) == 1:
        axes = [axes]

    for ax, stage, frame in zip(axes, time_points, slice_frames):
        plot_full_slice_base(ax, frame, point_size=0.52, alpha=0.30)
        heart = frame[frame["is_target_organ"]].copy()
        plot_heart_overlay(ax, heart, color_lookup=None, point_size=2.2, alpha=0.94)
        ax.text(
            0.5,
            -0.075,
            f"E{stage}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            weight="semibold",
        )

    fig.suptitle(
        f"{organ.title()} over developmental time",
        fontsize=13.5,
        weight="semibold",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88), w_pad=0.8)
    fig.canvas.draw()

    mean_by_time = dict(zip(adjacent_means["time_pair"], adjacent_means["mean_EI_gain"]))
    for idx, pair_time in enumerate(adjacent_time_pairs(time_points)):
        left = axes[idx].get_position()
        right = axes[idx + 1].get_position()
        x0 = left.x0 + 0.60 * (left.x1 - left.x0)
        x1 = right.x0 + 0.40 * (right.x1 - right.x0)
        y = max(left.y1, right.y1) + 0.04
        value = mean_by_time.get(pair_time, float("nan"))
        label = "mean EI gain = NA" if not np.isfinite(value) else f"mean EI gain = {value:+.3f}"
        draw_figure_bracket(fig, x0, x1, y, label)
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


def draw_schematic_panel(ax: plt.Axes, row: str, rng: np.random.Generator, seed_shift: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    box = FancyBboxPatch(
        (0.04, 0.06),
        0.92,
        0.82,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="#f7f8fa",
        edgecolor="#d5dae1",
        linewidth=0.9,
    )
    ax.add_patch(box)

    if row == "spot":
        x = rng.normal(0.5 + seed_shift, 0.19, 220)
        y = rng.normal(0.48, 0.17, 220)
        mask = (x > 0.08) & (x < 0.92) & (y > 0.10) & (y < 0.82)
        ax.scatter(x[mask], y[mask], s=5.0, color="#687783", alpha=0.72, linewidths=0)
    elif row == "k150":
        centers = np.array(
            [
                [0.23, 0.30],
                [0.38, 0.62],
                [0.55, 0.34],
                [0.68, 0.62],
                [0.78, 0.39],
            ]
        )
        colors = ["#4477aa", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
        for center, color in zip(centers, colors):
            cloud = rng.normal(center + [seed_shift, 0.0], [0.055, 0.05], size=(34, 2))
            ax.scatter(cloud[:, 0], cloud[:, 1], s=12, color=color, alpha=0.88, linewidths=0)
            ax.scatter(center[0] + seed_shift, center[1], s=32, color="#252525", marker="x", linewidths=1.1)
    else:
        centers = np.array([[0.34, 0.40], [0.61, 0.56], [0.73, 0.35]])
        colors = ["#2a9d8f", "#e76f51", "#457b9d"]
        for idx, (center, color) in enumerate(zip(centers, colors)):
            ellipse_x = rng.normal(center[0] + seed_shift, 0.075, 60)
            ellipse_y = rng.normal(center[1], 0.06, 60)
            ax.scatter(ellipse_x, ellipse_y, s=18, color=color, alpha=0.82, linewidths=0)
            ax.scatter(
                center[0] + seed_shift,
                center[1],
                s=120,
                facecolor="none",
                edgecolor="#222222",
                linewidths=1.0 + idx * 0.2,
            )


def draw_axis_brace(ax: plt.Axes, x: float, y0: float, y1: float, text: str, fontsize: float) -> None:
    ax.plot([x, x], [y0, y1], color="#242424", lw=1.2, clip_on=False)
    ax.plot([x - 0.015, x], [y0, y0], color="#242424", lw=1.2, clip_on=False)
    ax.plot([x - 0.015, x], [y1, y1], color="#242424", lw=1.2, clip_on=False)
    ax.text(x + 0.015, (y0 + y1) / 2.0, text, ha="left", va="center", fontsize=fontsize)


def plot_fig1_hierarchy(
    metrics: pd.DataFrame,
    slice_root: Path,
    data_root: Path,
    organ: str,
    time_points: Sequence[str],
    level_pairs: Sequence[LevelPair],
    output_path: Path,
    dpi: int,
) -> None:
    pair_time = f"{time_points[0]}->{time_points[1]}"
    value_by_pair = {pair.key: metric_value(metrics, pair_time, pair) for pair in level_pairs}

    stages = [time_points[0], time_points[1]]
    slice_by_stage = {stage: load_full_slice(slice_root, stage, organ) for stage in stages}
    maps_by_stage = {
        stage: {
            "seurat_k150": load_domain_map(data_root, "seurat_k150", organ, stage),
            "seurat_k40": load_domain_map(data_root, "seurat_k40", organ, stage),
        }
        for stage in stages
    }
    color_by_layer = {}
    for layer in ("seurat_k150", "seurat_k40"):
        ids: List[str] = []
        for stage in stages:
            frame = maps_by_stage[stage][layer]
            if frame is not None:
                ids.extend(frame["domain_id"].astype(str).tolist())
        color_by_layer[layer] = domain_color_lookup(ids or ["missing"], cmap_name="tab20")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    grid = fig.add_gridspec(3, 4, width_ratios=[1.16, 0.16, 1.16, 0.86], hspace=0.10, wspace=0.05)

    row_specs = [("seurat_k40", "K40"), ("seurat_k150", "K150"), ("spot", "Spot / Local")]
    axes_left = []
    axes_right = []
    for row_idx, (layer, row_label) in enumerate(row_specs):
        left = fig.add_subplot(grid[row_idx, 0])
        right = fig.add_subplot(grid[row_idx, 2])
        for ax, stage in ((left, stages[0]), (right, stages[1])):
            slice_frame = slice_by_stage[stage]
            plot_full_slice_base(ax, slice_frame, point_size=0.42, alpha=0.25)
            if layer == "spot":
                heart = slice_frame[slice_frame["is_target_organ"]].copy()
                plot_heart_overlay(ax, heart, color_lookup=None, point_size=1.75, alpha=0.92)
            else:
                current = merge_slice_domain(slice_frame, maps_by_stage[stage][layer])
                if layer == "seurat_k150":
                    draw_spot_to_domain_links(ax, current, max_links=85, seed=11 if stage == stages[0] else 12)
                elif layer == "seurat_k40":
                    lower = merge_slice_domain(slice_frame, maps_by_stage[stage]["seurat_k150"])
                    draw_domain_to_domain_links(ax, lower, current, max_links=95)
                plot_heart_overlay(
                    ax,
                    current,
                    color_lookup=color_by_layer[layer],
                    point_size=1.75 if layer == "seurat_k150" else 1.85,
                    alpha=0.90,
                )
        left.text(-0.06, 0.50, row_label, ha="right", va="center", transform=left.transAxes, fontsize=11)
        axes_left.append(left)
        axes_right.append(right)

    arrow_axes = [fig.add_subplot(grid[row_idx, 1]) for row_idx in range(3)]
    for ax in arrow_axes:
        ax.set_axis_off()
        ax.annotate(
            "",
            xy=(0.92, 0.5),
            xytext=(0.08, 0.5),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#2d3742"},
            xycoords="axes fraction",
        )
        ax.text(0.5, 0.60, "Pij", ha="center", va="bottom", fontsize=10.5, weight="semibold")

    brace_ax = fig.add_subplot(grid[:, 3])
    brace_ax.set_axis_off()
    brace_ax.set_xlim(0, 1)
    brace_ax.set_ylim(0, 1)
    pair_for_k150_k40 = "seurat_k150->seurat_k40"
    pair_for_spot_k150 = "spot->seurat_k150"
    pair_for_spot_k40 = "spot->seurat_k40"
    draw_axis_brace(
        brace_ax,
        0.18,
        0.69,
        0.89,
        f"EI gain = {value_by_pair.get(pair_for_k150_k40, float('nan')):+.3f}",
        fontsize=10.0,
    )
    draw_axis_brace(
        brace_ax,
        0.18,
        0.38,
        0.58,
        f"EI gain = {value_by_pair.get(pair_for_spot_k150, float('nan')):+.3f}",
        fontsize=10.0,
    )
    draw_axis_brace(
        brace_ax,
        0.46,
        0.13,
        0.89,
        f"EI gain = {value_by_pair.get(pair_for_spot_k40, float('nan')):+.3f}",
        fontsize=10.0,
    )

    axes_left[0].set_title(f"T0: E{time_points[0]}", fontsize=12, weight="semibold", pad=8)
    axes_right[0].set_title(f"T1: E{time_points[1]}", fontsize=12, weight="semibold", pad=8)

    fig.text(
        0.02,
        0.49,
        "coarse-graining direction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        color="#30343b",
    )
    fig.patches.append(
        FancyArrowPatch(
            (0.035, 0.16),
            (0.035, 0.84),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=12,
            lw=1.2,
            color="#30343b",
        )
    )
    fig.suptitle(
        "EI existence hierarchy",
        fontsize=14,
        weight="semibold",
        y=0.98,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def missing_fig1_real_inputs(data_root: Path, organ: str, time_points: Sequence[str]) -> List[str]:
    needed = [
        ("seurat_k150", time_points[0]),
        ("seurat_k150", time_points[1]),
        ("seurat_k40", time_points[0]),
        ("seurat_k40", time_points[1]),
    ]
    missing = []
    for layer, stage in needed:
        if find_domain_map(data_root, layer, organ, stage) is None:
            missing.append(f"{layer}:{stage}")
    return missing


def write_tables(output_dir: Path, heatmap_table: pd.DataFrame, adjacent_means: pd.DataFrame) -> None:
    table_dir = output_dir / "tables"
    ensure_dir(table_dir)
    heatmap_table.to_csv(table_dir / "fig2_heatmap_values.csv", index=False)
    adjacent_means.to_csv(table_dir / "fig3_adjacent_mean_ei_gain.csv", index=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot EI existence figures from local MIGNet vertical ablation data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--organ", default=DEFAULT_ORGAN)
    parser.add_argument("--network-method", default=DEFAULT_NETWORK_METHOD)
    parser.add_argument("--pij-method", default=DEFAULT_PIJ_METHOD)
    parser.add_argument("--cluster-method", default="seurat", help="Reserved for output metadata and compatibility.")
    parser.add_argument("--time-points", nargs="+", default=list(DEFAULT_TIME_POINTS))
    parser.add_argument("--level-pairs", nargs="+", default=list(DEFAULT_LEVEL_PAIRS))
    parser.add_argument("--timeline-layer", default="seurat_k40")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    level_pairs = parse_level_pairs(args.level_pairs)
    ordered_time_pairs = time_pair_order(args.time_points)
    ordered_adjacent_pairs = adjacent_time_pairs(args.time_points)

    ensure_dir(args.output_dir)
    metrics = load_metrics(args.result_root, args.network_method, args.pij_method, args.organ)
    heatmap_table = build_heatmap_table(metrics, ordered_time_pairs, level_pairs)
    adjacent_means = build_adjacent_mean_table(metrics, ordered_adjacent_pairs, level_pairs)
    write_tables(args.output_dir, heatmap_table, adjacent_means)

    fig2_path = args.output_dir / "fig2_time_pair_level_pair_heatmap.png"
    fig3_path = args.output_dir / "fig3_four_timepoint_mean_ei_timeline.png"
    plot_fig2_heatmap(heatmap_table, level_pairs, fig2_path, args.dpi)
    plot_fig3_timeline(
        args.slice_root,
        args.organ,
        args.time_points,
        adjacent_means,
        fig3_path,
        args.dpi,
    )

    fig1_path = args.output_dir / "fig1_hierarchy_t0_t1_brackets.png"
    plot_fig1_hierarchy(
        metrics,
        args.slice_root,
        args.data_root,
        args.organ,
        args.time_points,
        level_pairs,
        fig1_path,
        args.dpi,
    )

    print(f"Wrote {fig1_path}")
    print(f"Wrote {fig2_path}")
    print(f"Wrote {fig3_path}")
    print(f"Wrote {args.output_dir / 'tables' / 'fig2_heatmap_values.csv'}")
    print(f"Wrote {args.output_dir / 'tables' / 'fig3_adjacent_mean_ei_gain.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
