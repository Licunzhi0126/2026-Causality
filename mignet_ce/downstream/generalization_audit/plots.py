from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .spatial_inputs import SpatialStage

# Keep these colors local to Generalization Audit so this plot can be used without
# importing the paper_assets plotting stack.
TEXT = "#23323D"
MUTED = "#73808A"
TEAL_DARK = "#1B3A55"
SPOT = "#CC4747"
CONNECTOR = "#7F8C96"

_REQUIRED_COLUMNS = {"spot_id", "domain_id", "x", "y"}
_LAYER_OFFSETS = {"spot": 0.0, "k150": 1.0, "k40": 2.0}
_Y_SCALE = 0.62


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "white",
        }
    )


def _validate_map(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    out = frame[["spot_id", "domain_id", "x", "y"]].copy()
    out["spot_id"] = out["spot_id"].astype(str)
    out["domain_id"] = out["domain_id"].astype(str)
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "spot_id", "domain_id"])
    if out["spot_id"].duplicated().any():
        examples = out.loc[out["spot_id"].duplicated(), "spot_id"].head(5).tolist()
        raise ValueError(f"{name} has duplicate spot_id values, e.g. {examples}")
    if out.empty:
        raise ValueError(f"{name} contains no usable spots")
    return out


def _prepare_stage(k150_map: pd.DataFrame, k40_map: pd.DataFrame, stage_name: str) -> dict[str, object]:
    k150 = _validate_map(k150_map, f"{stage_name} K150 map")
    k40 = _validate_map(k40_map, f"{stage_name} K40 map")

    joined = k150.merge(
        k40[["spot_id", "domain_id", "x", "y"]].rename(
            columns={"domain_id": "domain40", "x": "x40", "y": "y40"}
        ),
        on="spot_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined.loc[joined["_merge"] != "both", ["spot_id", "_merge"]]
    if not unmatched.empty:
        raise ValueError(
            f"{stage_name}: K150/K40 maps do not contain the same spot IDs "
            f"({len(unmatched)} unmatched rows; examples={unmatched.head(5).to_dict('records')})"
        )
    if not np.allclose(joined["x"].to_numpy(float), joined["x40"].to_numpy(float), rtol=0, atol=1e-5):
        raise ValueError(f"{stage_name}: x coordinates disagree between K150 and K40 maps")
    if not np.allclose(joined["y"].to_numpy(float), joined["y40"].to_numpy(float), rtol=0, atol=1e-5):
        raise ValueError(f"{stage_name}: y coordinates disagree between K150 and K40 maps")

    # One common isotropic transform preserves the spatial aspect ratio.
    x = joined["x"].to_numpy(float)
    y = joined["y"].to_numpy(float)
    x_mid = 0.5 * (float(x.min()) + float(x.max()))
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    joined["xn"] = (x - x_mid) / scale + 0.5
    joined["yn"] = 0.5 - (y - y_mid) / scale

    centers150 = (
        joined.groupby("domain_id", sort=True)[["xn", "yn"]]
        .mean()
        .rename_axis("domain150")
        .reset_index()
    )
    centers40 = (
        joined.groupby("domain40", sort=True)[["xn", "yn"]]
        .mean()
        .rename_axis("domain40")
        .reset_index()
    )

    # Map each K150 domain to the K40 domain receiving the largest number of
    # its spots. This is a data-derived dominant-overlap correspondence, not a
    # claim that the cluster labels are biologically identical.
    overlap = pd.crosstab(joined["domain_id"], joined["domain40"])
    dominant40 = overlap.idxmax(axis=1).rename("domain40").reset_index()
    dominant40 = dominant40.rename(columns={"domain_id": "domain150"})
    pairs = centers150.merge(dominant40, on="domain150", how="left", validate="one_to_one")
    pairs = pairs.merge(
        centers40.rename(columns={"xn": "x40n", "yn": "y40n"}),
        on="domain40",
        how="left",
        validate="many_to_one",
    )
    if pairs[["x40n", "y40n"]].isna().any().any():
        raise ValueError(f"{stage_name}: failed to map one or more K150 domains to K40 centers")

    # Domain colors are deterministic within each layer and stage. They do not
    # encode cross-time identity.
    labels150 = sorted(joined["domain_id"].unique().tolist())
    labels40 = sorted(joined["domain40"].unique().tolist())
    cmap150 = plt.get_cmap("turbo")
    cmap40 = plt.get_cmap("turbo")
    color150 = {label: cmap150(v) for label, v in zip(labels150, np.linspace(0.03, 0.97, len(labels150)))}
    color40 = {label: cmap40(v) for label, v in zip(labels40, np.linspace(0.03, 0.97, len(labels40)))}

    return {
        "spots": joined,
        "centers150": centers150,
        "centers40": centers40,
        "pairs": pairs,
        "color150": color150,
        "color40": color40,
        "n_spots": int(len(joined)),
        "n_k150": int(joined["domain_id"].nunique()),
        "n_k40": int(joined["domain40"].nunique()),
    }


def _layer_y(y_norm: np.ndarray | pd.Series, layer: str) -> np.ndarray:
    return np.asarray(y_norm, dtype=float) * _Y_SCALE + _LAYER_OFFSETS[layer]


def _draw_stage(ax: plt.Axes, stage_data: dict[str, object], stage: str) -> None:
    spots = stage_data["spots"]
    centers150 = stage_data["centers150"]
    centers40 = stage_data["centers40"]
    pairs = stage_data["pairs"]
    color150 = stage_data["color150"]
    color40 = stage_data["color40"]

    # Every native spot gets a line to the centroid of its assigned K150 domain.
    # No random subsampling: the count is exactly the number of spots.
    center150_lookup = centers150.set_index("domain150") if "domain150" in centers150.columns else centers150.set_index("domain_id")
    spot_x = spots["xn"].to_numpy(float)
    spot_y = spots["yn"].to_numpy(float)
    target150 = spots["domain_id"].astype(str).map(center150_lookup["xn"]).to_numpy(float)
    target150_y = spots["domain_id"].astype(str).map(center150_lookup["yn"]).to_numpy(float)
    spot_to_150 = np.stack(
        [
            np.column_stack([spot_x, _layer_y(spot_y, "spot")]),
            np.column_stack([target150, _layer_y(target150_y, "k150")]),
        ],
        axis=1,
    )
    ax.add_collection(
        LineCollection(
            spot_to_150,
            colors=CONNECTOR,
            linewidths=0.24,
            alpha=0.035,
            zorder=0,
            rasterized=True,
        )
    )

    # Each K150 domain centroid connects to the K40 centroid with which it has
    # the largest observed spot overlap.
    k150_to_40 = np.stack(
        [
            np.column_stack([pairs["xn"].to_numpy(float), _layer_y(pairs["yn"], "k150")]),
            np.column_stack([pairs["x40n"].to_numpy(float), _layer_y(pairs["y40n"], "k40")]),
        ],
        axis=1,
    )
    ax.add_collection(
        LineCollection(
            k150_to_40,
            colors=CONNECTOR,
            linewidths=0.45,
            alpha=0.17,
            zorder=0.5,
            rasterized=True,
        )
    )

    # Native-resolution Spot layer.
    ax.scatter(
        spots["xn"], _layer_y(spots["yn"], "spot"),
        s=2.2, color=SPOT, alpha=0.92, linewidths=0, rasterized=True, zorder=2,
    )

    # Show all spot coordinates colored by the assigned domain at each coarse scale.
    ax.scatter(
        spots["xn"], _layer_y(spots["yn"], "k150"),
        s=2.25, c=[color150[v] for v in spots["domain_id"].astype(str)],
        alpha=0.92, linewidths=0, rasterized=True, zorder=2,
    )
    ax.scatter(
        spots["xn"], _layer_y(spots["yn"], "k40"),
        s=2.25, c=[color40[v] for v in spots["domain40"].astype(str)],
        alpha=0.92, linewidths=0, rasterized=True, zorder=2,
    )

    # Centroids are overlaid as small white-edged markers so the connector target
    # is visible even where many spot points coincide visually.
    ax.scatter(
        centers150["xn"], _layer_y(centers150["yn"], "k150"),
        s=13, c=[color150[v] for v in centers150["domain150"].astype(str)],
        edgecolors="white", linewidths=0.55, marker="o", zorder=4,
    )
    ax.scatter(
        centers40["xn"], _layer_y(centers40["yn"], "k40"),
        s=15, c=[color40[v] for v in centers40["domain40"].astype(str)],
        edgecolors="white", linewidths=0.60, marker="o", zorder=4,
    )

    for label, layer in (("Spot", "spot"), ("K150", "k150"), ("K40", "k40")):
        y_center = _LAYER_OFFSETS[layer] + _Y_SCALE * 0.5
        ax.text(-0.055, y_center, label, ha="right", va="center", fontsize=8.7, color=TEXT)

    ax.set_xlim(-0.20, 1.04)
    ax.set_ylim(-0.12, 2.77)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"E{stage}", fontsize=12.0, fontweight="bold", color=TEAL_DARK, pad=6)


def _draw_pij_column(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 2.77)
    for layer in ("spot", "k150", "k40"):
        y = _LAYER_OFFSETS[layer] + _Y_SCALE * 0.5
        ax.annotate(
            "", xy=(0.90, y), xytext=(0.10, y),
            arrowprops={"arrowstyle": "->", "color": TEAL_DARK, "linewidth": 1.35},
        )
        ax.text(0.50, y + 0.11, "Pij", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=TEXT)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_square_bracket(ax: plt.Axes, x: float, y0: float, y1: float, *, cap: float, lw: float = 1.35) -> None:
    """Draw a clean square bracket opening toward the hierarchy (left)."""
    ax.plot([x, x], [y0, y1], color=TEXT, linewidth=lw, solid_capstyle="butt", clip_on=False)
    ax.plot([x - cap, x], [y0, y0], color=TEXT, linewidth=lw, solid_capstyle="butt", clip_on=False)
    ax.plot([x - cap, x], [y1, y1], color=TEXT, linewidth=lw, solid_capstyle="butt", clip_on=False)


def _draw_gain_brackets(ax: plt.Axes, gains: Mapping[str, float]) -> None:
    """Render two adjacent local brackets plus one independent total bracket."""
    y_spot = _LAYER_OFFSETS["spot"] + _Y_SCALE * 0.5
    y_k150 = _LAYER_OFFSETS["k150"] + _Y_SCALE * 0.5
    y_k40 = _LAYER_OFFSETS["k40"] + _Y_SCALE * 0.5

    # Small gap around the shared K150 boundary creates the double-tick rhythm
    # in the user's sketch without overlapping the two local bracket stems.
    gap = 0.045
    x_local = 0.48
    _draw_square_bracket(ax, x_local, y_k150 + gap, y_k40, cap=0.105, lw=1.35)
    _draw_square_bracket(ax, x_local, y_spot, y_k150 - gap, cap=0.105, lw=1.35)

    ax.text(
        x_local + 0.13, (y_k150 + gap + y_k40) / 2,
        f"{float(gains['K150 -> K40']):+.3f} bits",
        ha="left", va="center", fontsize=10.0, color=TEXT,
    )
    ax.text(
        x_local + 0.13, (y_spot + y_k150 - gap) / 2,
        f"{float(gains['Spot -> K150']):+.3f} bits",
        ha="left", va="center", fontsize=10.0, color=TEXT,
    )

    x_total = 1.73
    _draw_square_bracket(ax, x_total, y_spot, y_k40, cap=0.13, lw=1.55)
    ax.text(
        x_total + 0.16, (y_spot + y_k40) / 2,
        f"{float(gains['Spot -> K40']):+.3f} bits",
        ha="left", va="center", fontsize=10.2, color=TEXT,
    )

    ax.set_xlim(0.0, 2.95)
    ax.set_ylim(-0.12, 2.77)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Hierarchy ΔEI", fontsize=11.0, fontweight="bold", color=TEAL_DARK, pad=6)


def render_spatial_hierarchy_audit(
    *,
    source_stage: str,
    target_stage: str,
    source_k150_map: pd.DataFrame,
    source_k40_map: pd.DataFrame,
    target_k150_map: pd.DataFrame,
    target_k40_map: pd.DataFrame,
    gains: Mapping[str, float],
    output_path: str | Path,
    organ: str = "heart",
    style_preview: bool = False,
    dpi: int = 300,
    title: str = "Generalization Audit · Spatial-domain hierarchy",
    subtitle: str = "Every Spot → its K150-domain centroid  |  Every K150 centroid → dominant-overlap K40 centroid",
    footnote: str = "Domain colors are stage-local. Lines encode Spot→K150 membership and dominant-overlap K150→K40 mapping; they do not imply cross-time domain identity.",
) -> dict[str, object]:
    """Render the spatial hierarchy panel used by Generalization Audit.

    ``spot -> K150`` draws one connector per Spot to its assigned K150-domain
    centroid. ``K150 -> K40`` draws one connector per K150-domain centroid to
    its dominant-overlap K40-domain centroid. The function accepts real domain
    map tables for any organ; no organ-specific cluster labels are hard-coded.
    """
    required_gains = {"Spot -> K150", "K150 -> K40", "Spot -> K40"}
    missing_gains = required_gains.difference(gains)
    if missing_gains:
        raise ValueError(f"Missing ΔEI values: {sorted(missing_gains)}")
    for key in required_gains:
        if not np.isfinite(float(gains[key])):
            raise ValueError(f"ΔEI value for {key!r} must be finite")

    source = _prepare_stage(source_k150_map, source_k40_map, f"E{source_stage}")
    target = _prepare_stage(target_k150_map, target_k40_map, f"E{target_stage}")

    _set_style()
    fig = plt.figure(figsize=(17.8, 8.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.65, 0.70, 2.65, 3.10], wspace=0.24)
    ax_source = fig.add_subplot(gs[0, 0])
    ax_pij = fig.add_subplot(gs[0, 1])
    ax_target = fig.add_subplot(gs[0, 2])
    ax_gain = fig.add_subplot(gs[0, 3])

    _draw_stage(ax_source, source, source_stage)
    _draw_pij_column(ax_pij)
    _draw_stage(ax_target, target, target_stage)
    _draw_gain_brackets(ax_gain, gains)

    fig.suptitle(
        title,
        x=0.49, y=0.976, fontsize=15.0, fontweight="bold", color=TEAL_DARK,
    )
    fig.text(
        0.49, 0.942,
        subtitle,
        ha="center", va="center", fontsize=8.5, color=MUTED,
    )
    if style_preview:
        fig.text(
            0.985, 0.982, f"STYLE PREVIEW · {organ.upper()} MAPS ONLY",
            ha="right", va="top", fontsize=8.0, fontweight="bold", color="#A66723",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFF5E8", "edgecolor": "#E6C79F", "linewidth": 0.65},
        )
    fig.text(
        0.49, 0.025,
        footnote,
        ha="center", va="center", fontsize=7.4, color=MUTED,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.89, bottom=0.09)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "source": {"n_spots": source["n_spots"], "n_k150": source["n_k150"], "n_k40": source["n_k40"]},
        "target": {"n_spots": target["n_spots"], "n_k150": target["n_k150"], "n_k40": target["n_k40"]},
        "connections_per_stage": {
            "source": {
                "spot_to_k150": source["n_spots"],
                "k150_to_k40": source["n_k150"],
            },
            "target": {
                "spot_to_k150": target["n_spots"],
                "k150_to_k40": target["n_k150"],
            },
        },
        "mapping_rule": "K150 domain -> K40 domain with maximum observed spot overlap",
        "output_png": str(output_path),
        "output_pdf": str(output_path.with_suffix(".pdf")),
    }


def _spatial_pair_geometry(stage: SpatialStage) -> dict[str, object]:
    """Use all original spots for upper centroids and all lower units for wires."""
    spots, nodes = stage.spots.copy(), stage.nodes.copy()
    xy = spots[["x", "y"]].to_numpy(float)
    midpoint = (xy.min(axis=0) + xy.max(axis=0)) / 2
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-9)
    for frame in (spots, nodes):
        frame["xn"] = (frame["x"] - midpoint[0]) / scale + 0.5
        frame["yn"] = 0.5 - (frame["y"] - midpoint[1]) / scale
    centers = spots.groupby("domain_id", sort=True)[["xn", "yn"]].mean()
    target = centers.loc[nodes["domain_id"]].to_numpy(float)
    segments = np.stack((
        np.column_stack((nodes["xn"], nodes["yn"] * 0.62)),
        np.column_stack((target[:, 0], target[:, 1] * 0.62 + 1.05)),
    ), axis=1)
    cmap = plt.get_cmap("turbo")
    colors = {label: cmap(t) for label, t in zip(centers.index, np.linspace(0.03, 0.97, len(centers)))}
    return {"spots": spots, "nodes": nodes, "centers": centers, "segments": segments, "colors": colors}


def _draw_spatial_pair_stage(ax: plt.Axes, data: dict[str, object], stage: str,
                             lower_label: str, upper_label: str) -> dict[str, int]:
    spots, nodes, centers = data["spots"], data["nodes"], data["centers"]
    colors = data["colors"]
    # The upper outline is visual context. Only the centroid markers receive wires.
    ax.add_collection(LineCollection(data["segments"], colors=CONNECTOR,
                                     linewidths=0.38, alpha=0.14, rasterized=True, zorder=0))
    if lower_label != "Spot":
        ax.scatter(spots["xn"], spots["yn"] * 0.62, s=3, color="#DCE2E7",
                   linewidths=0, rasterized=True, zorder=1)
    ax.scatter(nodes["xn"], nodes["yn"] * 0.62, s=5 if lower_label == "Spot" else 16,
               color=SPOT, linewidths=0, rasterized=True, zorder=3)
    ax.scatter(spots["xn"], spots["yn"] * 0.62 + 1.05, s=4.5,
               c=[colors[v] for v in spots["domain_id"]], alpha=0.86,
               linewidths=0, rasterized=True, zorder=2)
    ax.scatter(centers["xn"], centers["yn"] * 0.62 + 1.05, s=20,
               c=[colors[v] for v in centers.index], edgecolors="white",
               linewidths=0.6, zorder=5)
    ax.text(0.0, 1.98, f"E{stage}", fontsize=13, fontweight="bold", color=TEAL_DARK)
    ax.text(1.0, 1.98, f"{len(spots):,} spots · {len(centers)} domains", ha="right", fontsize=8.2, color=MUTED)
    ax.text(0.0, 1.76, upper_label, fontsize=9.3, color=TEXT)
    ax.text(0.0, -0.18, f"{lower_label} · {len(nodes):,} nodes / {len(nodes):,} links",
            fontsize=9.0, color=TEXT)
    ax.set_xlim(-0.055, 1.055)
    ax.set_ylim(-0.27, 2.07)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return {"n_spots": len(spots), "n_lower_nodes": len(nodes), "n_upper_domains": len(centers),
            "n_connectors": len(data["segments"]),
            "n_distinct_connector_targets": len(np.unique(data["segments"][:, 1, :], axis=0))}


def render_spatial_pair(
    *, source: SpatialStage, target: SpatialStage,
    source_stage: str, target_stage: str, organ: str,
    method: str, lower_label: str, target_k: int, output_path: str | Path,
    delta_ei: float | None = None, dpi: int = 300, training_note: str = "",
) -> dict[str, object]:
    """Render one method in its own two-layer PNG/PDF; never combine methods.

    Lower nodes are either all Spots or all lower-domain centroids. Upper
    centroids always use original-spot geometry. Preview labels are explicit
    geometric placeholders; real-result figures require precomputed results.
    """
    if method not in {"seurat", "two_stage"}:
        raise ValueError(f"Unknown spatial method: {method}")
    if dpi < 1 or target_k < 1:
        raise ValueError("DPI and target K must be positive")
    preview = bool(source.audit.get("illustrative"))
    if preview != bool(target.audit.get("illustrative")):
        raise ValueError("Cannot mix illustrative and real stages in one figure")
    expected_method = "seurat" if method == "seurat" else "maturity_cci_grn_two_stage"
    for stage in (source, target):
        if stage.audit.get("method") not in ({method} if preview else {expected_method}):
            raise ValueError("Stage provenance disagrees with the figure method")
    if delta_ei is not None and (preview or not np.isfinite(delta_ei)):
        raise ValueError("Only a finite measured DeltaEI can appear on a real-result figure")
    _set_style()
    src, tgt = _spatial_pair_geometry(source), _spatial_pair_geometry(target)
    fig = plt.figure(figsize=(15.2, 7.6))
    grid = fig.add_gridspec(1, 4, width_ratios=[3.3, 0.62, 3.3, 1.35], wspace=0.20)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    upper_label = f"Seurat K{target_k}" if method == "seurat" else f"Learned Macro · target K={target_k}"
    if preview:
        upper_label += " · illustrative labels"
    counts = {
        "source": _draw_spatial_pair_stage(axes[0], src, source_stage, lower_label, upper_label),
        "target": _draw_spatial_pair_stage(axes[2], tgt, target_stage, lower_label, upper_label),
    }
    arrows = axes[1]
    arrows.set(xlim=(0, 1), ylim=(-0.27, 2.07))
    for y, label in ((0.31, "Pij lower"), (1.36, "Pij upper")):
        arrows.annotate("", xy=(0.94, y), xytext=(0.06, y),
                        arrowprops={"arrowstyle": "->", "color": TEAL_DARK, "linewidth": 1.4})
        arrows.text(0.5, y + 0.13, label, ha="center", fontsize=7.8, color=TEXT)
    arrows.axis("off")
    metric = axes[3]
    metric.set(xlim=(0, 1.7), ylim=(-0.27, 2.07))
    _draw_square_bracket(metric, 0.20, 0.31, 1.36, cap=0.12)
    metric.text(0.38, 1.04, "ΔEI", fontsize=12, fontweight="bold", color=TEAL_DARK)
    value = f"{delta_ei:+.3f} bits" if delta_ei is not None else "Not evaluated" if preview else "Not supplied"
    metric.text(0.38, 0.82, value, fontsize=10.6, color=TEXT)
    if preview:
        metric.text(0.38, 0.64, "Style preview", fontsize=8.2, color=MUTED)
    metric.axis("off")
    method_label = "Seurat natural clustering" if method == "seurat" else "Two-stage optimal coarse-graining"
    fig.suptitle(f"{organ.title()} · {method_label}", x=0.5, y=0.975,
                 fontsize=17, fontweight="bold", color=TEAL_DARK)
    fig.text(0.5, 0.925, "Every lower node has one connector; only upper-domain centroids receive connectors.",
             ha="center", fontsize=9.5, color=MUTED)
    if preview:
        status = f"STYLE PREVIEW · REAL {organ.upper()} COORDINATES · NO {'SEURAT' if method == 'seurat' else 'TWO-STAGE'} FIT"
    else:
        full = all(s.audit.get("coverage", {}).get("full_local_organ") is True for s in (source, target))
        status = "REAL RESULTS · " + ("FULL LOCAL ORGAN COVERAGE CHECKED" if full else "COVERAGE UNVERIFIED")
        if training_note:
            status += f" · {training_note}"
    fig.text(0.5, 0.878, status, ha="center", fontsize=9.1, fontweight="bold",
             color="#9A6227" if preview else TEAL_DARK,
             bbox={"facecolor": "#FFF5E8" if preview else "#EDF4F8", "edgecolor": "none", "pad": 6})
    footer = ("Grouping is illustrative; neither method was fitted and no EI was computed." if preview else
              "Domain colors are stage-local; displayed hard memberships do not replace saved soft assignments.")
    fig.text(0.5, 0.054, footer, ha="center", fontsize=8.2, color=MUTED)
    fig.text(0.5, 0.027, "White-ring markers: upper-domain centroids. Pij arrows are schematic time transitions, not domain tracking.",
             ha="center", fontsize=7.8, color=MUTED)
    fig.subplots_adjust(left=0.045, right=0.98, top=0.82, bottom=0.11)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return {"method": method, "illustrative": preview, "organ": organ,
            "source_stage": source_stage, "target_stage": target_stage,
            "target_k": target_k, "lower_label": lower_label, "counts": counts,
            "delta_ei": delta_ei, "status_label": status,
            "connector_rule": "Every lower node -> its assigned upper-domain centroid; no subsampling",
            "source": source.audit, "target": target.audit,
            "output_png": str(path.resolve()), "output_pdf": str(path.with_suffix('.pdf').resolve())}
