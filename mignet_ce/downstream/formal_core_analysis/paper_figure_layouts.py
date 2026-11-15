from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.collections import LineCollection

from .legacy_figure_style import HEART, MUTED, SPOT, TEAL, TEAL_DARK, TEXT, set_asset_style


def _norm_xy(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize spatial coordinates without distorting the tissue aspect ratio.

    Spatial y coordinates in the source maps increase downward, so y is inverted for
    publication plotting.  A single isotropic scale is used for both axes so the
    real heart geometry is preserved rather than stretched into a square.
    """

    out = frame.copy()
    x = pd.to_numeric(out["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(out["y"], errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    out = out.loc[finite].copy()
    x = x[finite]
    y = y[finite]
    x_mid = 0.5 * (float(np.min(x)) + float(np.max(x)))
    y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    out["xn"] = (x - x_mid) / scale + 0.5
    out["yn"] = 0.5 - (y - y_mid) / scale
    return out


def _domain_colors(values: pd.Series, cmap_name: str) -> list[object]:
    labels = values.astype(str)
    unique = sorted(labels.unique())
    cmap = plt.get_cmap(cmap_name)
    lookup = {label: cmap(i % cmap.N) for i, label in enumerate(unique)}
    return [lookup[label] for label in labels]


def _clean(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def centroid_connections(spot: pd.DataFrame, membership: pd.DataFrame, label: str,
                         lower: float, upper: float, yscale: float = .34):
    """One segment per real spot; endpoints use spatial means of its target group.

    Join by spot identity, never row order or a presumed nesting of domain labels.
    All geometry uses the same isotropic coordinate transform before layer offsets.
    """
    for frame, columns in ((spot, {"spot_id", "x", "y"}), (membership, {"spot_id", label})):
        if not columns <= set(frame) or frame.empty or frame.spot_id.isna().any() or frame.spot_id.duplicated().any():
            raise ValueError("Spatial membership requires unique, nonempty spot identities")
    if not np.isfinite(spot[["x", "y"]].apply(pd.to_numeric).to_numpy(float)).all():
        raise ValueError("Non-finite spatial coordinates")
    if set(spot.spot_id) != set(membership.spot_id) or membership[label].isna().any():
        raise ValueError("Missing or extra spatial membership")
    members = _norm_xy(spot)[["spot_id", "xn", "yn"]].merge(
        membership[["spot_id", label]], on="spot_id", validate="one_to_one")
    centers = members.groupby(label, sort=True)[["xn", "yn"]].mean()
    target = centers.loc[members[label]].to_numpy(float)
    source = members[["xn", "yn"]].to_numpy(float)
    source[:, 1] = source[:, 1] * yscale + lower
    target[:, 1] = target[:, 1] * yscale + upper
    return members, centers, np.stack((source, target), axis=1)


def _draw_centroid_mapping(ax, spot, membership, label, lower, upper, *, yscale=.34, cmap="tab20"):
    members, centers, segments = centroid_connections(spot, membership, label, lower, upper, yscale)
    colors = _domain_colors(members[label], cmap)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=.30, alpha=.12,
                                     rasterized=True, zorder=0))
    ax.scatter(members.xn, members.yn * yscale + upper, s=2.2, c=colors,
               alpha=.25, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(centers.xn, centers.yn * yscale + upper,
               s=17 if len(centers) <= 40 else 10,
               c=_domain_colors(pd.Series(centers.index), cmap), edgecolors="white",
               linewidths=.35, alpha=1, zorder=3)
    return len(centers)


def _metric_value(value) -> str:
    return f"{float(value):+.3f}" if np.isfinite(float(value)) else "—"


def _schematic_embryo(ax: plt.Axes, heart: pd.DataFrame | None = None, rng_seed: int = 7) -> None:
    """Draw only the unavailable whole-slice context schematically.

    When real heart spots are supplied, the red ROI itself uses their actual spatial
    geometry.  Thus only the surrounding embryo silhouette is schematic.
    """

    rng = np.random.default_rng(rng_seed)
    n = 6500
    u = rng.normal(size=(n, 2))
    u[:, 0] *= 1.65
    u[:, 1] *= 0.63
    u[:, 0] += 0.12 * np.sin(2.2 * u[:, 1])
    keep = (u[:, 0] ** 2 / 5.1 + u[:, 1] ** 2 / 0.7) < 1.0
    u = u[keep]
    ax.scatter(u[:, 0], u[:, 1], s=0.55, color="#CBD2D8", alpha=0.40, linewidths=0, rasterized=True)

    if heart is not None and not heart.empty:
        h = _norm_xy(heart)
        hx = (h["xn"].to_numpy(float) - 0.5) * 0.70 + 0.12
        hy = (h["yn"].to_numpy(float) - 0.5) * 0.52 - 0.03
        ax.scatter(hx, hy, s=1.25, color=HEART, alpha=0.88, linewidths=0, rasterized=True, zorder=3)
        xmin, xmax = float(np.min(hx)), float(np.max(hx))
        ymin, ymax = float(np.min(hy)), float(np.max(hy))
        padx = 0.10 * max(xmax - xmin, 0.08)
        pady = 0.10 * max(ymax - ymin, 0.08)
        ax.add_patch(Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2 * padx, (ymax - ymin) + 2 * pady, fill=False, edgecolor=HEART, linewidth=1.4))
        ax.text(0.12, ymax + pady + 0.06, "heart ROI", ha="center", va="bottom", fontsize=7.3, color=HEART, fontweight="bold")
    else:
        ax.add_patch(Rectangle((-0.10, -0.19), 0.42, 0.34, fill=False, edgecolor=HEART, linewidth=1.5))
        ax.text(0.11, 0.20, "heart ROI", ha="center", va="bottom", fontsize=7.5, color=HEART, fontweight="bold")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 1.2)
    _clean(ax)


def draw_context(ax: plt.Axes, full_slice: pd.DataFrame | None, heart: pd.DataFrame, title: str, organ: str = "heart") -> None:
    if full_slice is None or full_slice.empty:
        base = _norm_xy(heart)
        ax.scatter(base["xn"], base["yn"], s=1.2, color=HEART, alpha=0.85, linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        _clean(ax)
        ax.text(0.5, -0.06, f"measured {organ} region", transform=ax.transAxes, ha="center", fontsize=7.0, color=MUTED)
    else:
        ax.scatter(full_slice["x"], full_slice["y"], s=0.45, color="#C8D0D6", alpha=0.42, linewidths=0, rasterized=True)
        h = heart
        ax.scatter(h["x"], h["y"], s=1.2, color=HEART, alpha=0.9, linewidths=0, rasterized=True)
        xmin, xmax = h["x"].min(), h["x"].max()
        ymin, ymax = h["y"].min(), h["y"].max()
        padx = 0.12 * max(float(xmax - xmin), 1.0)
        pady = 0.12 * max(float(ymax - ymin), 1.0)
        ax.add_patch(Rectangle((xmin - padx, ymin - pady), (xmax - xmin) + 2 * padx, (ymax - ymin) + 2 * pady, fill=False, edgecolor=HEART, linewidth=1.4))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        _clean(ax)
    ax.set_title(title, fontsize=9.2, color=TEAL_DARK, pad=5)


def draw_heart_zoom(ax: plt.Axes, heart: pd.DataFrame, domain: pd.DataFrame | None = None, title: str = "heart zoom") -> None:
    base = _norm_xy(heart)
    colors: object = HEART
    if domain is not None and "domain_id" in domain.columns:
        merged = base[["spot_id", "xn", "yn"]].merge(domain[["spot_id", "domain_id"]], on="spot_id", how="left")
        colors = _domain_colors(merged["domain_id"].fillna("NA"), "tab20")
        ax.scatter(merged["xn"], merged["yn"], s=3.1, c=colors, linewidths=0, alpha=0.9, rasterized=True)
    else:
        ax.scatter(base["xn"], base["yn"], s=3.1, color=HEART, linewidths=0, alpha=0.9, rasterized=True)
    ax.set_aspect("equal")
    _clean(ax)
    ax.set_title(title, fontsize=9.2, color=TEAL_DARK, pad=5)


def _draw_zoom_arrow(fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes) -> None:
    b1 = ax_from.get_position()
    b2 = ax_to.get_position()
    arrow = FancyArrowPatch(
        (b1.x1 + 0.002, (b1.y0 + b1.y1) / 2),
        (b2.x0 - 0.002, (b2.y0 + b2.y1) / 2),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=TEAL,
    )
    fig.add_artist(arrow)


def _stack_stage(
    ax: plt.Axes,
    spot: pd.DataFrame,
    k150: pd.DataFrame,
    k40: pd.DataFrame,
    *,
    stage: str,
    max_connectors: int | None = None,
) -> None:
    coords = _norm_xy(spot)[["spot_id", "xn", "yn"]]
    yscale = 0.34
    offsets = {"spot": 0.0, "k150": 1.0, "k40": 2.0}

    ax.scatter(coords["xn"], coords["yn"] * yscale, s=2.3, color=SPOT,
               alpha=.90, linewidths=0, rasterized=True, zorder=2)
    _draw_centroid_mapping(ax, spot, k150, "domain_id", 0., 1., cmap="tab20")
    _draw_centroid_mapping(ax, spot, k40, "domain_id", 1., 2., cmap="tab20b")
    ax.text(-0.08, offsets["spot"] + 0.16, "Spot", ha="right", va="center", fontsize=8.4, color=TEXT)
    ax.text(-0.08, offsets["k150"] + 0.16, "K150", ha="right", va="center", fontsize=8.4, color=TEXT)
    ax.text(-0.08, offsets["k40"] + 0.16, "K40", ha="right", va="center", fontsize=8.4, color=TEXT)
    ax.set_xlim(-0.14, 1.04)
    ax.set_ylim(-0.12, 2.46)
    _clean(ax)
    ax.set_title(f"E{stage}", fontsize=12.0, fontweight="bold", color=TEAL_DARK, pad=4)


def _draw_pij_arrows(ax: plt.Axes, labels: tuple[str, ...] = ("Pij", "Pij", "Pij")) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 2.46)
    ys = (0.17, 1.17, 2.17)
    for y, label in zip(ys, labels):
        ax.annotate("", xy=(0.89, y), xytext=(0.11, y), arrowprops=dict(arrowstyle="->", color=TEAL_DARK, linewidth=1.35))
        ax.text(0.5, y + 0.14, label, ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=TEXT)
    _clean(ax)


def _simple_bracket(ax: plt.Axes, y0: float, y1: float, text: str, x: float, side: str = "left") -> None:
    tick = 0.06
    ax.plot([x, x], [y0, y1], color=TEXT, linewidth=1.0)
    direction = 1 if side == "left" else -1
    ax.plot([x, x + direction * tick], [y0, y0], color=TEXT, linewidth=1.0)
    ax.plot([x, x + direction * tick], [y1, y1], color=TEXT, linewidth=1.0)
    ax.text(x + direction * (tick + 0.05), (y0 + y1) / 2, text, ha="left" if direction > 0 else "right", va="center", fontsize=7.8, color=TEXT)


def _inward_brace(
    ax: plt.Axes,
    y0: float,
    y1: float,
    *,
    x: float,
    text: str,
    depth: float = 0.18,
    gap_fraction: float = 0.065,
    text_offset: float = 0.18,
    fontsize: float = 7.6,
) -> None:
    """Continuous inward-opening brace with smooth caps and a rounded waist."""
    mid = (y0 + y1) / 2
    shoulder = min((y1 - y0) * .17, .20)
    waist = min((y1 - y0) * .12, .14)
    vertices = [
        (x-depth, y0),
        (x, y0), (x, y0), (x, y0+shoulder),
        (x, mid-waist), (x-depth, mid-waist), (x-depth, mid),
        (x-depth, mid+waist), (x, mid+waist), (x, y1-shoulder),
        (x, y1), (x, y1), (x-depth, y1),
    ]
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * 12)
    ax.add_patch(PathPatch(path, fill=False, edgecolor=TEXT, linewidth=1.15,
                           capstyle="round", joinstyle="round"))
    ax.text(
        x + text_offset,
        mid,
        text,
        ha="left",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        linespacing=1.20,
    )

def render_figure1(
    *,
    source_stage: str,
    target_stage: str,
    source_spot: pd.DataFrame,
    target_spot: pd.DataFrame,
    source_k150: pd.DataFrame,
    target_k150: pd.DataFrame,
    source_k40: pd.DataFrame,
    target_k40: pd.DataFrame,
    gains: Mapping[str, float],
    output_path: Path,
    full_slice_source: pd.DataFrame | None = None,
    demo: bool = False,
    dpi: int = 300,
    preview_notice: str | None = None,
) -> None:
    set_asset_style()
    fig = plt.figure(figsize=(18.6, 7.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.30, 2.35, 0.58, 2.35, 3.05], wspace=0.18)
    ax_context = fig.add_subplot(gs[0, 0])
    ax_source = fig.add_subplot(gs[0, 1])
    ax_arrow = fig.add_subplot(gs[0, 2])
    ax_target = fig.add_subplot(gs[0, 3])
    ax_gain = fig.add_subplot(gs[0, 4])

    draw_context(ax_context, full_slice_source, source_spot, f"E{source_stage} sample")
    _stack_stage(ax_source, source_spot, source_k150, source_k40, stage=source_stage)
    _draw_pij_arrows(ax_arrow)
    _stack_stage(ax_target, target_spot, target_k150, target_k40, stage=target_stage)

    ax_gain.set_xlim(0, 5.20)
    ax_gain.set_ylim(-0.12, 2.46)
    _clean(ax_gain)

    # Inner braces share the same x-position and both open left, toward the hierarchy.
    # A small gap around K150 prevents their stems from visually colliding.
    g_s150 = gains.get("Spot -> K150", np.nan)
    g_15040 = gains.get("K150 -> K40", np.nan)
    g_s40 = gains.get("Spot -> K40", np.nan)
    _inward_brace(
        ax_gain, .195, 1.145, x=0.48,
        text=(
            r"$\Delta EI$ (Spot $\rightarrow$ K150)" "\n"
            r"$= EI(K150)-EI(Spot)$" "\n"
            f"= {_metric_value(g_s150)}"
        ),
        depth=0.18, text_offset=0.20, fontsize=7.3,
    )
    _inward_brace(
        ax_gain, 1.195, 2.145, x=0.48,
        text=(
            r"$\Delta EI$ (K150 $\rightarrow$ K40)" "\n"
            r"$= EI(K40)-EI(K150)$" "\n"
            f"= {_metric_value(g_15040)}"
        ),
        depth=0.18, text_offset=0.20, fontsize=7.3,
    )

    # Outer brace uses exactly the same inward orientation and spans the total hierarchy.
    _inward_brace(
        ax_gain, .17, 2.17, x=3.34,
        text=(
            r"Total $\Delta EI$ (Spot $\rightarrow$ K40)" "\n"
            r"$= EI(K40)-EI(Spot)$" "\n"
            f"= {_metric_value(g_s40)}"
        ),
        depth=0.24, text_offset=0.22, fontsize=7.5,
    )

    fig.suptitle("Figure 1 · EI existence across the coarse-graining hierarchy", x=0.52, y=0.985, fontsize=15, fontweight="bold", color=TEAL_DARK)
    fig.text(0.02, 0.50, "coarse-graining direction", rotation=90, va="center", ha="center", fontsize=8.5, color=MUTED)
    if preview_notice:
        fig.text(.5, .015, preview_notice, ha="center", va="bottom", fontsize=8, color="#8A5529")
    elif demo:
        fig.text(0.99, 0.99, "EXPLORATORY REAL DATA · NOT FINAL PAPER RESULTS", ha="right", va="top", fontsize=8.5, color="#A66723")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stack_stage_k10(
    ax: plt.Axes, spot: pd.DataFrame, k150: pd.DataFrame,
    k40: pd.DataFrame, k10: pd.DataFrame, *, stage: str,
) -> None:
    """Extend the established three-layer panel without changing its styling."""

    _stack_stage(ax, spot, k150, k40, stage=stage)
    _draw_centroid_mapping(ax, spot, k10, "domain_id", 2., 3., cmap="tab10")
    ax.text(-0.08, 3.16, "K10", ha="right", va="center", fontsize=8.4, color=TEXT)
    ax.set_ylim(-0.12, 3.46)


def render_figure1_k10(
    *, source_stage: str, target_stage: str,
    source_spot: pd.DataFrame, target_spot: pd.DataFrame,
    source_k150: pd.DataFrame, target_k150: pd.DataFrame,
    source_k40: pd.DataFrame, target_k40: pd.DataFrame,
    source_k10: pd.DataFrame, target_k10: pd.DataFrame,
    gains: Mapping[str, float], output_path: Path,
    full_slice_source: pd.DataFrame | None = None, demo: bool = False, dpi: int = 300,
    preview_notice: str | None = None,
) -> None:
    set_asset_style()
    fig = plt.figure(figsize=(23.8, 9.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.30, 2.35, 0.58, 2.35, 5.55], wspace=0.18)
    ax_context, ax_source, ax_arrow, ax_target, ax_gain = (
        fig.add_subplot(gs[0, index]) for index in range(5)
    )
    draw_context(ax_context, full_slice_source, source_spot, f"E{source_stage} sample")
    _stack_stage_k10(ax_source, source_spot, source_k150, source_k40, source_k10, stage=source_stage)
    _stack_stage_k10(ax_target, target_spot, target_k150, target_k40, target_k10, stage=target_stage)
    _draw_pij_arrows(ax_arrow)
    ax_arrow.annotate("", xy=(0.89, 3.17), xytext=(0.11, 3.17), arrowprops=dict(arrowstyle="->", color=TEAL_DARK, linewidth=1.35))
    ax_arrow.text(0.5, 3.31, "Pij", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=TEXT)
    ax_arrow.set_ylim(-0.12, 3.46)
    ax_gain.set_xlim(0, 11.2)
    ax_gain.set_ylim(-0.12, 3.46)
    _clean(ax_gain)
    braces = (
        ("Spot -> K150", .195, 1.145, 0.45),
        ("K150 -> K40", 1.195, 2.145, 0.45),
        ("K40 -> K10", 2.195, 3.145, 0.45),
        ("Spot -> K40", .17, 2.17, 3.05),
        ("K150 -> K10", 1.17, 3.17, 5.75),
        ("Spot -> K10", .17, 3.17, 8.45),
    )
    for label, y0, y1, x in braces:
        value = float(gains[label])
        pretty_label = label.replace(" -> ", r" $\rightarrow$ ")
        _inward_brace(
            ax_gain, y0, y1, x=x,
            text=f"$\\Delta EI$ ({pretty_label})\n= {_metric_value(value)}",
            depth=0.18, text_offset=0.20, fontsize=7.3,
        )
    fig.suptitle("Figure 1 · EI existence across the extended coarse-graining hierarchy", x=0.52, y=0.985, fontsize=15, fontweight="bold", color=TEAL_DARK)
    fig.text(0.02, 0.50, "coarse-graining direction", rotation=90, va="center", ha="center", fontsize=8.5, color=MUTED)
    if preview_notice:
        fig.text(.5, .015, preview_notice, ha="center", va="bottom", fontsize=8, color="#8A5529")
    elif demo:
        fig.text(0.99, 0.99, "EXPLORATORY REAL DATA · NOT FINAL PAPER RESULTS", ha="right", va="top", fontsize=8.5, color="#A66723")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _assignment_layer(spot: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    coords = _norm_xy(spot)[["spot_id", "xn", "yn"]]
    return coords.merge(assignments[["spot_id", "hard_cluster"]], on="spot_id", how="inner")


def _draw_optimal_stage(
    ax: plt.Axes,
    spot: pd.DataFrame,
    assignments: pd.DataFrame,
    stage: str,
    *,
    max_connectors: int | None = None,
    show_layer_labels: bool = True,
) -> int:
    coords = _norm_xy(spot)[["spot_id", "xn", "yn"]]
    yscale = .43
    ax.scatter(coords.xn, coords.yn * yscale, s=2.5, color=SPOT,
               alpha=.90, linewidths=0, rasterized=True, zorder=2)
    hardk = _draw_centroid_mapping(ax, spot, assignments, "hard_cluster", 0., 1.35,
                                    yscale=yscale, cmap="tab20")
    if show_layer_labels:
        # One shared set of layer labels is enough; suppressing them on the target
        # panel prevents any collision with the PIJ arrow column.
        ax.text(-0.055, 0.21, "Spot / Local", ha="right", va="center", fontsize=8.2, color=TEXT)
        ax.text(-0.055, 1.56, f"Optimal Macro (K={hardk})", ha="right", va="center", fontsize=8.2, color=TEXT)
    ax.set_xlim(-0.18 if show_layer_labels else -0.02, 1.04)
    ax.set_ylim(-0.08, 2.00)
    _clean(ax)
    ax.set_title(f"E{stage}", fontsize=12.0, fontweight="bold", color=TEAL_DARK, pad=4)
    return hardk


def render_figure2(
    *,
    source_stage: str,
    target_stage: str,
    source_spot: pd.DataFrame,
    target_spot: pd.DataFrame,
    source_assignments: pd.DataFrame,
    target_assignments: pd.DataFrame,
    delta_ei: float,
    method_label: str,
    output_path: Path,
    full_slice_source: pd.DataFrame | None = None,
    demo: bool = False,
    dpi: int = 300,
    preview_notice: str | None = None,
    input_label: str = "Spot",
    organ: str = "heart",
) -> None:
    """Render optimal coarse-graining with the same inward-brace grammar as Figure 1."""

    set_asset_style()
    fig = plt.figure(figsize=(19.2, 6.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.28, 2.60, 0.72, 2.60, 2.75], wspace=0.22)
    ax_context = fig.add_subplot(gs[0, 0])
    ax_source = fig.add_subplot(gs[0, 1])
    ax_arrow = fig.add_subplot(gs[0, 2])
    ax_target = fig.add_subplot(gs[0, 3])
    ax_metric = fig.add_subplot(gs[0, 4])

    draw_context(ax_context, full_slice_source, source_spot, f"E{source_stage} sample", organ=organ)
    kt = _draw_optimal_stage(ax_source, source_spot, source_assignments, source_stage, show_layer_labels=True)
    ktp = _draw_optimal_stage(ax_target, target_spot, target_assignments, target_stage, show_layer_labels=False)

    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(-0.08, 2.00)
    for y, label in ((0.22, "Pij_micro"), (1.57, "Pij_macro")):
        ax_arrow.annotate("", xy=(0.91, y), xytext=(0.09, y), arrowprops=dict(arrowstyle="->", color=TEAL_DARK, linewidth=1.35))
        ax_arrow.text(0.5, y + 0.14, label, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=TEXT)
    _clean(ax_arrow)

    ax_metric.set_xlim(0, 3.45)
    ax_metric.set_ylim(-0.08, 2.00)
    _clean(ax_metric)

    # Exactly the same inward-opening brace used by Figure 1.
    _inward_brace(
        ax_metric, .215, 1.565, x=0.48,
        text=(
            f"$\\Delta EI$ ({input_label} $\\rightarrow$ Optimal Macro)" "\n"
            f"$= EI(Macro)-EI({input_label})$" "\n"
            f"= {_metric_value(delta_ei)}"
        ),
        depth=0.22, text_offset=0.25, fontsize=7.7,
    )
    ax_metric.text(
        0.47, 1.94,
        f"K(E{source_stage}) = {kt}     K(E{target_stage}) = {ktp}",
        ha="left", va="top", fontsize=8.2, fontweight="bold", color=TEXT,
    )

    fig.suptitle(f"Figure 2 · Learned optimal coarse-graining over the {organ} region", x=0.52, y=0.985, fontsize=15, fontweight="bold", color=TEAL_DARK)
    fig.subplots_adjust(bottom=0.16, top=0.88)
    fig.text(
        0.60, 0.055, f"Method: {method_label}",
        ha="center", va="center", fontsize=7.2, color=MUTED,
    )
    if preview_notice:
        fig.text(.5, .015, preview_notice, ha="center", va="bottom", fontsize=8, color="#8A5529")
    elif demo:
        fig.text(0.99, 0.99, "EXPLORATORY REAL DATA · NOT FINAL PAPER RESULTS", ha="right", va="top", fontsize=8.2, color=MUTED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
