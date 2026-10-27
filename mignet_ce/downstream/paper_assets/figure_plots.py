"""Spatial figures for paper_assets, using the approved Generalization Audit grammar.

The shared renderer supplies the three-layer paper figure.  Extended K10 and
optimal-macro panels use the same isotropic geometry, *complete* centroid
connections, Pij arrows, and inward-facing square brackets.  Numerical values
are always supplied by the existing paper_assets workflow, not recomputed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from mignet_ce.downstream.generalization_audit.plots import (
    CONNECTOR,
    MUTED,
    SPOT,
    TEAL_DARK,
    TEXT,
    _draw_gain_brackets,
    _draw_pij_column,
    _draw_square_bracket,
    _draw_stage,
    _layer_y,
    _prepare_stage,
    _set_style,
    render_spatial_hierarchy_audit,
)


def _assert_spot_agreement(spot: pd.DataFrame, k150: pd.DataFrame, stage: str) -> None:
    """Fail rather than silently dropping spots from the paper figure."""
    required = {"spot_id", "x", "y"}
    if not required.issubset(spot.columns):
        raise ValueError(f"E{stage} spot coordinates missing {required - set(spot.columns)}")
    if spot["spot_id"].astype(str).duplicated().any():
        raise ValueError(f"E{stage} spot coordinates have duplicated spot IDs")
    a = spot.assign(spot_id=spot["spot_id"].astype(str)).set_index("spot_id")
    b = k150.assign(spot_id=k150["spot_id"].astype(str)).set_index("spot_id")
    if set(a.index) != set(b.index):
        raise ValueError(f"E{stage} spot IDs do not match K150 domain-map IDs")
    xya = a.loc[b.index, ["x", "y"]].to_numpy(float)
    xyb = b[["x", "y"]].to_numpy(float)
    if not (np.isfinite(xya).all() and np.allclose(xya, xyb, atol=1e-5, rtol=0)):
        raise ValueError(f"E{stage} spot coordinates do not match K150 domain-map coordinates")


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
) -> None:
    """Figure 1: exact approved Generalization Audit centroid/bracket layout.

    The formerly schematic embryo/ROI panel is intentionally omitted: the
    user-approved style is the spatial hierarchy itself. ``full_slice_source``
    remains accepted to keep the existing workflow API backward compatible.
    """
    _assert_spot_agreement(source_spot, source_k150, source_stage)
    _assert_spot_agreement(target_spot, target_k150, target_stage)
    render_spatial_hierarchy_audit(
        source_stage=source_stage,
        target_stage=target_stage,
        source_k150_map=source_k150,
        source_k40_map=source_k40,
        target_k150_map=target_k150,
        target_k40_map=target_k40,
        gains=gains,
        output_path=output_path,
        organ="heart",
        style_preview=demo,
        dpi=dpi,
        title="Figure 1 · Causal emergence across spatial scales",
        subtitle="Every Spot → its K150-domain centroid  |  Every K150 centroid → dominant-overlap K40 centroid",
        footnote=(
            "Domain membership and dominant spot overlap define vertical connections; "
            "Pij arrows depict time transitions, not domain identity tracking."
        ),
    )


def _validate_extension(stage: dict[str, object], k10: pd.DataFrame, name: str) -> dict[str, object]:
    """Join real K10 labels by spot ID, then derive dominant K40→K10 links."""
    required = {"spot_id", "domain_id", "x", "y"}
    if not required.issubset(k10.columns):
        raise ValueError(f"{name}: K10 map missing {required - set(k10.columns)}")
    ext = k10.copy()
    if ext["domain_id"].isna().any():
        raise ValueError(f"{name}: missing K10 domain labels")
    ext["spot_id"] = ext["spot_id"].astype(str)
    ext["domain_id"] = ext["domain_id"].astype(str)
    if ext["spot_id"].duplicated().any() or len(ext) != stage["n_spots"]:
        raise ValueError(f"{name}: K10 map must contain one row per spot")
    spots: pd.DataFrame = stage["spots"]
    joined = spots.merge(
        ext[["spot_id", "domain_id", "x", "y"]].rename(
            columns={"domain_id": "domain10", "x": "x10", "y": "y10"}
        ), on="spot_id", validate="one_to_one", how="left",
    )
    if joined["domain10"].isna().any():
        raise ValueError(f"{name}: K10 spot IDs do not match K40/K150 IDs")
    for col, other in (("x", "x10"), ("y", "y10")):
        if not np.allclose(joined[col].to_numpy(float), joined[other].to_numpy(float), atol=1e-5, rtol=0):
            raise ValueError(f"{name}: K10 coordinates differ from other layers")
    centers10 = joined.groupby("domain10", sort=True)[["xn", "yn"]].mean().reset_index()
    overlap = pd.crosstab(joined["domain40"], joined["domain10"])
    dominant = overlap.idxmax(axis=1).rename("domain10").reset_index()
    centers40 = stage["centers40"]
    mapping = centers40.merge(dominant, on="domain40", validate="one_to_one")
    mapping = mapping.merge(
        centers10.rename(columns={"xn": "x10n", "yn": "y10n"}),
        on="domain10", validate="many_to_one",
    )
    labels = sorted(joined["domain10"].unique().tolist())
    cmap = plt.get_cmap("turbo")
    colors = {d: cmap(t) for d, t in zip(labels, np.linspace(0.03, 0.97, len(labels)))}
    return {"spots": joined, "centers10": centers10, "mapping": mapping, "colors10": colors}


def _draw_k10_layer(ax: plt.Axes, extension: dict[str, object]) -> None:
    """Add a K10 layer and one wire per K40 domain centroid (no subsampling)."""
    spots = extension["spots"]
    centers10 = extension["centers10"]
    mapping = extension["mapping"]
    colors10 = extension["colors10"]
    src = np.column_stack((mapping["xn"], np.asarray(mapping["yn"]) * 0.62 + 2.0))
    tgt = np.column_stack((mapping["x10n"], np.asarray(mapping["y10n"]) * 0.62 + 3.0))
    links = np.stack([src, tgt], axis=1)
    ax.add_collection(LineCollection(
        links, colors=CONNECTOR, alpha=0.17, linewidths=0.45, zorder=0.5, rasterized=True
    ))
    y_spots = spots["yn"].to_numpy(float) * 0.62 + 3.0
    ax.scatter(spots["xn"], y_spots, s=2.25,
               c=[colors10[v] for v in spots["domain10"]],
               alpha=0.92, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(centers10["xn"], centers10["yn"].to_numpy(float) * 0.62 + 3.0,
               s=15, c=[colors10[v] for v in centers10["domain10"]],
               edgecolors="white", linewidths=0.60, zorder=4)
    ax.text(-0.055, 3.31, "K10", ha="right", va="center", fontsize=8.7, color=TEXT)
    ax.set_ylim(-0.12, 3.77)


def _draw_brackets_k10(ax: plt.Axes, gains: Mapping[str, float]) -> None:
    """Follow approved square-bracket style for three local and three totals."""
    required = (
        "Spot -> K150", "K150 -> K40", "K40 -> K10",
        "Spot -> K40", "K150 -> K10", "Spot -> K10",
    )
    for key in required:
        if key not in gains or not np.isfinite(float(gains[key])):
            raise ValueError(f"K10 figure requires a finite measured {key} value")
    # Three adjacent, inward-opening braces with small gaps at shared levels.
    brackets = (
        ("Spot -> K150", 0.31, 1.31 - 0.045, 0.35, 0.5),
        ("K150 -> K40", 1.31 + 0.045, 2.31 - 0.045, 0.35, 0.5),
        ("K40 -> K10", 2.31 + 0.045, 3.31, 0.35, 0.5),
        ("Spot -> K40", 0.31, 2.31, 1.88, 2.0),
        ("K150 -> K10", 1.31, 3.31, 1.88, 2.0),
        ("Spot -> K10", 0.31, 3.31, 3.44, 3.6),
    )
    for key, y0, y1, x, tx in brackets:
        _draw_square_bracket(ax, x, y0, y1, cap=0.085, lw=1.35)
        ax.text(tx, (y0 + y1) / 2, f"{float(gains[key]):+.3f} bits",
                ha="left", va="center", fontsize=9.4, color=TEXT)
    ax.set_xlim(0, 4.9)
    ax.set_ylim(-0.12, 3.77)
    ax.axis("off")
    ax.set_title("Hierarchy ΔEI", fontsize=11.0, fontweight="bold", color=TEAL_DARK, pad=6)


def render_figure1_k10(
    *, source_stage: str, target_stage: str,
    source_spot: pd.DataFrame, target_spot: pd.DataFrame,
    source_k150: pd.DataFrame, target_k150: pd.DataFrame,
    source_k40: pd.DataFrame, target_k40: pd.DataFrame,
    source_k10: pd.DataFrame, target_k10: pd.DataFrame,
    gains: Mapping[str, float], output_path: Path,
    full_slice_source: pd.DataFrame | None = None, demo: bool = False, dpi: int = 300,
) -> None:
    _assert_spot_agreement(source_spot, source_k150, source_stage)
    _assert_spot_agreement(target_spot, target_k150, target_stage)
    stage_src = _prepare_stage(source_k150, source_k40, f"E{source_stage}")
    stage_tgt = _prepare_stage(target_k150, target_k40, f"E{target_stage}")
    ext_src = _validate_extension(stage_src, source_k10, f"E{source_stage}")
    ext_tgt = _validate_extension(stage_tgt, target_k10, f"E{target_stage}")
    _set_style()
    fig = plt.figure(figsize=(21.5, 10.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.65, 0.70, 2.65, 4.7], wspace=0.23)
    ax_src, ax_arrow, ax_tgt, ax_gain = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _draw_stage(ax_src, stage_src, source_stage)
    _draw_k10_layer(ax_src, ext_src)
    _draw_stage(ax_tgt, stage_tgt, target_stage)
    _draw_k10_layer(ax_tgt, ext_tgt)
    _draw_pij_column(ax_arrow)
    ax_arrow.set_ylim(-0.12, 3.77)
    ax_arrow.annotate("", xy=(0.90, 3.31), xytext=(0.10, 3.31),
                      arrowprops={"arrowstyle": "->", "color": TEAL_DARK, "linewidth": 1.35})
    ax_arrow.text(0.50, 3.42, "Pij", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=TEXT)
    _draw_brackets_k10(ax_gain, gains)
    fig.suptitle("Figure 1 · Extended spatial hierarchy (K10)",
                 x=0.49, y=0.977, fontsize=15, fontweight="bold", color=TEAL_DARK)
    fig.text(0.49, 0.945,
             "Every spot → K150 centroid  |  Every K150 centroid → dominant K40 centroid  |  Every K40 centroid → dominant K10 centroid",
             ha="center", va="center", fontsize=8.2, color=MUTED)
    if demo:
        fig.text(0.985, 0.982, "STYLE PREVIEW · DEMO INPUTS", ha="right", va="top",
                 fontsize=8.0, fontweight="bold", color="#A66723")
    fig.text(0.49, 0.025,
             "All connections are within each developmental stage; between-stage Pij arrows show temporal dynamics.",
             ha="center", va="center", fontsize=7.4, color=MUTED)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.89, bottom=0.09)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _optimal_stage_data(spot: pd.DataFrame, assignments: pd.DataFrame, stage: str) -> dict[str, object]:
    if not {"spot_id", "x", "y"}.issubset(spot.columns):
        raise ValueError(f"E{stage}: missing Spot coordinates")
    if not {"spot_id", "hard_cluster"}.issubset(assignments.columns):
        raise ValueError(f"E{stage}: missing hard_cluster assignments")
    sp = spot[["spot_id", "x", "y"]].copy()
    cl = assignments[["spot_id", "hard_cluster"]].copy()
    if cl["hard_cluster"].isna().any():
        raise ValueError(f"E{stage}: missing macro hard_cluster labels")
    sp["spot_id"] = sp["spot_id"].astype(str)
    cl["spot_id"] = cl["spot_id"].astype(str)
    if sp["spot_id"].duplicated().any() or cl["spot_id"].duplicated().any():
        raise ValueError(f"E{stage}: duplicate Spot IDs")
    joined = sp.merge(cl, on="spot_id", how="outer", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"E{stage}: Spot and macro assignment IDs must match exactly")
    x = joined["x"].to_numpy(float)
    y = joined["y"].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"E{stage}: non-finite Spot coordinates")
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    joined["xn"] = (x - 0.5 * (x.max() + x.min())) / scale + 0.5
    joined["yn"] = 0.5 - (y - 0.5 * (y.max() + y.min())) / scale
    joined["hard_cluster"] = joined["hard_cluster"].astype(str)
    centers = joined.groupby("hard_cluster", sort=True)[["xn", "yn"]].mean()
    labels = sorted(joined["hard_cluster"].unique().tolist())
    cmap = plt.get_cmap("turbo")
    colors = {d: cmap(t) for d, t in zip(labels, np.linspace(0.03, 0.97, len(labels)))}
    return {"spots": joined, "centers": centers, "colors": colors, "hardk": len(labels)}


def _draw_optimal_stage(ax: plt.Axes, data: dict[str, object], stage: str) -> None:
    spots = data["spots"]
    centers = data["centers"]
    colors = data["colors"]
    spot_x = spots["xn"].to_numpy(float)
    spot_y = spots["yn"].to_numpy(float)
    tgt_x = spots["hard_cluster"].map(centers["xn"]).to_numpy(float)
    tgt_y = spots["hard_cluster"].map(centers["yn"]).to_numpy(float)
    # Every original spot connects to its learned hard macro-domain center.
    paths = np.stack([
        np.column_stack((spot_x, spot_y * 0.62)),
        np.column_stack((tgt_x, tgt_y * 0.62 + 1.0)),
    ], axis=1)
    ax.add_collection(LineCollection(paths, colors=CONNECTOR, alpha=0.035,
                                     linewidths=0.24, rasterized=True, zorder=0))
    ax.scatter(spot_x, spot_y * 0.62, s=2.2, c=SPOT, linewidths=0,
               alpha=0.92, rasterized=True, zorder=2)
    ax.scatter(spot_x, spot_y * 0.62 + 1.0, s=2.25,
               c=[colors[d] for d in spots["hard_cluster"]], linewidths=0,
               alpha=0.92, rasterized=True, zorder=2)
    ax.scatter(centers["xn"], centers["yn"].to_numpy(float) * 0.62 + 1.0,
               s=14, c=[colors[d] for d in centers.index],
               edgecolors="white", linewidths=0.55, zorder=4)
    ax.text(-0.055, 0.31, "Spot", ha="right", va="center", fontsize=8.7, color=TEXT)
    ax.text(-0.055, 1.31, f"Optimal Macro (K={data['hardk']})", ha="right", va="center", fontsize=8.2, color=TEXT)
    ax.set_xlim(-0.25, 1.04)
    ax.set_ylim(-0.12, 1.77)
    ax.axis("off")
    ax.set_title(f"E{stage}", fontsize=12, fontweight="bold", color=TEAL_DARK, pad=6)


def render_figure2(
    *, source_stage: str, target_stage: str,
    source_spot: pd.DataFrame, target_spot: pd.DataFrame,
    source_assignments: pd.DataFrame, target_assignments: pd.DataFrame,
    delta_ei: float, method_label: str, output_path: Path,
    full_slice_source: pd.DataFrame | None = None,
    demo: bool = False, dpi: int = 300,
) -> None:
    """Figure 2: same Spot→centroid wires and simple square bracket as Figure 1."""
    src = _optimal_stage_data(source_spot, source_assignments, source_stage)
    tgt = _optimal_stage_data(target_spot, target_assignments, target_stage)
    if not demo and not np.isfinite(delta_ei):
        raise ValueError("Figure 2 requires a finite measured trained ΔEI")
    _set_style()
    fig = plt.figure(figsize=(17.8, 6.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.65, 0.70, 2.65, 3.10], wspace=0.24)
    ax_src, ax_arrows, ax_tgt, ax_metric = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _draw_optimal_stage(ax_src, src, source_stage)
    _draw_optimal_stage(ax_tgt, tgt, target_stage)
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(-0.12, 1.77)
    for y, lbl in ((0.31, "Pij micro"), (1.31, "Pij macro")):
        ax_arrows.annotate("", xy=(0.90, y), xytext=(0.10, y),
                           arrowprops={"arrowstyle": "->", "color": TEAL_DARK, "linewidth": 1.35})
        ax_arrows.text(0.5, y + 0.12, lbl, ha="center", va="bottom", fontsize=7.7, color=TEXT)
    ax_arrows.axis("off")
    _draw_square_bracket(ax_metric, 0.50, 0.31, 1.31, cap=0.125, lw=1.55)
    label = f"{float(delta_ei):+.3f} bits" if not demo and np.isfinite(delta_ei) else "ΔEI pending"
    ax_metric.text(0.67, 0.81, label, fontsize=12.0, ha="left", va="center", color=TEXT)
    ax_metric.set_xlim(0, 2.9)
    ax_metric.set_ylim(-0.12, 1.77)
    ax_metric.axis("off")
    ax_metric.set_title("Optimal ΔEI", fontsize=11, color=TEAL_DARK, fontweight="bold", pad=6)
    fig.suptitle("Figure 2 · Learned optimal coarse-graining", x=0.49, y=0.977,
                 fontsize=15, fontweight="bold", color=TEAL_DARK)
    fig.text(0.49, 0.931,
             "Every Spot → its learned macro-domain centroid; no connector subsampling",
             ha="center", va="center", fontsize=8.4, color=MUTED)
    fig.text(0.49, 0.025, f"Method: {method_label} · independent Pij arrows represent source→target dynamics",
             fontsize=7.4, color=MUTED, ha="center", va="center")
    if demo:
        fig.text(0.985, 0.982, "STYLE PREVIEW · NOT A FIT", ha="right", va="top",
                 fontsize=8.0, fontweight="bold", color="#A66723")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.87, bottom=0.11)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
