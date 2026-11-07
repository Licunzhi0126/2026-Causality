"""Render each domain map over the same original tissue geometry."""

from __future__ import annotations

from colorsys import hls_to_rgb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mignet_ce.downstream.formal_core_analysis.methods import METHOD_LABELS
from mignet_ce.downstream.paper_assets.style import MUTED, TEAL_DARK, TEXT, set_asset_style

from .inputs import INPUT_LABELS, METHOD_ORDER, TARGET_K, LearnedDomains, StageMaps


SPOT_COLOR = "#778896"
PENDING_COLOR = "#D6DEE5"
PREVIEW_COLOR = "#884319"


def _scatter(
    ax: plt.Axes, stage: StageMaps, labels: pd.Series | None, *, size: float, pending: bool = False,
) -> None:
    xy = stage.coordinates
    colors = PENDING_COLOR if pending else SPOT_COLOR
    if labels is not None:
        domains, codes = np.unique(labels.reindex(xy.index), return_inverse=True)
        palette = np.array([
            hls_to_rgb((0.60 + i * 0.61803398875) % 1, 0.46 + 0.10 * (i % 3), 0.68)
            for i in range(len(domains))
        ])
        colors = palette[codes]
    ax.scatter(xy["x"], xy["y"], c=colors, s=size, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_axis_off()


def _header(fig: plt.Figure, title: str, subtitle: str, *, preview: bool = False) -> None:
    fig.text(0.055, 0.935, title, fontsize=18, weight="bold", color=TEAL_DARK, va="top")
    fig.text(0.055, 0.861, subtitle, fontsize=10, color=MUTED, va="top")
    if preview:
        fig.text(0.964, 0.93, "LAYOUT PREVIEW", ha="right", va="top", fontsize=9, weight="bold",
                 color=PREVIEW_COLOR, bbox={"facecolor": "#FCF0E5", "edgecolor": "none", "pad": 5})


def _row_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position(original=True)
    fig.text(0.055, (box.y0 + box.y1) / 2, label, ha="left", va="center", fontsize=10,
             weight="bold", color=TEAL_DARK)


def _save(fig: plt.Figure, path: Path) -> None:
    try:
        for extension in ("png", "pdf"):
            with path.with_suffix(f".{extension}").open("xb") as handle:
                fig.savefig(handle, format=extension, dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def render_natural(stages: tuple[StageMaps, StageMaps], path: Path) -> None:
    set_asset_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6))
    fig.subplots_adjust(left=0.17, right=0.964, bottom=0.14, top=0.73, wspace=0.18, hspace=0.20)
    _header(fig, "Spatial domains at three input scales", "Mouse heart · Original spatial coordinates")

    for col, (scale, title) in enumerate(INPUT_LABELS.items()):
        axes[0, col].set_title(title, y=1.04, color=TEXT)
        for row, stage in enumerate(stages):
            _scatter(axes[row, col], stage, stage.seurat.get(scale), size=6)
    for row, stage in enumerate(stages):
        _row_label(fig, axes[row, 0], f"E{stage.stage}")

    fig.text(0.055, 0.062, "Each point is an original Spot. Colors show the Seurat domains within each slice.",
             fontsize=8.5, color=MUTED)
    _save(fig, path)


def render_comparison(
    stages: tuple[StageMaps, StageMaps], scale: str,
    domains: dict[str, tuple[LearnedDomains, LearnedDomains]] | None, path: Path,
) -> None:
    set_asset_style()
    preview = domains is None
    fig = plt.figure(figsize=(15.4, 5.2))
    grid = fig.add_gridspec(1, 4, left=0.135, right=0.964, bottom=0.18, top=0.745, wspace=0.20)
    source, target = (f"E{stage.stage}" for stage in stages)
    _header(fig, f"Learned domains from {INPUT_LABELS[scale]}",
            f"Mouse heart · {source} → {target} · Target K = {TARGET_K[scale]}", preview=preview)

    for col, method in enumerate(METHOD_ORDER):
        cell = grid[col]
        axes = cell.subgridspec(2, 2, wspace=0.10, hspace=0.23).subplots()
        box = cell.get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 0.805, METHOD_LABELS[method], ha="center", va="center",
                 weight="bold", fontsize=10, color=TEXT)
        if col == 0:
            _row_label(fig, axes[0, 0], "Input layer")
            _row_label(fig, axes[1, 0], "Learned\ndomains")

        for side, stage in enumerate(stages):
            input_ax, domain_ax = axes[:, side]
            input_slot = input_ax.get_position(original=True)
            fig.text((input_slot.x0 + input_slot.x1) / 2, input_slot.y1 - 0.033, f"E{stage.stage}",
                     ha="center", va="center", fontsize=8.5, color=MUTED)
            _scatter(input_ax, stage, stage.seurat.get(scale), size=2.0)
            learned = None if preview else domains[method][side]
            _scatter(domain_ax, stage, None if preview else learned.labels, size=2.0, pending=preview)
            if preview:
                domain_ax.text(0.5, 0.5, "Awaiting\nassignments", transform=domain_ax.transAxes,
                               ha="center", va="center", fontsize=8, color=MUTED,
                               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
            metric = r"$K_{\mathrm{eff}}$" + (" unavailable" if preview else f" = {learned.keff:.2f}")
            # Keep the annotation baseline aligned even when the two tissues have different shapes.
            slot = domain_ax.get_position(original=True)
            fig.text((slot.x0 + slot.x1) / 2, slot.y0 - 0.025, metric, ha="center", va="top",
                     fontsize=8.5, color=MUTED if preview else TEAL_DARK)

    caption = (
        "Top row shows real inputs. Lower panels are placeholders for the saved assignments."
        if preview else
        "Colors show the highest-probability assignment within each panel. "
        "Effective counts use the full soft assignments."
    )
    fig.text(0.055, 0.062, caption, fontsize=8.5, color=MUTED)
    _save(fig, path)
