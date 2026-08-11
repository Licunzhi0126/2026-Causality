from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS

from ..style import (
    BLUE,
    GOLD,
    MUTED,
    NAVY,
    RED,
    SEQUENTIAL,
    TIME_COLORS,
    add_panel_label,
    savefig,
    set_publication_style,
)


def plot_spatial_state_ei(spatial_frames: list[pd.DataFrame], path: Path) -> None:
    """Figure 2: three adjacent time pairs at K150 and K40."""
    set_publication_style()
    layer_rank = {"seurat_k150": 0, "seurat_k40": 1}
    frames = sorted(
        spatial_frames,
        key=lambda frame: (
            layer_rank.get(str(frame["layer"].iloc[0]), 99),
            float(frame["source_time"].iloc[0]),
        ),
    )
    if len(frames) != 6:
        raise ValueError(f"Expected exactly six spatial frames, received {len(frames)}")
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4), constrained_layout=True)
    fig.suptitle(
        "Figure 2 | Spatial distribution of state-level effective information",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    for row in range(2):
        row_frames = frames[row * 3 : row * 3 + 3]
        maximum = max(float(frame["state_ei"].quantile(0.98)) for frame in row_frames)
        maximum = max(maximum, 1e-8)
        mappable = None
        for column, frame in enumerate(row_frames):
            ax = axes[row, column]
            layer = str(frame["layer"].iloc[0])
            mappable = ax.scatter(
                frame["x"],
                -frame["y"],
                c=frame["state_ei"],
                cmap=SEQUENTIAL,
                vmin=0,
                vmax=maximum,
                s=6.3 if layer == "seurat_k150" else 8.3,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            scale = "K150" if layer == "seurat_k150" else "K40"
            ax.set_title(f"{scale} | {frame['source_time'].iloc[0]} to {frame['target_time'].iloc[0]}")
            add_panel_label(ax, chr(ord("A") + row * 3 + column))
        colorbar = fig.colorbar(mappable, ax=axes[row, :].tolist(), fraction=0.018, pad=0.015)
        colorbar.set_label(f"State-level EI at {'K150' if row == 0 else 'K40'} (bit)")
    savefig(fig, path)


def plot_effective_spatial(effective: pd.DataFrame, state_spatial: pd.DataFrame, path: Path) -> None:
    """Figure 7: effective state usage and spatial morphology."""
    set_publication_style()
    times = sorted(effective["time"].astype(str).unique(), key=float)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 7 | Effective state usage and spatial morphology",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    for ax, metric, ylabel, title, panel in (
        (axes[0, 0], "hardK", "Number of occupied states", "Hard state count", "A"),
        (axes[0, 1], "Keff", "Effective number of states", "Usage-entropy effective count", "B"),
    ):
        for layer, color, marker, name in (
            ("seurat_k150", BLUE, "o", "K150"),
            ("seurat_k40", RED, "s", "K40"),
        ):
            subset = effective[effective["layer"] == layer].copy()
            subset["time"] = subset["time"].astype(str)
            subset = subset.set_index("time").loc[times].reset_index()
            ax.plot(subset["time"], subset[metric], marker=marker, color=color, lw=2.1, label=name)
        ax.set_xlabel("Embryonic time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.legend(loc="best")
        add_panel_label(ax, panel)

    ax = axes[0, 2]
    for layer, color, marker, nominal, name in (
        ("seurat_k150", BLUE, "o", 150, "K150"),
        ("seurat_k40", RED, "s", 40, "K40"),
    ):
        subset = effective[effective["layer"] == layer].copy()
        subset["time"] = subset["time"].astype(str)
        subset = subset.set_index("time").loc[times]
        ax.plot(times, subset["Keff"] / nominal, marker=marker, color=color, lw=2.1, label=name)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Embryonic time")
    ax.set_ylabel("Keff / nominal K")
    ax.set_title("Fraction of nominal capacity actually used")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    k40 = effective[effective["layer"] == "seurat_k40"].copy()
    k40["time"] = k40["time"].astype(str)
    k40 = k40.set_index("time").loc[times]
    ax.bar(np.arange(len(times)), k40["max_usage"], color=GOLD, label="Largest state usage")
    ax.plot(np.arange(len(times)), [1 / 40] * len(times), color=NAVY, ls="--", lw=1.4, label="Uniform K40 = 1/40")
    ax.set_xticks(np.arange(len(times)), times)
    ax.set_xlabel("Embryonic time")
    ax.set_ylabel("Usage fraction")
    ax.set_title("K40 occupancy imbalance")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "D")

    plotted = state_spatial[state_spatial["layer"] == "seurat_k40"].copy()
    ax = axes[1, 1]
    sizes = 18 + 110 * np.sqrt(plotted["spot_count"] / max(plotted["spot_count"].max(), 1))
    scatter = ax.scatter(
        plotted["boundary_ratio"],
        plotted["state_ei"],
        s=sizes,
        c=plotted["moran_i"],
        cmap=SEQUENTIAL,
        alpha=0.83,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("Boundary ratio")
    ax.set_ylabel("State-level EI (bit)")
    ax.set_title("State EI and spatial interiority")
    ax.grid(True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("Moran's I")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    for time in sorted(plotted["time"].astype(str).unique(), key=float):
        subset = plotted[plotted["time"].astype(str) == time]
        ax.scatter(
            subset["fragmentation"],
            subset["state_ei"],
            s=24 + 75 * np.sqrt(subset["spot_count"] / max(plotted["spot_count"].max(), 1)),
            color=TIME_COLORS.get(time, MUTED),
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
            label=time,
        )
    ax.set_xlabel("Fragmentation index")
    ax.set_ylabel("State-level EI (bit)")
    ax.set_title("EI versus spatial fragmentation")
    ax.grid(True)
    ax.legend(title="Source time", loc="best", ncol=2)
    add_panel_label(ax, "F")
    savefig(fig, path)


def plot_unified_spatial_state_ei(spatial_spots: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    pairs = list(dict.fromkeys(spatial_spots["time_pair"].astype(str)))
    mappings = ("Spot", *MAPPINGS)
    figure, axes = plt.subplots(
        len(mappings),
        len(pairs),
        figsize=(15.6, 3.05 * len(mappings)),
        squeeze=False,
        constrained_layout=True,
    )
    finite = spatial_spots.loc[np.isfinite(spatial_spots["state_ei"]), "state_ei"].to_numpy(dtype=float)
    vmax = max(float(np.quantile(finite, 0.98)), 1e-6)
    scatter = None
    panel = 0
    for row, mapping in enumerate(mappings):
        for column, pair in enumerate(pairs):
            ax = axes[row, column]
            subset = spatial_spots[
                (spatial_spots["mapping"] == mapping) & (spatial_spots["time_pair"] == pair)
            ]
            scatter = ax.scatter(
                subset["x"],
                subset["y"],
                c=subset["state_ei"],
                s=2.4 if mapping == "Spot" else 3.1,
                cmap=SEQUENTIAL,
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{mapping} · {pair.replace('->', '→')}", fontsize=8)
            add_panel_label(ax, chr(ord("A") + panel) if panel < 26 else f"P{panel + 1}")
            panel += 1
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, shrink=0.76, label="State-level EI contribution (bit)")
    figure.suptitle(
        "Spatial localization of EI across spot and four macro representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


def plot_unified_effective_spatial(
    effective: pd.DataFrame,
    spatial: pd.DataFrame,
    path: Path,
) -> None:
    set_publication_style()
    times = list(dict.fromkeys(effective["time"].astype(str)))
    pairs = list(dict.fromkeys(spatial["time_pair"].astype(str)))
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    for column, (metric, ylabel, title) in enumerate(
        [
            ("active_k", "Active states", "Active macrostate count"),
            ("Keff", "Effective state count", "Usage-effective states"),
            ("assignment_confidence", "Assignment confidence", "Assignment confidence"),
        ]
    ):
        ax = axes[0, column]
        for mapping in MAPPINGS:
            subset = effective[effective["mapping"] == mapping].copy()
            subset["time"] = subset["time"].astype(str)
            subset = subset.set_index("time").reindex(times)
            ax.plot(
                range(len(times)),
                subset[metric],
                marker=UNIFIED_MARKERS[mapping],
                color=UNIFIED_COLORS[mapping],
                lw=1.8,
                label=mapping,
            )
        ax.set_xticks(range(len(times)), times)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        add_panel_label(ax, chr(ord("A") + column))
        if column == 2:
            ax.legend(fontsize=6.3)
    for column, (metric, ylabel, title) in enumerate(
        [
            ("fragmentation", "Mean fragmentation", "Spatial fragmentation"),
            ("boundary_ratio", "Mean boundary ratio", "Boundary exposure"),
            ("moran_i", "Mean Moran's I", "Spatial autocorrelation"),
        ]
    ):
        ax = axes[1, column]
        aggregate = spatial.groupby(["mapping", "time_pair"])[metric].mean().reset_index()
        for mapping in MAPPINGS:
            subset = aggregate[aggregate["mapping"] == mapping].set_index("time_pair").reindex(pairs)
            ax.plot(
                range(len(pairs)),
                subset[metric],
                marker=UNIFIED_MARKERS[mapping],
                color=UNIFIED_COLORS[mapping],
                lw=1.8,
            )
        ax.set_xticks(range(len(pairs)), [pair.replace("->", "→") for pair in pairs])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        add_panel_label(ax, chr(ord("D") + column))
    figure.suptitle(
        "Effective states and spatial morphology across four representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)
