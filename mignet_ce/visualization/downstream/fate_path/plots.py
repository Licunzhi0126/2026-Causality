from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..mappings import COLORS as UNIFIED_COLORS, DISPLAY_NAMES, MARKERS as UNIFIED_MARKERS, MAPPINGS

from ..style import (
    GOLD,
    NAVY,
    RED,
    SEQUENTIAL,
    add_panel_label,
    savefig,
    set_publication_style,
)


def plot_fate_paths(paths: pd.DataFrame, path: Path) -> None:
    """Figure 9: four-time-point maximum-product macro fate paths."""
    set_publication_style()
    times = [column.removeprefix("state_") for column in paths.columns if column.startswith("state_")]
    times = sorted(times, key=float)
    if len(times) != 4:
        raise ValueError(f"Expected four state columns, got {times}")
    top = paths.sort_values(["source_ei", "path_probability"], ascending=False).head(12).copy()
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 9 | Four-time-point macro fate paths",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )

    ax = axes[0, 0]
    y_positions: dict[tuple[str, str], float] = {}
    for time_index, time in enumerate(times):
        states = list(dict.fromkeys(top[f"state_{time}"].astype(str).tolist()))
        positions = np.linspace(0.08, 0.92, len(states)) if len(states) > 1 else np.asarray([0.5])
        for state, y_value in zip(states, positions):
            y_positions[(time, state)] = float(y_value)
            ax.scatter(time_index, y_value, s=115, color="white", edgecolor=NAVY, linewidth=1.0, zorder=3)
            ax.text(
                time_index,
                y_value,
                state.replace("domain_", "D"),
                ha="center",
                va="center",
                fontsize=5.7,
                color=NAVY,
                zorder=4,
            )
    maximum_probability = max(float(top["path_probability"].max()), 1e-12)
    for _, row in top.iterrows():
        for left, right in zip(times[:-1], times[1:]):
            start = (times.index(left), y_positions[(left, str(row[f"state_{left}"]))])
            end = (times.index(right), y_positions[(right, str(row[f"state_{right}"]))])
            scaled = row["path_probability"] / maximum_probability
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=RED,
                alpha=0.18 + 0.58 * scaled,
                lw=0.6 + 3.8 * scaled,
                zorder=1,
            )
    ax.set_xlim(-0.2, len(times) - 0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(times)), times)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Representative high-EI fate graph")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ranked = paths.sort_values("path_probability", ascending=False).head(10).copy()
    labels = [
        f"{source.replace('domain_', 'D')} to {target.replace('domain_', 'D')}"
        for source, target in zip(ranked[f"state_{times[0]}"], ranked[f"state_{times[-1]}"])
    ]
    ax.barh(np.arange(len(ranked))[::-1], ranked["path_probability"].to_numpy()[::-1], color=RED)
    ax.set_yticks(np.arange(len(ranked))[::-1], labels[::-1], fontsize=6.5)
    ax.set_xlabel("Maximum-product path probability")
    ax.set_title("Top complete paths")
    ax.grid(axis="x")
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    ax.scatter(
        paths["first_branch_entropy"],
        paths["source_ei"],
        c=paths["path_probability"],
        cmap=SEQUENTIAL,
        s=55,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("First-step branch entropy (bit)")
    ax.set_ylabel("Source state EI (bit)")
    ax.set_title("High EI and branch concentration")
    ax.grid(True)
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    scatter = ax.scatter(
        paths["source_ei"],
        paths["path_probability"],
        c=paths["first_branch_entropy"],
        cmap=SEQUENTIAL,
        s=55,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("Source state EI (bit)")
    ax.set_ylabel("Maximum-product path probability")
    ax.set_title("Source information vs path dominance")
    ax.grid(True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("First-step branch entropy")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    scatter = ax.scatter(
        paths["first_branch_entropy"],
        paths["endpoint_entropy"],
        c=paths["source_ei"],
        cmap=SEQUENTIAL,
        s=55,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("First-step branch entropy (bit)")
    ax.set_ylabel(f"Endpoint entropy at {times[-1]} (bit)")
    ax.set_title("Local branching vs terminal uncertainty")
    ax.grid(True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("Source state EI")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    endpoints = (
        paths.groupby(f"state_{times[-1]}")
        .agg(
            path_count=(f"state_{times[0]}", "size"),
            total_probability=("path_probability", "sum"),
        )
        .sort_values("total_probability", ascending=False)
        .head(10)
    )
    ax.bar(np.arange(len(endpoints)), endpoints["total_probability"], color=GOLD)
    ax.set_xticks(
        np.arange(len(endpoints)),
        [str(state).replace("domain_", "D") for state in endpoints.index],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Summed max-path probability")
    ax.set_title("Most frequent terminal states")
    ax.grid(axis="y")
    add_panel_label(ax, "F")
    savefig(fig, path)


def plot_unified_fate_paths(paths: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    figure, axes = plt.subplots(2, 4, figsize=(18.0, 9.0), constrained_layout=True)
    for column, mapping in enumerate(MAPPINGS):
        ax = axes[0, column]
        subset = paths[paths["mapping"] == mapping]
        scatter = ax.scatter(
            subset["first_branch_entropy"],
            subset["path_probability"],
            c=subset["source_ei"],
            cmap=SEQUENTIAL,
            s=18,
            alpha=0.75,
            edgecolor="none",
        )
        ax.set_xlabel("First-step branch entropy (bit)")
        ax.set_ylabel("Main-path probability")
        ax.set_title(DISPLAY_NAMES[mapping], fontsize=8)
        ax.grid(True)
        add_panel_label(ax, chr(ord("A") + column))
        if column == 3:
            figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="Source state EI")

    ax = axes[1, 0]
    for mapping in MAPPINGS:
        subset = paths[paths["mapping"] == mapping]
        ax.scatter(
            subset["source_ei"],
            subset["endpoint_entropy"],
            s=15,
            alpha=0.5,
            color=UNIFIED_COLORS[mapping],
            marker=UNIFIED_MARKERS[mapping],
            label=DISPLAY_NAMES[mapping],
        )
    ax.set_xlabel("Source state EI (bit)")
    ax.set_ylabel("Endpoint entropy (bit)")
    ax.set_title("EI and terminal fate concentration")
    ax.grid(True)
    ax.legend(fontsize=6.1)
    add_panel_label(ax, "E")

    ax = axes[1, 1]
    categories = ["Path probability", "Branch entropy", "Endpoint entropy"]
    x_values = np.arange(3)
    width = min(0.78 / len(MAPPINGS), 0.22)
    offsets = (np.arange(len(MAPPINGS)) - (len(MAPPINGS) - 1) / 2) * width
    for offset, mapping in zip(offsets, MAPPINGS):
        subset = paths[paths["mapping"] == mapping]
        values = [
            subset["path_probability"].mean(),
            subset["first_branch_entropy"].mean(),
            subset["endpoint_entropy"].mean(),
        ]
        ax.bar(x_values + offset, values, width * 0.92, color=UNIFIED_COLORS[mapping])
    ax.set_xticks(x_values, categories, rotation=18, ha="right")
    ax.set_title("Fate-graph summary")
    ax.grid(axis="y")
    add_panel_label(ax, "F")

    ax = axes[1, 2]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for mapping in MAPPINGS:
        subset = paths[paths["mapping"] == mapping]
        ax.plot(
            quantiles,
            np.quantile(subset["path_probability"], quantiles),
            marker=UNIFIED_MARKERS[mapping],
            color=UNIFIED_COLORS[mapping],
            lw=1.8,
        )
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Main-path probability")
    ax.set_title("Path-stability distribution")
    ax.grid(True)
    add_panel_label(ax, "G")

    ax = axes[1, 3]
    for mapping in MAPPINGS:
        subset = paths[paths["mapping"] == mapping]
        ax.scatter(
            subset["first_branch_entropy"],
            subset["endpoint_entropy"],
            s=14,
            alpha=0.5,
            color=UNIFIED_COLORS[mapping],
            marker=UNIFIED_MARKERS[mapping],
        )
    ax.set_xlabel("First branch entropy (bit)")
    ax.set_ylabel("Endpoint entropy (bit)")
    ax.set_title("Local branching vs terminal uncertainty")
    ax.grid(True)
    add_panel_label(ax, "H")
    figure.suptitle(
        "Long-range macro fate paths across four representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)
