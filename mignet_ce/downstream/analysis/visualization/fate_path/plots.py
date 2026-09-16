from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import NAVY, SEQUENTIAL, add_panel_label, savefig, set_publication_style


def plot_unified_fate_paths(paths: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    figure, axes = plt.subplots(2, len(MAPPINGS), figsize=(21.0, 9.0), constrained_layout=True)
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
        ax.set_title(mapping, fontsize=8)
        ax.grid(True)
        add_panel_label(ax, chr(ord("A") + column))
        if column == len(MAPPINGS) - 1:
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
            label=mapping,
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
    for extra in axes[1, 4:]:
        extra.axis("off")
    figure.suptitle(
        "Long-range macro fate paths across five representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


