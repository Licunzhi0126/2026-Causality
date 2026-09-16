from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import NAVY, SEQUENTIAL, add_panel_label, savefig, set_publication_style


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
        "Spatial localization of EI across spot and five macro representations",
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
            ("Keff", "Effective prototypes", "Effective prototype usage"),
            ("soft_usage_entropy", "Usage entropy (bit)", "Soft prototype usage entropy"),
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
            ("moran_i", "Mean Moran's I", "Soft spatial continuity"),
            ("neighborhood_smoothness", "Mean neighborhood difference", "Soft neighborhood smoothness"),
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
        "Effective states and spatial morphology across five representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


