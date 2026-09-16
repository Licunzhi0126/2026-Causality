from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import MUTED, NAVY, add_panel_label, savefig, set_publication_style


def plot_unified_ei_overview(metrics: pd.DataFrame, path: Path) -> None:
    """Unified causal-emergence and information decomposition figure."""

    set_publication_style()
    pairs = list(dict.fromkeys(metrics["time_pair"].astype(str)))
    x_values = np.arange(len(pairs))
    width = min(0.78 / len(MAPPINGS), 0.22)
    offsets = (np.arange(len(MAPPINGS)) - (len(MAPPINGS) - 1) / 2) * width
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Causal emergence across five complete macro representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )

    ax = axes[0, 0]
    for offset, mapping in zip(offsets, MAPPINGS):
        subset = metrics[metrics["mapping"] == mapping].set_index("time_pair").reindex(pairs)
        ax.bar(x_values + offset, subset["delta_EI"], width * 0.92, color=UNIFIED_COLORS[mapping])
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x_values, [pair.replace("->", "→") for pair in pairs])
    ax.set_ylabel("ΔEI vs spot (bit)")
    ax.set_title("Causal emergence relative to spot")
    ax.grid(axis="y")
    add_panel_label(ax, "A")

    for panel, (metric, ylabel, title) in enumerate(
        [
            ("H_effect", "Effect diversity H(Y) (bit)", "Effect diversity"),
            ("H_noise", "Conditional noise H(Y|X) (bit)", "Transition noise"),
        ],
        start=1,
    ):
        ax = axes[0, panel]
        for mapping in MAPPINGS:
            subset = metrics[metrics["mapping"] == mapping].set_index("time_pair").reindex(pairs)
            ax.plot(
                x_values,
                subset[metric],
                marker=UNIFIED_MARKERS[mapping],
                color=UNIFIED_COLORS[mapping],
                lw=1.8,
            )
        ax.set_xticks(x_values, [pair.replace("->", "→") for pair in pairs])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        add_panel_label(ax, chr(ord("A") + panel))

    ax = axes[1, 0]
    for mapping in MAPPINGS:
        subset = metrics[metrics["mapping"] == mapping]
        ax.scatter(
            subset["degeneracy"],
            subset["determinism"],
            s=58,
            color=UNIFIED_COLORS[mapping],
            marker=UNIFIED_MARKERS[mapping],
            edgecolor="white",
            label=mapping,
        )
    ax.set_xlabel("Degeneracy (bit)")
    ax.set_ylabel("Determinism (bit)")
    ax.set_title("Determinism–degeneracy plane")
    ax.grid(True)
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    spot = metrics.groupby("time_pair")["micro_EI"].mean().reindex(pairs)
    ax.plot(x_values, spot, "--o", color=MUTED, label="Spot")
    for mapping in MAPPINGS:
        subset = metrics[metrics["mapping"] == mapping].set_index("time_pair").reindex(pairs)
        ax.plot(
            x_values,
            subset["EI"],
            marker=UNIFIED_MARKERS[mapping],
            color=UNIFIED_COLORS[mapping],
            lw=1.8,
            label=mapping,
        )
    ax.set_xticks(x_values, [pair.replace("->", "→") for pair in pairs])
    ax.set_ylabel("Effective information (bit)")
    ax.set_title("Absolute EI across scales")
    ax.grid(axis="y")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    for mapping in MAPPINGS:
        subset = metrics[metrics["mapping"] == mapping]
        ax.scatter(
            subset["H_noise"],
            subset["H_effect"],
            s=58,
            color=UNIFIED_COLORS[mapping],
            marker=UNIFIED_MARKERS[mapping],
            edgecolor="white",
        )
    ax.set_xlabel("Conditional noise H(Y|X) (bit)")
    ax.set_ylabel("Effect diversity H(Y) (bit)")
    ax.set_title("Mechanistic trade-off")
    ax.grid(True)
    add_panel_label(ax, "F")

    handles = [
        Line2D(
            [0],
            [0],
            marker=UNIFIED_MARKERS[mapping],
            color="none",
            markerfacecolor=UNIFIED_COLORS[mapping],
            label=mapping,
        )
        for mapping in MAPPINGS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.2)
    savefig(fig, path)


