from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import MUTED, NAVY, add_panel_label, savefig, set_publication_style


def _unified_legend(ax) -> None:
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
    ax.legend(handles=handles, fontsize=6.4, ncol=2)


def plot_dynamical_closure_three_panels(
    closure: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Canonical full-run dynamical-closure figure."""

    set_publication_style()
    data = closure.merge(
        metrics[["mapping", "time_pair", "delta_EI"]],
        on=["mapping", "time_pair"],
        how="left",
    )
    pairs = list(dict.fromkeys(data["time_pair"].astype(str)))
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)

    ax = axes[0]
    for mapping in MAPPINGS:
        subset = data[data["mapping"] == mapping]
        ax.scatter(
            subset["delta_EI"],
            subset["closure_quality"],
            color=UNIFIED_COLORS[mapping],
            marker=UNIFIED_MARKERS[mapping],
            s=76,
            edgecolor="white",
            label=mapping,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                str(row["time_pair"]).split("->")[0],
                (row["delta_EI"], row["closure_quality"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.2,
            )
    ax.axvline(0, color=MUTED, ls="--", lw=0.9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("ΔEI vs spot (bit)")
    ax.set_ylabel("ClosureQuality = Iretain / Iavailable")
    ax.set_title("Causal emergence × closure")
    ax.grid(True)
    _unified_legend(ax)
    add_panel_label(ax, "A")

    ax = axes[1]
    position = 0.0
    positions: list[float] = []
    labels: list[str] = []
    for pair in pairs:
        subset = data[data["time_pair"] == pair].set_index("mapping").reindex(MAPPINGS)
        for mapping, row in subset.iterrows():
            available = float(row["I_available_bits"])
            retained = float(row["I_retained_bits"] / available) if available > 1e-12 else np.nan
            leaked = float(row["closure_leakage_bits"] / available) if available > 1e-12 else np.nan
            ax.bar(position, retained, width=0.72, color=UNIFIED_COLORS[mapping])
            ax.bar(
                position,
                leaked,
                width=0.72,
                bottom=retained,
                color="#E5EAEF",
                edgecolor=UNIFIED_COLORS[mapping],
                hatch="//",
            )
            positions.append(position)
            labels.append(mapping.replace("Optimized ", "Opt-"))
            position += 1.0
        position += 0.8
    ax.set_xticks(positions, labels, rotation=58, ha="right", fontsize=5.9)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Fraction of available future information")
    ax.set_title("Future-information budget")
    ax.grid(axis="y")
    add_panel_label(ax, "B")

    ax = axes[2]
    x_values = np.arange(len(pairs))
    width = min(0.78 / len(MAPPINGS), 0.22)
    offsets = (np.arange(len(MAPPINGS)) - (len(MAPPINGS) - 1) / 2) * width
    for offset, mapping in zip(offsets, MAPPINGS):
        subset = data[data["mapping"] == mapping].set_index("time_pair").reindex(pairs)
        ax.bar(
            x_values + offset,
            subset["direct_induced_q_js"],
            width * 0.92,
            color=UNIFIED_COLORS[mapping],
        )
    ax.set_xticks(x_values, [pair.replace("->", "→") for pair in pairs])
    ax.set_ylabel("Mean row-wise JS(Qdirect, Qinduced) (bit)")
    ax.set_title("Macro-dynamics consistency")
    ax.grid(axis="y")
    _unified_legend(ax)
    add_panel_label(ax, "C")
    figure.suptitle("Causal emergence and dynamical sufficiency", color=NAVY, fontsize=16, weight="bold")
    savefig(figure, path)


def plot_cross_representation_consistency(consistency: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    mapping_pairs = list(combinations(MAPPINGS, 2))
    times = list(dict.fromkeys(consistency["time"].astype(str)))
    columns = 5
    rows = int(np.ceil(len(mapping_pairs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(21.0, 4.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for panel, (left, right) in enumerate(mapping_pairs):
        ax = axes.flat[panel]
        subset = consistency[
            (consistency["mapping_a"] == left) & (consistency["mapping_b"] == right)
        ].copy()
        subset["time"] = subset["time"].astype(str)
        subset = subset.set_index("time").reindex(times)
        ax.plot(range(len(times)), subset["soft_NMI"], "-o", color=UNIFIED_COLORS[left], label="soft NMI")
        ax.set_ylim(-0.05, 1.03)
        ax.set_xticks(range(len(times)), times)
        ax.set_ylabel("Agreement")
        ax.set_title(f"{left} vs {right}", fontsize=7.5)
        ax.grid(axis="y")
        if panel == 0:
            ax.legend()
        add_panel_label(ax, chr(ord("A") + panel))
    for extra in axes.flat[len(mapping_pairs):]:
        extra.axis("off")
    figure.suptitle(
        "All pairwise cross-representation comparisons",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


