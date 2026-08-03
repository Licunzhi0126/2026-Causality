from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from ..style import (
    BLUE,
    CYAN,
    DARK,
    GOLD,
    MUTED,
    NAVY,
    RED,
    add_panel_label,
    bar_labels,
    grouped_bars,
    pair_order,
    savefig,
    set_publication_style,
)


def plot_ei_overview(metrics: pd.DataFrame, decomposition: pd.DataFrame, path: Path) -> None:
    """Figure 1: EI gain and its information-theoretic decomposition."""
    set_publication_style()
    order = pair_order(metrics)
    selected = metrics.set_index("time_pair").loc[order].reset_index()
    pivot = decomposition.pivot(
        index="time_pair",
        columns="space",
        values=["H_effect", "H_noise", "EI", "determinism", "degeneracy"],
    ).loc[order]
    x_values = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 1 | Causal emergence and EI mechanism decomposition",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )

    ax = axes[0, 0]
    grouped_bars(
        ax,
        order,
        [
            ("K150 EI", selected["EI_lower"].to_numpy(), BLUE),
            ("K40 EI", selected["EI_upper"].to_numpy(), RED),
        ],
        "Effective information (bit)",
        "Micro-to-macro EI comparison",
        annotate=True,
    )
    ax.legend(loc="upper center", ncol=2)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    bars = ax.bar(
        x_values,
        selected["EI_gain"],
        color=[RED if value >= 0 else BLUE for value in selected["EI_gain"]],
    )
    bar_labels(ax, bars)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Delta EI (bit)")
    ax.set_title("Causal-emergence gain")
    ax.grid(axis="y")
    ax.text(
        0.02,
        0.96,
        f"Positive pairs: {(selected['EI_gain'] > 0).sum()}/{len(selected)}",
        transform=ax.transAxes,
        va="top",
        color=NAVY,
        fontsize=8,
    )
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    ax.plot(x_values, pivot[("H_effect", "lower")], marker="o", color=BLUE, lw=2, label="K150")
    ax.plot(x_values, pivot[("H_effect", "upper")], marker="s", color=RED, lw=2, label="K40")
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Effect entropy H(Y) (bit)")
    ax.set_title("Diversity of reachable effects")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    ax.plot(x_values, pivot[("H_noise", "lower")], marker="o", color=BLUE, lw=2, label="K150")
    ax.plot(x_values, pivot[("H_noise", "upper")], marker="s", color=RED, lw=2, label="K40")
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Conditional entropy H(Y|X) (bit)")
    ax.set_title("Transition noise after intervention")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    noise_reduction = (
        pivot[("H_noise", "lower")].to_numpy() - pivot[("H_noise", "upper")].to_numpy()
    )
    effect_change = (
        pivot[("H_effect", "upper")].to_numpy() - pivot[("H_effect", "lower")].to_numpy()
    )
    width = 0.31
    ax.bar(x_values - width / 2, noise_reduction, width=width, color=CYAN, label="Noise reduction")
    ax.bar(
        x_values + width / 2,
        effect_change,
        width=width,
        color=GOLD,
        label="Effect-diversity change",
    )
    ax.plot(x_values, selected["EI_gain"], marker="o", color=NAVY, lw=1.8, label="Observed Delta EI")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Contribution to Delta EI (bit)")
    ax.set_title("Why EI increases")
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    colors = mpl.colormaps["tab10"](np.linspace(0, 1, len(order)))
    degeneracy_scale = 1e9
    for index, time_pair in enumerate(order):
        subset = decomposition[decomposition["time_pair"] == time_pair].set_index("space")
        lower = subset.loc["lower"]
        upper = subset.loc["upper"]
        arrow = FancyArrowPatch(
            (lower["degeneracy"] * degeneracy_scale, lower["determinism"]),
            (upper["degeneracy"] * degeneracy_scale, upper["determinism"]),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.4,
            color=colors[index],
            alpha=0.9,
        )
        ax.add_patch(arrow)
        ax.scatter(
            lower["degeneracy"] * degeneracy_scale,
            lower["determinism"],
            s=36,
            marker="o",
            color=colors[index],
            edgecolor="white",
            zorder=3,
        )
        ax.scatter(
            upper["degeneracy"] * degeneracy_scale,
            upper["determinism"],
            s=48,
            marker="s",
            color=colors[index],
            edgecolor="white",
            zorder=3,
            label=time_pair,
        )
    max_degeneracy = max(float(decomposition["degeneracy"].max()) * degeneracy_scale, 1e-6)
    ax.set_xlim(-0.05 * max_degeneracy, 1.25 * max_degeneracy)
    ax.set_xlabel("Degeneracy (10^-9 bit)")
    ax.set_ylabel("Determinism (bit)")
    ax.set_title("Determinism-degeneracy displacement")
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Time pair",
        fontsize=6.7,
        title_fontsize=7.2,
    )
    ax.text(0.03, 0.04, "Circle: K150   Square: K40", transform=ax.transAxes, fontsize=7.2, color=MUTED)
    add_panel_label(ax, "F")
    savefig(fig, path)
