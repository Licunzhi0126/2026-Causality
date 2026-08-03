from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import (
    BLUE,
    CYAN,
    DARK,
    GOLD,
    GREEN,
    METHOD_COLORS,
    METHOD_LABELS,
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


def plot_single_step_closure(closure: pd.DataFrame, path: Path) -> None:
    """Figure 3: one-step commutativity test."""
    set_publication_style()
    order = pair_order(closure)
    pivot_error = closure.pivot(index="time_pair", columns="macro_Q", values="relative_frobenius").loc[order]
    pivot_js = closure.pivot(index="time_pair", columns="macro_Q", values="mean_row_js").loc[order]
    pivot_ei = closure.pivot(index="time_pair", columns="macro_Q", values="EI_Q").loc[order]
    methods = [
        method
        for method in ("direct_K40_Pij", "overlap_aggregated", "clipped_least_squares")
        if method in pivot_error.columns
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle("Figure 3 | Single-step macro-dynamics closure", color=NAVY, fontsize=16, weight="bold")

    ax = axes[0, 0]
    grouped_bars(
        ax,
        order,
        [(METHOD_LABELS[method], pivot_error[method].to_numpy(), METHOD_COLORS[method]) for method in methods],
        "Relative Frobenius error",
        "Commutativity error",
        annotate=True,
    )
    ax.legend(loc="upper center")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    grouped_bars(
        ax,
        order,
        [(METHOD_LABELS[method], pivot_js[method].to_numpy(), METHOD_COLORS[method]) for method in methods],
        "Mean row-wise JS divergence (bit)",
        "Distributional mismatch",
        annotate=True,
    )
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    grouped_bars(
        ax,
        order,
        [(METHOD_LABELS[method], pivot_ei[method].to_numpy(), METHOD_COLORS[method]) for method in methods],
        "EI of macro transition Q (bit)",
        "Information retained by Q",
        annotate=True,
    )
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    excess_best = pivot_error["direct_K40_Pij"] - pivot_error["clipped_least_squares"]
    bars = ax.bar(np.arange(len(order)), excess_best, color=RED)
    bar_labels(ax, bars)
    ax.set_xticks(np.arange(len(order)), order, rotation=28, ha="right")
    ax.set_ylabel("Excess relative error")
    ax.set_title("Independent K40 error above best fit")
    ax.grid(axis="y")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    excess_overlap = pivot_error["direct_K40_Pij"] - pivot_error["overlap_aggregated"]
    bars = ax.bar(np.arange(len(order)), excess_overlap, color=GOLD)
    bar_labels(ax, bars)
    ax.set_xticks(np.arange(len(order)), order, rotation=28, ha="right")
    ax.set_ylabel("Excess relative error")
    ax.set_title("Independent K40 error above overlap Q")
    ax.grid(axis="y")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    markers = {"direct_K40_Pij": "s", "overlap_aggregated": "o", "clipped_least_squares": "^"}
    for method in methods:
        subset = closure[closure["macro_Q"] == method]
        ax.scatter(
            subset["relative_frobenius"],
            subset["EI_Q"],
            s=65,
            marker=markers[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            label=METHOD_LABELS[method],
        )
        for _, row in subset.iterrows():
            ax.annotate(
                str(row["time_pair"]),
                (row["relative_frobenius"], row["EI_Q"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.4,
                color=DARK,
            )
    ax.set_xlabel("Relative closure error")
    ax.set_ylabel("EI of Q (bit)")
    ax.set_title("Closure-information trade-off")
    ax.grid(True)
    ax.legend(loc="best")
    add_panel_label(ax, "F")
    savefig(fig, path)


def plot_multistep_closure(closure: pd.DataFrame, path: Path) -> None:
    """Figure 4: composed versus direct long-range transitions."""
    set_publication_style()
    order = closure["comparison"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 4 | Multi-step closure and long-range prediction",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    for ax, metric, ylabel, title in (
        (axes[0, 0], "relative_frobenius", "Relative Frobenius error", "Composed vs direct transition"),
        (axes[0, 1], "mean_row_js", "Mean row-wise JS divergence (bit)", "Row-distribution mismatch"),
    ):
        pivot = closure.pivot(index="comparison", columns="space", values=metric).loc[order]
        grouped_bars(
            ax,
            order,
            [("K150", pivot["lower"].to_numpy(), BLUE), ("K40", pivot["upper"].to_numpy(), RED)],
            ylabel,
            title,
            annotate=True,
        )
        ax.legend(loc="best")
    add_panel_label(axes[0, 0], "A")
    add_panel_label(axes[0, 1], "B")

    for ax, space, scale, panel in (
        (axes[0, 2], "lower", "K150", "C"),
        (axes[1, 0], "upper", "K40", "D"),
    ):
        subset = closure[closure["space"] == space].set_index("comparison").loc[order]
        grouped_bars(
            ax,
            order,
            [
                ("Composed EI", subset["EI_composed"].to_numpy(), CYAN),
                ("Direct EI", subset["EI_direct"].to_numpy(), NAVY),
            ],
            "Effective information (bit)",
            f"{scale}: information after composition",
            annotate=True,
        )
        ax.legend(loc="best")
        add_panel_label(ax, panel)

    ax = axes[1, 1]
    pivot = closure.pivot(index="comparison", columns="space", values="EI_difference").loc[order]
    grouped_bars(
        ax,
        order,
        [("K150", pivot["lower"].to_numpy(), BLUE), ("K40", pivot["upper"].to_numpy(), RED)],
        "Composed EI - direct EI (bit)",
        "Long-range information loss",
        annotate=True,
    )
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.legend(loc="best")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    for space, color, marker, label in (("lower", BLUE, "o", "K150"), ("upper", RED, "s", "K40")):
        subset = closure[closure["space"] == space]
        ax.scatter(
            subset["relative_frobenius"],
            -subset["EI_difference"],
            s=70,
            marker=marker,
            color=color,
            edgecolor="white",
            label=label,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                str(row["comparison"]),
                (row["relative_frobenius"], -row["EI_difference"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.4,
            )
    ax.set_xlabel("Relative closure error")
    ax.set_ylabel("EI loss after composition (bit)")
    ax.set_title("Dynamics error vs information loss")
    ax.grid(True)
    ax.legend(loc="best")
    add_panel_label(ax, "F")
    savefig(fig, path)


def plot_multiscale(consistency: pd.DataFrame, purity: pd.DataFrame, path: Path) -> None:
    """Figure 6: direct versus hierarchical multiscale consistency."""
    set_publication_style()
    order = pair_order(consistency)
    selected = consistency.set_index("time_pair").loc[order].reset_index()
    times = sorted(purity["time"].astype(str).unique(), key=float)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 6 | K150-to-K40 multiscale consistency",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    x_values = np.arange(len(order))

    ax = axes[0, 0]
    bars = ax.bar(x_values, selected["relative_frobenius"], color=BLUE)
    bar_labels(ax, bars)
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Direct K40 vs route through K150")
    ax.grid(axis="y")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    bars = ax.bar(x_values, selected["mean_row_js"], color=CYAN)
    bar_labels(ax, bars)
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("Mean row-wise JS divergence (bit)")
    ax.set_title("Row-level transition mismatch")
    ax.grid(axis="y")
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    grouped_bars(
        ax,
        order,
        [
            ("EI via K150", selected["EI_via_K150"].to_numpy(), GOLD),
            ("Direct K40 EI", selected["EI_direct_K40"].to_numpy(), NAVY),
        ],
        "Effective information (bit)",
        "Information along two coarse-graining routes",
        annotate=True,
    )
    ax.legend(loc="best")
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    bars = ax.bar(
        x_values,
        selected["EI_path_difference"],
        color=[BLUE if value < 0 else RED for value in selected["EI_path_difference"]],
    )
    bar_labels(ax, bars)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x_values, order, rotation=28, ha="right")
    ax.set_ylabel("EI via K150 - direct K40 EI (bit)")
    ax.set_title("Path-dependent EI difference")
    ax.grid(axis="y")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    distributions = [purity.loc[purity["time"].astype(str) == time, "purity"].to_numpy() for time in times]
    boxplot = ax.boxplot(distributions, tick_labels=times, patch_artist=True, widths=0.65, showfliers=False)
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#DCEAF2")
        patch.set_edgecolor(BLUE)
    for median in boxplot["medians"]:
        median.set_color(RED)
        median.set_linewidth(1.6)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Embryonic time")
    ax.set_ylabel("K150-to-K40 purity")
    ax.set_title("How uniquely each K150 state maps to K40")
    ax.grid(axis="y")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    summary = (
        purity.groupby(purity["time"].astype(str))
        .agg(
            median_purity=("purity", "median"),
            fraction_purity_08=("purity", lambda values: float((values >= 0.8).mean())),
            median_entropy=("mapping_entropy_norm", "median"),
        )
        .reindex(times)
    )
    ax.bar(
        np.arange(len(times)),
        summary["fraction_purity_08"],
        color=GREEN,
        label="Fraction with purity >= 0.8",
    )
    ax.plot(np.arange(len(times)), summary["median_purity"], color=RED, marker="o", lw=2, label="Median purity")
    ax.plot(
        np.arange(len(times)),
        1 - summary["median_entropy"],
        color=BLUE,
        marker="s",
        lw=2,
        label="1 - median mapping entropy",
    )
    ax.set_xticks(np.arange(len(times)), times)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Embryonic time")
    ax.set_ylabel("Hierarchy clarity")
    ax.set_title("Time-resolved hierarchy quality")
    ax.grid(axis="y")
    ax.legend(loc="lower left")
    add_panel_label(ax, "F")
    savefig(fig, path)
