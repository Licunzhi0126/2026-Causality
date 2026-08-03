from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import (
    BLUE,
    GOLD,
    GREEN,
    MUTED,
    NAVY,
    RED,
    add_panel_label,
    bar_labels,
    savefig,
    set_publication_style,
)


def plot_random_null(null_table: pd.DataFrame, path: Path) -> None:
    """Figure 5: matched random coarse-graining null model."""
    set_publication_style()
    pairs = list(dict.fromkeys(null_table["time_pair"].astype(str).tolist()))[:3]
    if len(pairs) != 3:
        raise ValueError(f"Expected three adjacent time pairs, got {pairs}")
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 5 | Matched random coarse-graining null model",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    summary_rows: list[dict[str, float | str]] = []
    for index, time_pair in enumerate(pairs):
        ax = axes[0, index]
        subset = null_table[null_table["time_pair"] == time_pair]
        null_values = subset.loc[subset["kind"] == "matched_random", "EI"].to_numpy()
        observed = float(subset.loc[subset["kind"] == "observed", "EI"].iloc[0])
        mean = float(np.mean(null_values))
        standard_deviation = float(np.std(null_values, ddof=1))
        p_value = float((1 + np.sum(null_values >= observed)) / (1 + len(null_values)))
        z_score = float((observed - mean) / max(standard_deviation, 1e-12))
        q025, q975 = np.quantile(null_values, [0.025, 0.975])
        summary_rows.append(
            {
                "time_pair": time_pair,
                "observed": observed,
                "mean": mean,
                "q025": float(q025),
                "q975": float(q975),
                "z": z_score,
                "p": p_value,
            }
        )
        q_low, q_high = np.quantile(null_values, [0.005, 0.995])
        span = max(float(q_high - q_low), 1e-6)
        ax.hist(null_values, bins=28, color="#C9DCE8", edgecolor="white")
        ax.axvline(mean, color=NAVY, ls="--", lw=1.4, label=f"Null mean = {mean:.3f}")
        ax.set_xlim(q_low - 0.08 * span, q_high + 0.08 * span)
        direction = "above" if observed > q_high else "below" if observed < q_low else "inside"
        if direction == "inside":
            annotation = f"Observed = {observed:.3f}\n(inside null range)"
            xy = (observed, ax.get_ylim()[1] * 0.72)
            xycoords = "data"
        else:
            annotation = f"Observed = {observed:.3f}\n(outside null range)"
            xy = (0.985 if direction == "above" else 0.015, 0.82)
            xycoords = "axes fraction"
        ax.annotate(
            annotation,
            xy=xy,
            xycoords=xycoords,
            xytext=(0.68 if direction != "below" else 0.32, 0.82),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.8},
            color=RED,
            fontsize=7.2,
            ha="right" if direction != "below" else "left",
            va="center",
        )
        ax.set_xlabel("Matched-random aggregated EI (bit)")
        ax.set_ylabel("Random assignments")
        ax.set_title(f"{time_pair} | z = {z_score:.1f}, p = {p_value:.4f}")
        ax.grid(axis="y")
        ax.legend(loc="upper right")
        add_panel_label(ax, chr(ord("A") + index))

    summary = pd.DataFrame(summary_rows)
    x_values = np.arange(len(summary))
    ax = axes[1, 0]
    error = np.vstack([summary["mean"] - summary["q025"], summary["q975"] - summary["mean"]])
    ax.errorbar(
        x_values,
        summary["mean"],
        yerr=error,
        fmt="o",
        color=BLUE,
        capsize=4,
        label="Null mean and 95% interval",
    )
    ax.scatter(x_values, summary["observed"], marker="D", s=60, color=RED, label="Observed")
    ax.set_xticks(x_values, summary["time_pair"], rotation=25, ha="right")
    ax.set_ylabel("Aggregated EI (bit)")
    ax.set_title("Observed EI relative to the null interval")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    bars = ax.bar(x_values, summary["z"], color=GOLD)
    bar_labels(ax, bars, fmt=".1f")
    ax.axhline(1.96, color=MUTED, ls="--", lw=1, label="z = 1.96")
    ax.set_xticks(x_values, summary["time_pair"], rotation=25, ha="right")
    ax.set_ylabel("Standardized separation (z)")
    ax.set_title("Effect size relative to null spread")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    significance = -np.log10(summary["p"])
    bars = ax.bar(x_values, significance, color=GREEN)
    bar_labels(ax, bars, fmt=".2f")
    ax.axhline(-np.log10(0.05), color=MUTED, ls="--", lw=1, label="p = 0.05")
    ax.set_xticks(x_values, summary["time_pair"], rotation=25, ha="right")
    ax.set_ylabel("-log10 empirical p")
    ax.set_title("Empirical significance")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "F")
    savefig(fig, path)
