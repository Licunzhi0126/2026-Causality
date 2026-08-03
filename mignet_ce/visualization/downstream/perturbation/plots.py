from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import (
    MUTED,
    NAVY,
    TARGET_COLORS,
    TARGET_LABELS,
    add_panel_label,
    bar_labels,
    savefig,
    set_publication_style,
)


def plot_perturbation(curves: pd.DataFrame, path: Path) -> None:
    """Figure 10: Pij row-homogenization dose response."""
    set_publication_style()
    pairs = list(dict.fromkeys(curves["time_pair"].astype(str).tolist()))[:3]
    if len(pairs) != 3:
        raise ValueError(f"Expected three adjacent time pairs, got {pairs}")
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 10 | Pij row-homogenization perturbation dose-response",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    target_order = ["high_state_ei", "high_cci_out", "high_grn_concentration", "matched_random"]
    for index, time_pair in enumerate(pairs):
        ax = axes[0, index]
        subset = curves[curves["time_pair"] == time_pair]
        for target in target_order:
            current = subset[subset["target"] == target].sort_values("dose")
            ax.plot(
                current["dose"],
                current["ei_drop_mean"],
                marker="o",
                ms=3.3,
                lw=1.9,
                color=TARGET_COLORS[target],
                label=TARGET_LABELS[target],
            )
            if target == "matched_random":
                ax.fill_between(
                    current["dose"],
                    current["ei_drop_low"],
                    current["ei_drop_high"],
                    color=TARGET_COLORS[target],
                    alpha=0.16,
                )
        ax.set_xlabel("Perturbation dose")
        ax.set_ylabel("EI drop (bit)")
        ax.set_title(time_pair)
        ax.grid(True)
        if index == 2:
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        add_panel_label(ax, chr(ord("A") + index))

    full_dose = curves[np.isclose(curves["dose"], 1.0)].copy()
    ax = axes[1, 0]
    summary = full_dose.groupby("target")["ei_drop_mean"].agg(["mean", "std"]).reindex(target_order)
    bars = ax.bar(
        np.arange(len(target_order)),
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
        color=[TARGET_COLORS[target] for target in target_order],
    )
    bar_labels(ax, bars)
    ax.set_xticks(
        np.arange(len(target_order)),
        [TARGET_LABELS[target] for target in target_order],
        rotation=28,
        ha="right",
    )
    ax.set_ylabel("Mean full-dose EI drop (bit)")
    ax.set_title("Average sensitivity across time pairs")
    ax.grid(axis="y")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    random_full = full_dose[full_dose["target"] == "matched_random"].set_index("time_pair")["ei_drop_mean"]
    width = 0.24
    x_values = np.arange(len(pairs))
    targeted = ["high_state_ei", "high_cci_out", "high_grn_concentration"]
    for index, target in enumerate(targeted):
        values = (
            full_dose[full_dose["target"] == target].set_index("time_pair")["ei_drop_mean"].reindex(pairs)
            - random_full.reindex(pairs)
        )
        ax.bar(
            x_values + (index - 1) * width,
            values,
            width=width,
            color=TARGET_COLORS[target],
            label=TARGET_LABELS[target],
        )
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x_values, pairs, rotation=25, ha="right")
    ax.set_ylabel("Targeted drop - matched-random drop (bit)")
    ax.set_title("Perturbation specificity at full dose")
    ax.grid(axis="y")
    ax.legend(loc="best")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    auc_rows = []
    for (time_pair, target), group in curves.groupby(["time_pair", "target"]):
        group = group.sort_values("dose")
        auc_rows.append(
            {
                "time_pair": time_pair,
                "target": target,
                "auc": float(np.trapz(group["ei_drop_mean"].to_numpy(), group["dose"].to_numpy())),
            }
        )
    auc = pd.DataFrame(auc_rows)
    width = 0.19
    for index, target in enumerate(target_order):
        values = auc[auc["target"] == target].set_index("time_pair")["auc"].reindex(pairs)
        ax.bar(
            x_values + (index - 1.5) * width,
            values,
            width=width,
            color=TARGET_COLORS[target],
            label=TARGET_LABELS[target],
        )
    ax.set_xticks(x_values, pairs, rotation=25, ha="right")
    ax.set_ylabel("Area under EI-drop curve")
    ax.set_title("Dose-integrated perturbation effect")
    ax.grid(axis="y")
    add_panel_label(ax, "F")
    savefig(fig, path)
