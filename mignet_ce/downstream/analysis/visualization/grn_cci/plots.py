from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import MUTED, NAVY, add_panel_label, savefig, set_publication_style


def plot_unified_mechanism(mechanism: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    pairs = list(dict.fromkeys(mechanism["time_pair"].astype(str)))
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    specifications = [
        ("grn_concentration", "GRN concentration"),
        ("cci_out_log", "CCI out-strength"),
        ("cci_in_log", "CCI in-strength"),
    ]
    for column, (predictor, label) in enumerate(specifications):
        ax = axes[0, column]
        for mapping in MAPPINGS:
            correlations = []
            for pair in pairs:
                subset = mechanism[
                    (mechanism["mapping"] == mapping) & (mechanism["time_pair"] == pair)
                ]
                correlations.append(
                    spearmanr(subset[predictor], subset["state_ei"], nan_policy="omit").statistic
                    if len(subset) > 2
                    else np.nan
                )
            ax.plot(
                range(len(pairs)),
                correlations,
                marker=UNIFIED_MARKERS[mapping],
                color=UNIFIED_COLORS[mapping],
                lw=1.8,
                label=mapping,
            )
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_xticks(range(len(pairs)), [pair.replace("->", "→") for pair in pairs])
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Spearman ρ with state EI")
        ax.set_title(label)
        ax.grid(axis="y")
        if column == 2:
            ax.legend(fontsize=6.2)
        add_panel_label(ax, chr(ord("A") + column))
    for column, (predictor, label) in enumerate(specifications):
        ax = axes[1, column]
        for mapping in MAPPINGS:
            subset = mechanism[mechanism["mapping"] == mapping]
            ax.scatter(
                subset[predictor],
                subset["state_ei"],
                s=13,
                alpha=0.5,
                color=UNIFIED_COLORS[mapping],
                marker=UNIFIED_MARKERS[mapping],
            )
        ax.set_xlabel(label)
        ax.set_ylabel("State EI (bit)")
        ax.set_title(f"{label} vs state EI")
        ax.grid(True)
        add_panel_label(ax, chr(ord("D") + column))
    figure.suptitle(
        "GRN–CCI mechanism audit on a common spot substrate",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


