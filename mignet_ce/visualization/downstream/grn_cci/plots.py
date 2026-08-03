from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..style import (
    BLUE,
    DARK,
    DIVERGING,
    MUTED,
    NAVY,
    RED,
    SEQUENTIAL,
    TIME_COLORS,
    add_panel_label,
    savefig,
    set_publication_style,
)


def plot_mechanism(
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
) -> None:
    """Figure 8: exploratory GRN, CCI and spatial correlates of state EI."""
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)
    fig.suptitle(
        "Figure 8 | GRN, CCI and spatial correlates of state-level EI",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    source_time = merged["source_time"].astype(str)
    scatter_specs = (
        ("grn_concentration", "GRN concentration", "State EI vs GRN concentration", "A"),
        ("grn_top10_share", "Top-10 regulator weight share", "State EI vs dominant-regulator share", "B"),
        ("cci_out_log", "log1p CCI out-strength", "State EI vs outgoing communication", "C"),
    )
    for ax, (column, xlabel, title, panel) in zip(axes[0, :], scatter_specs):
        for time in sorted(source_time.unique(), key=float):
            subset = merged[source_time == time]
            ax.scatter(
                subset[column],
                subset["state_ei"],
                s=20 + 65 * subset["spot_count"] / max(merged["spot_count"].max(), 1),
                color=TIME_COLORS.get(time, MUTED),
                alpha=0.7,
                edgecolor="white",
                linewidth=0.25,
                label=time,
            )
        correlation = merged[[column, "state_ei"]].corr().iloc[0, 1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("State-level EI (bit)")
        ax.set_title(f"{title}\nPearson r = {correlation:.2f}")
        ax.grid(True)
        if panel == "A":
            ax.legend(title="Source time", loc="best", ncol=2)
        add_panel_label(ax, panel)

    ax = axes[1, 0]
    scatter = ax.scatter(
        merged["cci_out_entropy_norm"],
        merged["transition_entropy"],
        c=merged["grn_concentration"],
        cmap=SEQUENTIAL,
        s=35,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    correlation = merged[["cci_out_entropy_norm", "transition_entropy"]].corr().iloc[0, 1]
    ax.set_xlabel("Normalized CCI output entropy")
    ax.set_ylabel("Transition entropy H(Y|X) (bit)")
    ax.set_title(f"Communication dispersion vs future uncertainty\nPearson r = {correlation:.2f}")
    ax.grid(True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("GRN concentration")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    coefficients = coefficients.sort_values("coefficient")
    colors = [BLUE if value < 0 else RED for value in coefficients["coefficient"]]
    ax.barh(coefficients["predictor"], coefficients["coefficient"], color=colors)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("Standardized coefficient")
    r_squared = float(coefficients["model_r2"].iloc[0]) if len(coefficients) else np.nan
    ax.set_title(f"Exploratory multivariable model | R2 = {r_squared:.2f}")
    ax.grid(axis="x")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    predictors = [
        "grn_concentration",
        "grn_top10_share",
        "cci_out_log",
        "cci_in_log",
        "cci_out_entropy_norm",
        "boundary_ratio",
        "moran_i",
        "spot_count",
    ]
    outcomes = ["state_ei", "transition_entropy"]
    heatmap = correlations.pivot(index="outcome", columns="predictor", values="pearson_r").reindex(
        index=outcomes,
        columns=predictors,
    )
    maximum = max(float(np.nanmax(np.abs(heatmap.to_numpy()))), 0.5)
    image = ax.imshow(
        heatmap.to_numpy(),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0, vmin=-maximum, vmax=maximum),
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(predictors)),
        [predictor.replace("_", " ") for predictor in predictors],
        rotation=38,
        ha="right",
    )
    ax.set_yticks(np.arange(len(outcomes)), ["State EI", "Transition entropy"])
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap.iloc[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if abs(value) > 0.34 else DARK,
                )
    ax.set_title("Pearson correlation map")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label("Pearson r")
    add_panel_label(ax, "F")
    savefig(fig, path)
