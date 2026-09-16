"""Read-only figure adapters for formal GRN perturbation result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import BLUE, CYAN, DARK, GOLD, GREEN, NAVY, RED, SEQUENTIAL, add_panel_label, savefig, set_publication_style


OPERATOR_COLORS = {
    "deletion": RED,
    "attenuation": GOLD,
    "addition": GREEN,
    "amplification": CYAN,
    "rewire": BLUE,
}


def _operator_color(name: str) -> str:
    return next((color for key, color in OPERATOR_COLORS.items() if key in name), NAVY)


def _mean_sem(table: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    grouped = table.groupby(keys, as_index=False)[value].agg(["mean", "sem"]).reset_index()
    return grouped


def plot_grn_perturbation(table: pd.DataFrame, path: Path) -> None:
    """Render response, operator, time, propagation, ranking, and heatmap summaries."""

    required = {
        "method", "strength", "time_pair", "delta_CE_HV", "delta_CE_VH",
        "path_delta_relative_l1", "delta_closure_js_HV",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"GRN perturbation table misses columns: {sorted(missing)}")
    set_publication_style()
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    response = _mean_sem(table, ["method", "strength"], "delta_CE_HV")
    for method, subset in response.groupby("method", sort=False):
        subset = subset.sort_values("strength")
        color = _operator_color(str(method))
        ax.plot(subset["strength"], subset["mean"], color=color, lw=1.25, alpha=0.85)
        ax.fill_between(
            subset["strength"].to_numpy(float),
            (subset["mean"] - subset["sem"]).to_numpy(float),
            (subset["mean"] + subset["sem"]).to_numpy(float),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set(xlabel="GRN perturbation strength", ylabel="Δ causal emergence (bit)", title="Strength–response with repeat uncertainty")
    ax.grid(axis="y")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ranked = table.groupby("method")["delta_CE_HV"].mean().sort_values()
    ax.barh(np.arange(len(ranked)), ranked.to_numpy(), color=[_operator_color(str(x)) for x in ranked.index])
    ax.set_yticks(np.arange(len(ranked)), [str(x).replace("_", " ") for x in ranked.index], fontsize=6.2)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set(xlabel="Mean ΔCE (bit)", title="Operator comparison")
    ax.grid(axis="x")
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    time_response = table.groupby(["time_pair", "strength"], as_index=False)["delta_CE_HV"].mean()
    palette = (NAVY, BLUE, GOLD, RED)
    for color, (pair, subset) in zip(palette, time_response.groupby("time_pair", sort=False)):
        ax.plot(subset["strength"], subset["delta_CE_HV"], "-o", ms=2.7, lw=1.6, color=color, label=str(pair).replace("->", "→"))
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set(xlabel="Strength", ylabel="Mean ΔCE (bit)", title="Time-pair comparison")
    ax.grid(axis="y")
    ax.legend()
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    propagation = table.groupby("method", as_index=False)[["delta_CE_HV", "delta_CE_VH"]].mean()
    ax.scatter(propagation["delta_CE_HV"], propagation["delta_CE_VH"], c=[_operator_color(str(x)) for x in propagation["method"]], s=42, edgecolor="white")
    limits = np.asarray([ax.get_xlim(), ax.get_ylim()], dtype=float)
    lower, upper = float(limits.min()), float(limits.max())
    ax.plot([lower, upper], [lower, upper], "--", color=DARK, lw=0.8)
    ax.set(xlabel="Horizontal→vertical ΔCE", ylabel="Vertical→horizontal ΔCE", title="Propagation-order comparison")
    ax.grid(True)
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    effect = table.groupby("method")["path_delta_relative_l1"].mean().sort_values(ascending=False)
    ax.bar(np.arange(len(effect)), effect.to_numpy(), color=[_operator_color(str(x)) for x in effect.index])
    ax.set_xticks(np.arange(len(effect)), [str(x).replace("_", " ") for x in effect.index], rotation=55, ha="right", fontsize=5.9)
    ax.set(ylabel="Relative L1 discrepancy", title="Propagation discrepancy ranking")
    ax.grid(axis="y")
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    heatmap = table.pivot_table(index="method", columns="time_pair", values="delta_closure_js_HV", aggfunc="mean")
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap=SEQUENTIAL)
    ax.set_yticks(np.arange(len(heatmap.index)), [str(x).replace("_", " ") for x in heatmap.index], fontsize=6.0)
    ax.set_xticks(np.arange(len(heatmap.columns)), [str(x).replace("->", "→") for x in heatmap.columns], rotation=25, ha="right")
    ax.set_title("Dynamic-closure response heatmap")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Δ closure JS")
    add_panel_label(ax, "F")

    frontend = ", ".join(sorted(table.get("frontend_method", pd.Series(["unknown"])).astype(str).unique()))
    figure.suptitle(
        f"Frozen GRN perturbations | {frontend}",
        color=NAVY,
        fontsize=15,
        weight="bold",
    )
    savefig(figure, path)
