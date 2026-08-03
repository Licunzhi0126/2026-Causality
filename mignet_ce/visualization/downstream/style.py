from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


NAVY = "#183F63"
BLUE = "#2F74B8"
RED = "#D45959"
CYAN = "#57C7D8"
GOLD = "#E6A044"
GREEN = "#4D9A73"
PURPLE = "#8064A2"
LIGHT = "#F4F8FA"
DARK = "#263746"
MUTED = "#71808C"
GRID = "#D8E0E5"

DIVERGING = mpl.colors.LinearSegmentedColormap.from_list("ce_div", [BLUE, "#F7F7F7", RED])
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list("ce_seq", ["#EAF4F7", CYAN, BLUE, NAVY])
TIME_COLORS = {"11.5": NAVY, "12.5": BLUE, "13.5": GOLD, "14.5": RED}
METHOD_COLORS = {
    "direct_K40_Pij": RED,
    "overlap_aggregated": CYAN,
    "clipped_least_squares": BLUE,
}
METHOD_LABELS = {
    "direct_K40_Pij": "Independent K40 Pij",
    "overlap_aggregated": "Overlap-aggregated Q",
    "clipped_least_squares": "Best-fit lower bound",
}
TARGET_COLORS = {
    "high_state_ei": RED,
    "high_cci_out": BLUE,
    "high_grn_concentration": GREEN,
    "matched_random": "#8E99A2",
}
TARGET_LABELS = {
    "high_state_ei": "High state EI",
    "high_cci_out": "High CCI out-strength",
    "high_grn_concentration": "High GRN concentration",
    "matched_random": "Matched random",
}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A7B2BA",
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.68,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13.5,
        weight="bold",
        color=NAVY,
        va="top",
        ha="left",
    )


def pair_order(frame) -> list[str]:
    if "lag" in frame.columns:
        return frame.sort_values(["lag", "time_pair"])["time_pair"].drop_duplicates().tolist()
    return sorted(
        frame["time_pair"].astype(str).unique(),
        key=lambda value: (float(value.split("->")[1]) - float(value.split("->")[0]), value),
    )


def bar_labels(ax: plt.Axes, bars, fmt: str = ".2f", pad: float = 2.5, fontsize: float = 6.6) -> None:
    labels = [format(bar.get_height(), fmt) if np.isfinite(bar.get_height()) else "" for bar in bars]
    ax.bar_label(bars, labels=labels, padding=pad, fontsize=fontsize, color=DARK)


def grouped_bars(
    ax: plt.Axes,
    categories: list[str],
    series: list[tuple[str, np.ndarray, str]],
    ylabel: str,
    title: str,
    rotate: int = 28,
    annotate: bool = False,
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = min(0.76 / max(len(series), 1), 0.32)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (label, values, color) in zip(offsets, series):
        bars = ax.bar(x + offset, values, width=width * 0.92, color=color, label=label)
        if annotate:
            bar_labels(ax, bars)
    ax.set_xticks(x, categories, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y")
