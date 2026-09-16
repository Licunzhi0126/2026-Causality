from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


NAVY = "#183F63"
BLUE = "#2F74B8"
RED = "#D45959"
CYAN = "#57C7D8"
GOLD = "#E6A044"
GREEN = "#4D9A73"
DARK = "#263746"
MUTED = "#71808C"
GRID = "#D8E0E5"

SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list("ce_seq", ["#EAF4F7", CYAN, BLUE, NAVY])


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
