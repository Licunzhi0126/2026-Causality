from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


TEAL = "#2E7774"
TEAL_DARK = "#1B3A55"
TEAL_LIGHT = "#E8F3F1"
GREEN = "#1D6A45"
GREEN_BG = "#E6F4EA"
RED = "#B42318"
RED_BG = "#FDECEC"
BLUE_BG = "#DCEEF7"
ROW_BG = "#F4F7FA"
GRID = "#D7E1E7"
TEXT = "#23323D"
MUTED = "#73808A"
HEART = "#C94845"
SPOT = "#CC4747"


def set_asset_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
