from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LINE_ORDER = (
    "spot:seuratK40",
    "spot:seuratK150",
    "seuratK150:seuratK40",
)


def plot_alpha_sensitivity(
    frame: pd.DataFrame,
    path: Path,
    *,
    time_pair: str = "12.5->13.5",
    canonical_alpha: float = 0.01,
) -> None:
    subset = frame[frame["time_pair"].astype(str) == str(time_pair)].copy()
    if subset.empty:
        raise ValueError(f"No sensitivity rows found for {time_pair}")
    observed = set(subset["hierarchy_pair"].astype(str))
    expected = set(LINE_ORDER)
    if observed != expected:
        raise ValueError(
            f"Sensitivity figure requires exactly {sorted(expected)}; found {sorted(observed)}."
        )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for label in LINE_ORDER:
        current = subset[subset["hierarchy_pair"] == label].sort_values("alpha")
        ax.plot(current["alpha"], current["delta_EI_bits"], marker="o", lw=1.8, ms=4.0, label=label)
    ax.axhline(0.0, lw=0.9)
    ax.axvline(float(canonical_alpha), ls="--", lw=0.9)
    ax.set_xlabel("CCI mixing weight α")
    ax.set_ylabel("ΔEI (bits)")
    ax.set_title(f"GRN–CCI weight sensitivity: {time_pair.replace('->', '→')}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
