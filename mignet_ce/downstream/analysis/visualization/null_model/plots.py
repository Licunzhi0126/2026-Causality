from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ...mappings import COLORS as UNIFIED_COLORS, MARKERS as UNIFIED_MARKERS, MAPPINGS
from ..style import NAVY, add_panel_label, savefig, set_publication_style


def plot_unified_random_null(null_table: pd.DataFrame, path: Path) -> None:
    set_publication_style()
    pairs = list(dict.fromkeys(null_table["time_pair"].astype(str)))
    figure, axes = plt.subplots(2, len(pairs), figsize=(15.6, 9.2), constrained_layout=True)
    specifications = [
        ("EI", "Effective information (bit)", "EI against matched partitions"),
        ("closure_leakage_bits", "Closure leakage (bit)", "Closure against matched partitions"),
    ]
    for row, (metric, ylabel, title) in enumerate(specifications):
        for column, pair in enumerate(pairs):
            ax = axes[row, column]
            for mapping_index, mapping in enumerate(MAPPINGS):
                subset = null_table[
                    (null_table["mapping"] == mapping) & (null_table["time_pair"] == pair)
                ]
                null_values = subset.loc[subset["kind"] == "matched_random", metric].dropna().to_numpy(dtype=float)
                observed = subset.loc[subset["kind"] == "observed", metric]
                if len(null_values):
                    violin = ax.violinplot([null_values], positions=[mapping_index], widths=0.66, showextrema=False)
                    for body in violin["bodies"]:
                        body.set_facecolor(UNIFIED_COLORS[mapping])
                        body.set_edgecolor(UNIFIED_COLORS[mapping])
                        body.set_alpha(0.24)
                if len(observed):
                    ax.scatter(
                        mapping_index,
                        float(observed.iloc[0]),
                        s=54,
                        color=UNIFIED_COLORS[mapping],
                        marker=UNIFIED_MARKERS[mapping],
                        edgecolor="white",
                        zorder=4,
                    )
            ax.set_xticks(range(len(MAPPINGS)), [mapping.replace("Optimized ", "Opt-") for mapping in MAPPINGS], rotation=22, ha="right", fontsize=6.0)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} · {pair.replace('->', '→')}", fontsize=8)
            ax.grid(axis="y")
            add_panel_label(ax, chr(ord("A") + row * len(pairs) + column))
    figure.suptitle(
        "Matched random-partition null across five macro representations",
        color=NAVY,
        fontsize=16,
        weight="bold",
    )
    savefig(figure, path)


