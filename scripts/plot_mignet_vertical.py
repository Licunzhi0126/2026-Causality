#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mignet_ce.config import PAIR_PRESETS

PAIR_ORDER_BY_FAMILY: Dict[str, Dict[str, List[str]]] = {
    "legacy_mixed": {
        "adjacent": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["legacy_mixed_adjacent"]],
        "all": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["legacy_mixed_adjacent"]],
        "cross_level": [],
    },
    "louvain": {
        "adjacent": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["louvain_adjacent"]],
        "all": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["louvain_all"]],
        "cross_level": [],
    },
    "seurat": {
        "adjacent": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["seurat_adjacent"]],
        "all": [f"{pair.lower_layer}->{pair.upper_layer}" for pair in PAIR_PRESETS["seurat_all"]],
        "cross_level": [],
    },
}
for _family in ("louvain", "seurat"):
    _adjacent = set(PAIR_ORDER_BY_FAMILY[_family]["adjacent"])
    PAIR_ORDER_BY_FAMILY[_family]["cross_level"] = [
        key for key in PAIR_ORDER_BY_FAMILY[_family]["all"] if key not in _adjacent
    ]

LEVEL_LABELS: Dict[str, str] = {
    "spot": "Spot",
    "seurat_less_than5": "Seurat <5",
    "seurat_k150": "Seurat k150",
    "seurat_k40": "Seurat k40",
    "louvain_k40": "Louvain k40",
    "louvain_k150": "Louvain k150",
    "louvain_less_than5": "Louvain <5",
}
ORGAN_ORDER = ["heart", "brain", "lung"]
ORGAN_LABELS = {"heart": "Heart", "brain": "Brain", "lung": "Lung"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pair_label(lower: str, upper: str) -> str:
    return f"{LEVEL_LABELS.get(lower, lower)} -> {LEVEL_LABELS.get(upper, upper)}"


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if "EI_local" in df.columns and "EI_lower" not in df.columns:
        rename["EI_local"] = "EI_lower"
    if "EI_global" in df.columns and "EI_upper" not in df.columns:
        rename["EI_global"] = "EI_upper"
    df = df.rename(columns=rename)
    required = {"organ", "lower_layer", "upper_layer", "time_pair", "EI_lower", "EI_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required metric columns: {sorted(missing)}")
    df = df.copy()
    df["lower_layer"] = df["lower_layer"].astype(str)
    df["upper_layer"] = df["upper_layer"].astype(str)
    df["organ"] = df["organ"].astype(str)
    df["pair_key"] = df["lower_layer"] + "->" + df["upper_layer"]
    df["pair_label"] = [_pair_label(l, u) for l, u in zip(df["lower_layer"], df["upper_layer"])]
    df["EI_delta"] = pd.to_numeric(df["EI_upper"], errors="coerce") - pd.to_numeric(df["EI_lower"], errors="coerce")
    df["EI_lower"] = pd.to_numeric(df["EI_lower"], errors="coerce")
    df["EI_upper"] = pd.to_numeric(df["EI_upper"], errors="coerce")
    return df


def _ordered_values(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    values = list(dict.fromkeys(map(str, values)))
    preferred_present = [value for value in preferred if value in values]
    extras = sorted([value for value in values if value not in preferred_present])
    return preferred_present + extras


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(map(str, values)))


def _pair_order_for(family: str, pair_category: str) -> List[str]:
    if family == "all":
        ordered: List[str] = []
        for name in ("legacy_mixed", "louvain", "seurat"):
            ordered.extend(PAIR_ORDER_BY_FAMILY[name][pair_category])
        return _dedupe(ordered)
    return list(PAIR_ORDER_BY_FAMILY[family][pair_category])


def _filter_metrics(df: pd.DataFrame, family: str, pair_category: str) -> pd.DataFrame:
    if family == "all" and pair_category == "all":
        return df
    pair_order = set(_pair_order_for(family, pair_category))
    if not pair_order:
        return df.iloc[0:0].copy()
    return df[df["pair_key"].isin(pair_order)].copy()


def _ordered_pair_keys(work: pd.DataFrame, family: str, pair_category: str) -> List[str]:
    preferred = _pair_order_for(family, pair_category)
    pair_order = [key for key in preferred if key in set(work["pair_key"])]
    pair_order += sorted([key for key in work["pair_key"].unique() if key not in pair_order])
    return pair_order


def plot_delta_heatmap(
    df: pd.DataFrame,
    output: Path,
    time_pair: str | None = None,
    family: str = "all",
    pair_category: str = "all",
) -> None:
    work = df.copy()
    if time_pair is not None:
        work = work[work["time_pair"].astype(str) == str(time_pair)]
    work = _filter_metrics(work, family=family, pair_category=pair_category)
    if work.empty:
        raise ValueError("No rows available for heatmap after filtering.")

    pair_order = _ordered_pair_keys(work, family=family, pair_category=pair_category)
    organ_order = _ordered_values(work["organ"], ORGAN_ORDER)

    pivot = (
        work.groupby(["organ", "pair_key"], as_index=False)["EI_delta"]
        .mean()
        .pivot(index="organ", columns="pair_key", values="EI_delta")
        .reindex(index=organ_order, columns=pair_order)
    )
    pivot.index = [ORGAN_LABELS.get(idx, idx.title()) for idx in pivot.index]
    pivot.columns = [_pair_label(*col.split("->", 1)) for col in pivot.columns]

    fig_width = max(8.2, 2.9 * max(1, pivot.shape[1]))
    fig_height = max(4.8, 1.05 * max(1, pivot.shape[0]) + 2.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    sns.set_theme(style="white", font_scale=0.95)
    vmax = np.nanmax(np.abs(pivot.to_numpy())) if pivot.size else 1.0
    vmax = max(float(vmax), 1e-9)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.8,
        linecolor="#e8edf2",
        cbar_kws={"label": "EI gain (upper - lower)", "shrink": 0.82},
    )
    title_suffix = f" ({time_pair})" if time_pair else ""
    family_suffix = f" {family.replace('_', ' ').title()} {pair_category.replace('_', '-')}"
    ax.set_title(f"Vertical EI Gain Heatmap -{family_suffix}{title_suffix}", fontsize=14, weight="semibold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_adjacent_pairs(
    df: pd.DataFrame,
    output: Path,
    time_pair: str | None = None,
    family: str = "all",
    pair_category: str = "all",
) -> None:
    work = df.copy()
    if time_pair is not None:
        work = work[work["time_pair"].astype(str) == str(time_pair)]
    work = _filter_metrics(work, family=family, pair_category=pair_category)
    if work.empty:
        raise ValueError("No rows available for adjacent-pair plot after filtering.")

    work = work.groupby(["organ", "pair_key", "pair_label"], as_index=False)[["EI_lower", "EI_upper"]].mean()
    pair_order = _ordered_pair_keys(work, family=family, pair_category=pair_category)
    organ_order = _ordered_values(work["organ"], ORGAN_ORDER)
    pair_to_y = {pair: idx for idx, pair in enumerate(pair_order)}

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig_height = max(5.2, 1.1 * len(pair_order) + 1.8)
    fig_width = max(9.8, 3.7 * len(organ_order))
    fig, axes = plt.subplots(1, len(organ_order), figsize=(fig_width, fig_height), dpi=180, sharex=True, sharey=True)
    if len(organ_order) == 1:
        axes = [axes]

    lower_color = "#2f6f9f"
    upper_color = "#c75c2c"
    line_color = "#98a5b3"
    x_values = work[["EI_lower", "EI_upper"]].to_numpy().ravel()
    x_min = float(np.nanmin(x_values)) if x_values.size else 0.0
    x_max = float(np.nanmax(x_values)) if x_values.size else 1.0
    pad = max((x_max - x_min) * 0.12, 1e-3)

    for ax, organ in zip(axes, organ_order):
        sub = work[work["organ"] == organ]
        ax.set_title(ORGAN_LABELS.get(organ, organ.title()), fontsize=12, weight="semibold")
        for row in sub.itertuples(index=False):
            y = pair_to_y[row.pair_key]
            ax.plot([row.EI_lower, row.EI_upper], [y, y], color=line_color, lw=2.1, zorder=1)
            ax.scatter(row.EI_lower, y, s=58, color=lower_color, edgecolor="white", lw=0.8, zorder=3, label="Lower EI")
            ax.scatter(row.EI_upper, y, s=58, color=upper_color, edgecolor="white", lw=0.8, zorder=3, label="Upper EI")
            delta = row.EI_upper - row.EI_lower
            ax.text(
                max(row.EI_lower, row.EI_upper) + pad * 0.18,
                y,
                f"{delta:+.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#46515d",
            )
        ax.set_xlim(x_min - pad, x_max + pad * 1.8)
        ax.set_xlabel("Effective information")
        ax.grid(axis="x", color="#e5e9ee", lw=0.8)
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    y_labels = [_pair_label(*pair.split("->", 1)) for pair in pair_order]
    axes[0].set_yticks(range(len(pair_order)))
    axes[0].set_yticklabels(y_labels)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=lower_color, markeredgecolor="white", markersize=8, label="Lower EI"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=upper_color, markeredgecolor="white", markersize=8, label="Upper EI"),
    ]
    title_suffix = f" ({time_pair})" if time_pair else ""
    family_suffix = f"{family.replace('_', ' ').title()} {pair_category.replace('_', '-')}"
    fig.suptitle(f"Vertical EI Pairs by Organ - {family_suffix}{title_suffix}", fontsize=14, weight="semibold", y=0.98)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot corrected MIGNet vertical EI figures.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to mignet_vertical metrics.csv.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--time-pair", default=None, help="Optional filter like 12.5->13.5.")
    parser.add_argument("--family", choices=["all", "legacy_mixed", "louvain", "seurat"], default="all")
    parser.add_argument("--pair-category", choices=["all", "adjacent", "cross_level"], default="all")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metrics = _load_metrics(args.metrics)
    output_dir = args.output_dir or args.metrics.parent / "figures"
    _ensure_dir(output_dir)
    suffix = f"_{str(args.time_pair).replace('->', '_to_')}" if args.time_pair else ""
    family_suffix = f"{args.family}_{args.pair_category}"
    heatmap = output_dir / f"{family_suffix}_ei_gain_heatmap{suffix}.png"
    pairs = output_dir / f"{family_suffix}_coarse_graining_ei_pairs_by_organ{suffix}.png"
    plot_delta_heatmap(metrics, heatmap, time_pair=args.time_pair, family=args.family, pair_category=args.pair_category)
    plot_adjacent_pairs(metrics, pairs, time_pair=args.time_pair, family=args.family, pair_category=args.pair_category)
    print(f"Wrote {heatmap}")
    print(f"Wrote {pairs}")


if __name__ == "__main__":
    main()
