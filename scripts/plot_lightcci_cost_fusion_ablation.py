#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHOD_MAP: dict[str, tuple[str, str]] = {
    "compare_L_sot": ("L", "Cosine"),
    "compare_L_euc_sot": ("L", "Euclidean"),
    "compare_L_E_costmix_cos_sot": ("L+E", "Cosine"),
    "compare_L_E_costmix_euc_sot": ("L+E", "Euclidean"),
    "compare_L_Sr_costmix_cos_sot": ("L+SR", "Cosine"),
    "compare_L_Sr_costmix_euc_sot": ("L+SR", "Euclidean"),
    "compare_L_E_Sr_costmix_cos_sot": ("L+E+SR", "Cosine"),
    "compare_L_E_Sr_costmix_euc_sot": ("L+E+SR", "Euclidean"),
    "compare_E_sot": ("E", "Cosine"),
    "compare_E_euc_sot": ("E", "Euclidean"),
    "compare_E_Sr_costmix_cos_sot": ("E+SR", "Cosine"),
    "compare_E_Sr_costmix_euc_sot": ("E+SR", "Euclidean"),
}
FEATURE_ORDER = ("L", "L+E", "L+SR", "L+E+SR", "E", "E+SR")
DISTANCE_ORDER = ("Cosine", "Euclidean")
GROUP_COLUMNS = ("organ", "time_pair", "lower_layer", "upper_layer")


def _network_root(result_root: Path) -> Path:
    root = Path(result_root)
    nested = root / "network=light_cci"
    return nested if nested.is_dir() else root


def load_cost_fusion_metrics(
    result_root: Path,
    *,
    duplicate_policy: str = "error",
) -> pd.DataFrame:
    if duplicate_policy not in {"error", "mean"}:
        raise ValueError("duplicate_policy must be one of ['error', 'mean'].")
    network_root = _network_root(result_root)
    frames: list[pd.DataFrame] = []
    required = {*GROUP_COLUMNS, "EI_gain"}
    for method, (feature_group, distance) in METHOD_MAP.items():
        metrics_path = network_root / f"pij={method}" / "metrics.csv"
        if not metrics_path.is_file():
            continue
        frame = pd.read_csv(metrics_path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{metrics_path} is missing required columns {sorted(missing)}.")
        selected = frame.loc[:, [*GROUP_COLUMNS, "EI_gain"]].copy()
        selected["method"] = method
        selected["feature_group"] = feature_group
        selected["distance"] = distance
        frames.append(selected)
    if not frames:
        raise FileNotFoundError(
            f"No target cost-fusion metrics.csv files were found below {network_root}."
        )

    metrics = pd.concat(frames, ignore_index=True)
    duplicate_keys = ["method", *GROUP_COLUMNS]
    duplicates = metrics.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        duplicate_rows = metrics.loc[duplicates, duplicate_keys].sort_values(duplicate_keys)
        if duplicate_policy == "error":
            raise ValueError(
                "Duplicate method/group rows found:\n"
                + duplicate_rows.to_string(index=False)
            )
        metrics = (
            metrics.groupby([*duplicate_keys, "feature_group", "distance"], as_index=False, dropna=False)["EI_gain"]
            .mean()
        )
    return metrics


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def plot_cost_fusion_ablation(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int = 240,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    x = np.arange(len(FEATURE_ORDER), dtype=float)
    width = 0.36
    offsets = {"Cosine": -width / 2.0, "Euclidean": width / 2.0}
    colors = {"Cosine": "#4C78A8", "Euclidean": "#F58518"}

    grouped = metrics.groupby(list(GROUP_COLUMNS), sort=True, dropna=False)
    for group_values, frame in grouped:
        group = dict(zip(GROUP_COLUMNS, group_values))
        lookup = frame.set_index(["feature_group", "distance"])["EI_gain"]
        finite_values: list[float] = []
        fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        for distance in DISTANCE_ORDER:
            legend_drawn = False
            for index, feature_group in enumerate(FEATURE_ORDER):
                value = lookup.get((feature_group, distance), np.nan)
                position = x[index] + offsets[distance]
                if pd.notna(value) and np.isfinite(float(value)):
                    numeric = float(value)
                    finite_values.append(numeric)
                    ax.bar(
                        position,
                        numeric,
                        width=width,
                        color=colors[distance],
                        label=distance if not legend_drawn else None,
                    )
                    legend_drawn = True
                else:
                    ax.text(position, 0.0, "NA", ha="center", va="bottom", fontsize=8, color="#666666")

        extent = max([abs(value) for value in finite_values] + [1e-6])
        low = min(finite_values + [0.0])
        high = max(finite_values + [0.0])
        padding = max(0.08 * extent, 0.02)
        ax.set_ylim(low - padding, high + padding)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x, FEATURE_ORDER)
        ax.set_ylabel("EI_gain")
        ax.set_xlabel("Information combination")
        ax.set_title(
            f"{group['organ']} | {group['time_pair']} | "
            f"{group['lower_layer']} → {group['upper_layer']}"
        )
        ax.legend(
            handles=[Patch(color=colors[distance], label=distance) for distance in DISTANCE_ORDER],
            title="Vector distance",
            frameon=False,
        )
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)

        filename = _safe_filename(
            f"{group['organ']}__{group['time_pair']}__{group['lower_layer']}_to_{group['upper_layer']}.png"
        )
        path = output_dir / filename
        fig.savefig(path, dpi=max(200, int(dpi)))
        plt.close(fig)
        written.append(path)
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the fixed 12-method LightCCI cost-fusion EI_gain ablation."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duplicate-policy", choices=["error", "mean"], default="error")
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    metrics = load_cost_fusion_metrics(
        args.result_root,
        duplicate_policy=args.duplicate_policy,
    )
    paths = plot_cost_fusion_ablation(metrics, args.output_dir, dpi=args.dpi)
    print(f"Wrote {len(paths)} PNG files to {args.output_dir}")


if __name__ == "__main__":
    main()
