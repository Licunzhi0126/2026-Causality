from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AssetConfig
from .io import normalize_time_pair, select_coarse_run
from .style import GRID, RED, TEAL, TEAL_DARK, TEXT, set_asset_style


def _filter_network(metrics: pd.DataFrame, network_method: str, strict: bool) -> pd.DataFrame:
    if "network_method" not in metrics.columns:
        return metrics.copy()
    subset = metrics[metrics["network_method"].astype(str) == str(network_method)].copy()
    if subset.empty and strict:
        available = sorted(metrics["network_method"].dropna().astype(str).unique())
        raise KeyError(f"No rows for network_method={network_method!r}. Available: {available}")
    return subset if not subset.empty else metrics.copy()


def _single_metric(
    metrics: pd.DataFrame,
    *,
    pij_method: str,
    lower: str,
    upper: str,
    time_pair: str,
    strict: bool,
) -> float:
    pair = normalize_time_pair(time_pair)
    subset = metrics[
        (metrics["pij_method"].astype(str) == str(pij_method))
        & (metrics["lower_layer"].astype(str) == str(lower))
        & (metrics["upper_layer"].astype(str) == str(upper))
        & (metrics["time_pair"].astype(str) == pair)
    ]
    if subset.empty:
        if strict:
            raise KeyError(f"Missing PIJ metric: method={pij_method}, level={lower}->{upper}, pair={pair}")
        return float("nan")
    values = pd.to_numeric(subset["EI_gain"], errors="coerce").dropna()
    if values.empty:
        if strict:
            raise ValueError(f"No numeric EI_gain for {pij_method}, {lower}->{upper}, {pair}")
        return float("nan")
    if len(values) > 1 and strict:
        unique = np.unique(np.round(values.to_numpy(float), 12))
        if len(unique) > 1:
            raise ValueError(f"Multiple non-identical EI_gain rows for {pij_method}, {lower}->{upper}, {pair}")
    return float(values.iloc[0])


def build_table1(metrics: pd.DataFrame, cfg: AssetConfig) -> pd.DataFrame:
    """One merged PIJ-ablation table with hierarchy as an explicit column."""

    work = _filter_network(metrics, cfg.network_method, cfg.strict)
    rows: list[dict[str, object]] = []
    for level_label, (lower, upper) in cfg.levels.items():
        for display_name, method_name in cfg.pij_ablations.items():
            row: dict[str, object] = {"Method": display_name, "Hierarchy": level_label}
            for pair in cfg.time_pairs:
                row[pair] = _single_metric(
                    work,
                    pij_method=method_name,
                    lower=lower,
                    upper=upper,
                    time_pair=pair,
                    strict=cfg.strict,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_table2(metrics: pd.DataFrame, cfg: AssetConfig) -> pd.DataFrame:
    """DeltaEI across Spot->K150, K150->K40 and Spot->K40 for the primary PIJ method."""

    work = _filter_network(metrics, cfg.network_method, cfg.strict)
    rows = []
    for pair in cfg.time_pairs:
        row: dict[str, object] = {"Time pair": pair}
        for level_label, (lower, upper) in cfg.levels.items():
            row[level_label] = _single_metric(
                work,
                pij_method=cfg.primary_pij_method,
                lower=lower,
                upper=upper,
                time_pair=pair,
                strict=cfg.strict,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_counts_adjacent_chain(runs: pd.DataFrame, method: str, cfg: AssetConfig) -> dict[str, int | float]:
    t0, t1, t2 = tuple(cfg.time_points)
    run01 = select_coarse_run(
        runs, method, cfg.coarse_scale, f"{t0}->{t1}",
        k=cfg.coarse_k, seed=cfg.coarse_seed, strict=cfg.strict,
    )
    run12 = select_coarse_run(
        runs, method, cfg.coarse_scale, f"{t1}->{t2}",
        k=cfg.coarse_k, seed=cfg.coarse_seed, strict=cfg.strict,
    )
    return {
        t0: int(run01["hardK_t_paper"]) if pd.notna(run01["hardK_t_paper"]) else np.nan,
        t1: int(run12["hardK_t_paper"]) if pd.notna(run12["hardK_t_paper"]) else np.nan,
        t2: int(run12["hardK_tp_paper"]) if pd.notna(run12["hardK_tp_paper"]) else np.nan,
    }


def build_table3(runs: pd.DataFrame, cfg: AssetConfig) -> pd.DataFrame:
    rows = []
    for method in cfg.coarse_methods:
        row: dict[str, object] = {"Method": method}
        for pair in cfg.time_pairs:
            selected = select_coarse_run(
                runs, method, cfg.coarse_scale, pair,
                k=cfg.coarse_k, seed=cfg.coarse_seed, strict=cfg.strict,
            )
            row[f"DeltaEI {pair}"] = float(selected["delta_EI_paper"])
        counts = _cluster_counts_adjacent_chain(runs, method, cfg)
        for stage in cfg.time_points:
            row[f"K@{stage}"] = counts[str(stage)]
        rows.append(row)
    return pd.DataFrame(rows)


def _format_value(value: object, integer: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if integer:
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value)
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _max_cells_within_hierarchy(frame: pd.DataFrame, value_columns: list[str]) -> set[tuple[int, str]]:
    """Return (row-position, column-name) cells that are maxima within each hierarchy block."""

    marked: set[tuple[int, str]] = set()
    if "Hierarchy" not in frame.columns:
        return marked
    for _, group in frame.groupby("Hierarchy", sort=False):
        for col in value_columns:
            vals = pd.to_numeric(group[col], errors="coerce")
            finite = vals[np.isfinite(vals.to_numpy(float, na_value=np.nan))]
            if finite.empty:
                continue
            best = float(finite.max())
            for idx, value in vals.items():
                if pd.notna(value) and np.isclose(float(value), best, rtol=1e-10, atol=1e-12):
                    marked.add((int(frame.index.get_loc(idx)), col))
    return marked


def _max_cells_by_column(frame: pd.DataFrame, value_columns: list[str]) -> set[tuple[int, str]]:
    marked: set[tuple[int, str]] = set()
    for col in value_columns:
        vals = pd.to_numeric(frame[col], errors="coerce")
        finite = vals.dropna()
        if finite.empty:
            continue
        best = float(finite.max())
        for idx, value in vals.items():
            if pd.notna(value) and np.isclose(float(value), best, rtol=1e-10, atol=1e-12):
                marked.add((int(frame.index.get_loc(idx)), col))
    return marked


def draw_dataframe_table(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    subtitle: str | None = None,
    integer_columns: set[str] | None = None,
    first_col_width: float = 0.34,
    column_widths: Mapping[str, float] | None = None,
    red_cells: set[tuple[int, str]] | None = None,
) -> None:
    """Render a white-background publication table.

    The table is strictly monochrome (black/gray on white) except for red text
    marking selected best DeltaEI values. No teal/green/blue styling is used.
    """

    integer_columns = integer_columns or set()
    red_cells = red_cells or set()
    ax.axis("off")
    ax.text(0.0, 1.085, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color="#111111", va="bottom")
    if subtitle:
        ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic", va="bottom")

    columns = list(frame.columns)
    cell_text = []
    for _, row in frame.iterrows():
        cell_text.append([
            _format_value(row[col], integer=col in integer_columns) if col != columns[0] else str(row[col])
            for col in columns
        ])

    if column_widths:
        raw_widths = np.array([float(column_widths.get(col, 1.0)) for col in columns], dtype=float)
        raw_widths = np.where(raw_widths > 0, raw_widths, 1.0)
        col_widths = (raw_widths / raw_widths.sum()).tolist()
    else:
        remaining = max(0.01, 1.0 - first_col_width)
        col_widths = [first_col_width] + [remaining / max(1, len(columns) - 1)] * (len(columns) - 1)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colWidths=col_widths,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0.02, 1, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.55)
        cell.PAD = 0.012
        txt = cell.get_text()
        txt.set_color("#111111")
        if r == 0:
            txt.set_fontweight("bold")
            txt.set_fontsize(8.6)
            txt.set_color("#111111")
            cell.set_edgecolor("#777777")
            cell.set_linewidth(0.9)
            continue
        if c == 0:
            txt.set_fontweight("bold")
            txt.set_ha("left")
        col = columns[c]
        if col in {"Hierarchy", "Time pair"} or col in integer_columns:
            txt.set_fontweight("bold")
        if (r - 1, col) in red_cells:
            txt.set_color(RED)
            txt.set_fontweight("bold")


def render_table1_bundle(table: pd.DataFrame, path: Path, *, demo: bool = False, dpi: int = 300) -> None:
    set_asset_style()
    fig, ax = plt.subplots(figsize=(14.6, 10.8))
    value_columns = [c for c in table.columns if c not in {"Method", "Hierarchy"}]
    red_cells = _max_cells_within_hierarchy(table, value_columns)
    draw_dataframe_table(
        ax,
        table,
        title="Table 1 · PIJ method ablation across hierarchy",
        subtitle="Three hierarchy-specific ablations are merged; within each hierarchy and time pair, the largest DeltaEI is shown in red.",
        column_widths={
            "Method": 0.29,
            "Hierarchy": 0.20,
            "11.5->12.5": 0.17,
            "12.5->13.5": 0.17,
            "11.5->13.5": 0.17,
        },
        red_cells=red_cells,
    )

    mpl_table = ax.tables[0]
    hierarchy_counts = table.groupby("Hierarchy", sort=False).size().tolist()
    running = 0
    for count in hierarchy_counts[:-1]:
        running += int(count)
        row_index = running
        for c in range(len(table.columns)):
            cell = mpl_table[(row_index, c)]
            cell.set_edgecolor("#555555")
            cell.set_linewidth(1.25)

    if demo:
        fig.text(0.99, 0.995, "STYLE PREVIEW · DEMO VALUES", ha="right", va="top", fontsize=8.5, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_table2(frame: pd.DataFrame, path: Path, *, demo: bool = False, dpi: int = 300) -> None:
    set_asset_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.1))
    draw_dataframe_table(
        ax,
        frame,
        title="Table 2 · EI existence across hierarchy",
        subtitle="Primary PIJ method; columns are the three coarse-graining levels.",
        first_col_width=0.25,
    )
    if demo:
        fig.text(0.99, 0.99, "STYLE PREVIEW · DEMO VALUES", ha="right", va="top", fontsize=8.5, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_table3(frame: pd.DataFrame, path: Path, *, demo: bool = False, dpi: int = 300) -> None:
    set_asset_style()
    fig, ax = plt.subplots(figsize=(15.3, 4.8))
    integer_columns = {col for col in frame.columns if col.startswith("K@")}
    delta_columns = [col for col in frame.columns if col.startswith("DeltaEI ")]
    red_cells = _max_cells_by_column(frame, delta_columns)
    draw_dataframe_table(
        ax,
        frame,
        title="Table 3 · Optimal coarse-graining DeltaEI and learned cluster counts",
        subtitle="K uses the adjacent-chain policy; the largest DeltaEI in each time-pair column is shown in red.",
        integer_columns=integer_columns,
        first_col_width=0.29,
        red_cells=red_cells,
    )
    if demo:
        fig.text(0.99, 0.99, "STYLE PREVIEW · DEMO VALUES", ha="right", va="top", fontsize=8.5, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
