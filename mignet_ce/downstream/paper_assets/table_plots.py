from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import RED, set_asset_style

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


def render_table1_bundle(table: pd.DataFrame, path: Path, *, demo: bool = False, dpi: int = 300, title: str = "Table 1 · PIJ method ablation across hierarchy") -> None:
    set_asset_style()
    fig, ax = plt.subplots(figsize=(14.6, 10.8))
    value_columns = [c for c in table.columns if c not in {"Method", "Hierarchy"}]
    red_cells = _max_cells_within_hierarchy(table, value_columns)
    draw_dataframe_table(
        ax, table, title=title,
        subtitle="Hierarchy is shown once per block; thick rules separate hierarchy blocks. Best DeltaEI within each block/time pair is red.",
        column_widths={"Method": 0.29, "Hierarchy": 0.20, "11.5->12.5": 0.17, "12.5->13.5": 0.17, "11.5->13.5": 0.17},
        red_cells=red_cells,
    )

    mpl_table = ax.tables[0]
    hierarchy_col = list(table.columns).index("Hierarchy")
    groups = list(table.groupby("Hierarchy", sort=False).indices.items())
    for _label, positions in groups:
        positions = sorted(int(p) for p in positions)
        first_row, last_row = positions[0] + 1, positions[-1] + 1
        middle_row = positions[len(positions) // 2] + 1
        # Visually merge the hierarchy cells: no internal horizontal rules and
        # one centered label for the full block.
        for pos in positions:
            cell = mpl_table[(pos + 1, hierarchy_col)]
            if pos + 1 != middle_row:
                cell.get_text().set_text("")
            if pos == positions[0]:
                cell.visible_edges = "TLR"
            elif pos == positions[-1]:
                cell.visible_edges = "BLR"
            else:
                cell.visible_edges = "LR"
            cell.set_edgecolor("#555555")
            cell.set_linewidth(1.15)
        # Bold block separators across the complete table width.
        for c in range(len(table.columns)):
            top = mpl_table[(first_row, c)]
            bottom = mpl_table[(last_row, c)]
            top.set_edgecolor("#444444")
            bottom.set_edgecolor("#444444")
            top.set_linewidth(1.25)
            bottom.set_linewidth(1.25)

    if demo:
        fig.text(0.99, 0.995, "STYLE PREVIEW · DEMO VALUES", ha="right", va="top", fontsize=8.5, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def render_table2(
    frame: pd.DataFrame, path: Path, *, demo: bool = False, dpi: int = 300,
    title: str = "Table 2 · EI existence across hierarchy",
    subtitle: str = "Primary PIJ method; columns are the three coarse-graining levels.",
) -> None:
    set_asset_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.1))
    draw_dataframe_table(
        ax,
        frame,
        title=title,
        subtitle=subtitle,
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


def render_metric_grid(frame: pd.DataFrame, path: Path, *, title: str, subtitle: str, dpi: int = 300) -> None:
    """Render the matched DeltaEI and CG input-scale matrices identically."""

    set_asset_style()
    fig, ax = plt.subplots(figsize=(10.6, 4.1))
    draw_dataframe_table(
        ax, frame, title=title, subtitle=subtitle, first_col_width=0.25,
        red_cells=_max_cells_by_column(frame, list(frame.columns[1:])),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _feature_ablation_panel_frame(hierarchies: tuple[str, ...]) -> pd.DataFrame:
    """Return the paper-facing 10-row feature-ablation layout for each hierarchy.

    Values are intentionally left blank. Scientific result generation lives in the
    PIJ pipeline; paper_assets only owns the presentation contract.
    """

    row_spec = (
        ("CCI", "NMF", "KL"),
        ("CCI", "NMF", "KL + OT"),
        ("CCI", "LAP", "KL"),
        ("CCI", "LAP", "KL + OT"),
        ("GRN", "Dual-end gating", "KL"),
        ("GRN", "Dual-end gating", "KL + OT"),
        ("CCI + GRN", "NMF + dual-end gating", "KL"),
        ("CCI + GRN", "NMF + dual-end gating", "KL + OT"),
        ("CCI + GRN", "LAP + dual-end gating", "KL"),
        ("CCI + GRN", "LAP + dual-end gating", "KL + OT"),
    )
    rows: list[dict[str, object]] = []
    for hierarchy in hierarchies:
        for input_name, feature_method, probability_method in row_spec:
            rows.append(
                {
                    "Hierarchy": hierarchy,
                    "Input": input_name,
                    "Feature method": feature_method,
                    "Probability method": probability_method,
                    "11.5->12.5": "—",
                    "12.5->13.5": "—",
                    "11.5->13.5": "—",
                }
            )
    return pd.DataFrame(rows)


def _visually_merge_repeated_cells(
    table,
    frame: pd.DataFrame,
    *,
    column: str,
    group_columns: tuple[str, ...],
    linewidth: float = 0.85,
) -> None:
    """Visually row-span identical labels within higher-level groups."""

    columns = list(frame.columns)
    col_idx = columns.index(column)
    for _, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        for value, subgroup in group.groupby(column, sort=False, dropna=False):
            positions = sorted(int(frame.index.get_loc(idx)) for idx in subgroup.index)
            if not positions:
                continue
            middle = positions[len(positions) // 2] + 1
            for pos in positions:
                cell = table[(pos + 1, col_idx)]
                if pos + 1 != middle:
                    cell.get_text().set_text("")
                if len(positions) == 1:
                    cell.visible_edges = "TBLR"
                elif pos == positions[0]:
                    cell.visible_edges = "TLR"
                elif pos == positions[-1]:
                    cell.visible_edges = "BLR"
                else:
                    cell.visible_edges = "LR"
                cell.set_edgecolor("#777777")
                cell.set_linewidth(linewidth)


def _draw_feature_ablation_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    panel_title: str,
) -> None:
    ax.axis("off")
    columns = list(frame.columns)
    widths = np.array([0.17, 0.12, 0.22, 0.16, 0.11, 0.11, 0.11], dtype=float)
    widths = (widths / widths.sum()).tolist()
    table = ax.table(
        cellText=frame.astype(str).values.tolist(),
        colLabels=columns,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0.015, 1, 0.955],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.45)
        cell.PAD = 0.006
        text = cell.get_text()
        text.set_color("#111111")
        if r == 0:
            text.set_fontweight("bold")
            text.set_fontsize(7.1)
            cell.set_edgecolor("#666666")
            cell.set_linewidth(0.9)
        elif c in (0, 1):
            text.set_fontweight("bold")

    # Merge in hierarchical order so the visual structure matches the supplied
    # spreadsheet: hierarchy -> biological input -> feature representation.
    _visually_merge_repeated_cells(table, frame, column="Hierarchy", group_columns=("Hierarchy",), linewidth=1.05)
    _visually_merge_repeated_cells(table, frame, column="Input", group_columns=("Hierarchy", "Input"), linewidth=0.85)
    _visually_merge_repeated_cells(
        table, frame, column="Feature method", group_columns=("Hierarchy", "Input", "Feature method"), linewidth=0.75
    )

    # Strong hierarchy-block boundaries; lighter boundaries split CCI / GRN /
    # CCI+GRN input blocks without over-boxing every row.
    hierarchy_groups = list(frame.groupby("Hierarchy", sort=False).indices.items())
    for _, positions in hierarchy_groups:
        positions = sorted(int(p) for p in positions)
        first_row, last_row = positions[0] + 1, positions[-1] + 1
        for c in range(len(columns)):
            table[(first_row, c)].set_edgecolor("#3F3F3F")
            table[(last_row, c)].set_edgecolor("#3F3F3F")
            table[(first_row, c)].set_linewidth(1.15)
            table[(last_row, c)].set_linewidth(1.15)
        block = frame.iloc[positions]
        for _, input_group in block.groupby("Input", sort=False):
            p = sorted(int(frame.index.get_loc(idx)) for idx in input_group.index)
            if p:
                last = p[-1] + 1
                for c in range(1, len(columns)):
                    table[(last, c)].set_edgecolor("#858585")
                    table[(last, c)].set_linewidth(max(table[(last, c)].get_linewidth(), 0.72))

    ax.text(0.0, 0.992, panel_title, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def render_feature_ablation_preview(
    path: Path,
    *,
    dpi: int = 300,
    left_hierarchies: tuple[str, ...] = ("Spot -> K150", "K150 -> K40", "K40 -> K10"),
    right_hierarchies: tuple[str, ...] = ("Spot -> K40", "Spot -> K10", "K150 -> K10"),
) -> None:
    """Render the requested six-group feature-ablation table layout.

    This is deliberately a *layout preview*: result cells stay blank until the
    missing controlled PIJ variants are produced by the scientific pipeline.
    """

    set_asset_style()
    fig, axes = plt.subplots(1, 2, figsize=(23.5, 12.8))
    fig.suptitle("Feature extraction / PIJ construction ablation", fontsize=13.5, fontweight="bold", y=0.997)
    fig.text(
        0.5,
        0.979,
        "Six hierarchy groups; each group uses the same 10 controlled rows. Values intentionally blank in this style preview.",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#555555",
        style="italic",
    )
    left = _feature_ablation_panel_frame(left_hierarchies)
    right = _feature_ablation_panel_frame(right_hierarchies)
    _draw_feature_ablation_panel(axes[0], left, panel_title="Table 1A · Adjacent hierarchy chain")
    _draw_feature_ablation_panel(axes[1], right, panel_title="Table 1B · Cross-scale hierarchy pairs")
    fig.text(0.995, 0.995, "STYLE PREVIEW · NO SCIENTIFIC VALUES", ha="right", va="top", fontsize=8.0, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_feature_ablation_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    demo: bool = False,
    dpi: int = 300,
) -> None:
    """Render one controlled PIJ-ablation table in the legacy Table-1 style.

    Table 1A and Table 1B are deliberately separate figures.  The visual
    language matches the existing paper assets: white background, light gray
    cell rules, black labels, red within-hierarchy maxima, and stronger rules
    around each hierarchy block.
    """

    required = {
        "Hierarchy", "Input", "Feature method", "PIJ construction",
        "11.5->12.5", "12.5->13.5", "11.5->13.5",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Feature-ablation table lacks {sorted(required - set(frame.columns))}")
    frame = frame[
        ["Hierarchy", "Input", "Feature method", "PIJ construction",
         "11.5->12.5", "12.5->13.5", "11.5->13.5"]
    ].copy()
    set_asset_style()
    fig, ax = plt.subplots(figsize=(15.3, 13.0))
    value_columns = ["11.5->12.5", "12.5->13.5", "11.5->13.5"]
    red_cells = _max_cells_within_hierarchy(frame, value_columns)
    draw_dataframe_table(
        ax,
        frame,
        title=title,
        subtitle=(
            "Controlled feature/transition ablation. KL and KL + OT share the exact same KL cost; "
            "red marks the largest DeltaEI within each hierarchy/time pair."
        ),
        column_widths={
            "Hierarchy": 0.16,
            "Input": 0.13,
            "Feature method": 0.24,
            "PIJ construction": 0.15,
            "11.5->12.5": 0.106,
            "12.5->13.5": 0.106,
            "11.5->13.5": 0.106,
        },
        red_cells=red_cells,
    )
    table = ax.tables[0]

    # Visually row-span the semantic hierarchy while retaining the same basic
    # Matplotlib table style used by the original Table 1.
    _visually_merge_repeated_cells(
        table, frame, column="Hierarchy", group_columns=("Hierarchy",), linewidth=1.10
    )
    _visually_merge_repeated_cells(
        table, frame, column="Input", group_columns=("Hierarchy", "Input"), linewidth=0.82
    )
    _visually_merge_repeated_cells(
        table,
        frame,
        column="Feature method",
        group_columns=("Hierarchy", "Input", "Feature method"),
        linewidth=0.72,
    )

    columns = list(frame.columns)
    # Strong rules isolate each 10-row hierarchy block; medium rules separate
    # CCI / GRN / CCI+GRN within a block.
    for _, positions_raw in frame.groupby("Hierarchy", sort=False).indices.items():
        positions = sorted(int(p) for p in positions_raw)
        first_row, last_row = positions[0] + 1, positions[-1] + 1
        for c in range(len(columns)):
            for r in (first_row, last_row):
                table[(r, c)].set_edgecolor("#444444")
                table[(r, c)].set_linewidth(1.25)
        block = frame.iloc[positions]
        for _, input_group in block.groupby("Input", sort=False):
            group_pos = sorted(int(frame.index.get_loc(idx)) for idx in input_group.index)
            if not group_pos:
                continue
            last = group_pos[-1] + 1
            for c in range(1, len(columns)):
                cell = table[(last, c)]
                cell.set_edgecolor("#888888")
                cell.set_linewidth(max(cell.get_linewidth(), 0.72))

    if demo:
        fig.text(
            0.99, 0.995, "STYLE PREVIEW · DEMO VALUES",
            ha="right", va="top", fontsize=8.5, color="#777777",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
