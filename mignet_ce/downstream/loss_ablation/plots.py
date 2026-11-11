"""Render the same grouped table geometry for previews and trained results."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .tables import ENDPOINTS, INFORMATION_COLUMNS, OCCUPANCY_COLUMNS, table_grid
from .variants import METHOD_LABELS, STAGES, TARGET_K, TIME_PAIRS


SCALE_LABELS = {"spot": "Spot", "seurat_k150": "Seurat K150", "seurat_k40": "Seurat K40"}


def preview_rows(scales: tuple[str, ...], methods: tuple[str, ...] = METHOD_LABELS) -> list[dict[str, object]]:
    rows = []
    for scale in scales:
        k = TARGET_K[scale]
        for index, method in enumerate(methods):
            for stage_number, stage in enumerate(STAGES):
                for pair_number, (source, target) in enumerate(TIME_PAIRS):
                    available = .048 + .008 * pair_number + .0006 * index
                    retained = .0008 + .0004 * index + .004 * stage_number
                    row = {
                        "input_scale": scale, "method": method, "stage": stage,
                        "source_time": source, "target_time": target,
                        "delta_ei_bits": .79 + .032 * index + .013 * pair_number - .04 * stage_number,
                        "cq": retained / available, "i_retain_bits": retained,
                        "i_available_bits": available, "fallback_used": False,
                    }
                    for end_number, endpoint in enumerate(ENDPOINTS):
                        hard_k = k - (2 + index + end_number + pair_number) % (k // 2)
                        gap = .35 + .22 * index + .31 * end_number + .17 * stage_number
                        row[f"hard_k_{endpoint}"] = hard_k
                        row[f"keff_hard_{endpoint}"] = hard_k - gap
                        row[f"hard_k_minus_keff_hard_{endpoint}"] = gap
                    rows.append(row)
    return rows


def _metric_text(row: dict, metric: str, occupancy: bool) -> str:
    if occupancy:
        fmt = "d" if metric == "hard_k" else ".2f"
        return " / ".join(format(row[f"{metric}_{end}"], fmt) for end in ENDPOINTS)
    value = row[metric]
    if metric == "delta_ei_bits":
        return f"{value:+.4f}"
    if 0 < abs(value) < .0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def draw_tables(
    rows: list[dict[str, object]], out_dir: Path, scales: tuple[str, ...], *, preview: bool = False,
    methods: tuple[str, ...] = METHOD_LABELS,
) -> None:
    grid = table_grid(rows, scales, methods)
    fallback = any(row["fallback_used"] for row in rows)
    for name, title, columns, subtitle in (
        ("information", "Table 1  |  Information metrics", INFORMATION_COLUMNS,
         "ΔEI, Iretain and Iavailable in bits. CQ is a ratio."),
        ("cluster_occupancy", "Table 2  |  Hard cluster occupancy", OCCUPANCY_COLUMNS,
         "Values shown as source / target within each time pair."),
    ):
        occupancy = name == "cluster_occupancy"
        width = 15.3
        body_rows = len(methods) * len(STAGES)
        panel_units = body_rows + 3.65
        grid_height = len(scales) * panel_units
        height = 1.38 + .34 * grid_height
        figure = Figure(figsize=(width, height), facecolor="white")
        FigureCanvasAgg(figure)
        ax = figure.add_axes((.04, .53 / height, .92, 1 - 1.38 / height))
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(grid_height, 0)

        method_width = .12 if max(map(len, methods)) > 12 else .10
        column_widths = [method_width, .075]
        metric_width = (1.0 - sum(column_widths)) / len(TIME_PAIRS)
        weights = (1.0, 1.05, 1.40) if occupancy else (1.0, 1.0, 1.0, 1.0)
        column_widths.extend(metric_width * weight / sum(weights) for _ in TIME_PAIRS for weight in weights)
        boundaries = [0.0]
        for column_width in column_widths:
            boundaries.append(boundaries[-1] + column_width)

        def cell(column, y, value, *, colspan=1, rowspan=1, fill="white", bold=False, left=False):
            x = boundaries[column]
            w = boundaries[column + colspan] - x
            ax.add_patch(Rectangle((x, y), w, rowspan, facecolor=fill, edgecolor="#CFD3D7",
                                   linewidth=.55, clip_on=False))
            ax.text(x + .012 if left else x + w / 2, y + rowspan / 2, value,
                    ha="left" if left else "center", va="center", fontsize=9.6,
                    color="#20262D", fontweight="semibold" if bold else "normal")

        for scale_index, scale in enumerate(scales):
            top = scale_index * panel_units
            ax.text(0, top + .22, f"{SCALE_LABELS[scale]}  ·  Target K = {TARGET_K[scale]}",
                    va="top", fontsize=10.6, fontweight="semibold", color="#36434F")
            header_y = top + 1.08
            cell(0, header_y, "Method", rowspan=2, fill="#E9EDF0", bold=True, left=True)
            cell(1, header_y, "Stage", rowspan=2, fill="#E9EDF0", bold=True)
            for pair_index, (source, target) in enumerate(TIME_PAIRS):
                start = 2 + pair_index * len(columns)
                cell(start, header_y, f"{source}–{target}", colspan=len(columns), fill="#E9EDF0", bold=True)
                for offset, label in enumerate(columns.values()):
                    cell(start + offset, header_y + 1, label, fill="#F3F5F6", bold=True)

            for method_index, method in enumerate(methods):
                y = header_y + 2 + method_index * len(STAGES)
                cell(0, y, method, rowspan=2, fill="#F3F5F6", bold=True, left=True)
                for stage_index, stage in enumerate(STAGES):
                    row_y = y + stage_index
                    fill = "#FAFBFC" if stage_index else "white"
                    cell(1, row_y, f"Stage {stage_index + 1}", fill=fill)
                    for pair_index, pair in enumerate(TIME_PAIRS):
                        row = grid[scale, method, stage, pair]
                        start = 2 + pair_index * len(columns)
                        for offset, metric in enumerate(columns):
                            value = _metric_text(row, metric, occupancy)
                            if row["fallback_used"] and offset == 0:
                                value += " †"
                            cell(start + offset, row_y, value, fill=fill)

        figure.text(.04, 1 - .23 / height, title, va="top", fontsize=16.5,
                    color="#20262D", fontweight="semibold")
        figure.text(.04, 1 - .56 / height, subtitle, va="top", fontsize=10, color="#57636E")
        if preview:
            figure.text(.96, 1 - .27 / height, "LAYOUT PREVIEW · EXAMPLE VALUES", ha="right",
                        va="top", fontsize=10, color="#57636E", fontweight="semibold")
        if fallback:
            figure.text(.04, .27 / height, "† Stage 2 selected by the relaxed checkpoint rule.",
                        va="bottom", fontsize=9, color="#57636E")
        figure.savefig(out_dir / f"{name}.png", dpi=210, facecolor="white")
        if not preview:
            figure.savefig(out_dir / f"{name}.pdf", facecolor="white")
