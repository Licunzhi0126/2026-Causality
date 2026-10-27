"""New publication visualization layer. Never calls deprecated paper_assets.

Figure 2 deliberately contains NO ROI enlargement or invented embryo silhouette.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .methods import METHOD_LABELS


def _merge_group_column_visually(ax, fig, table, rendered: list[list[str]], group_col: int) -> None:
    """Draw a true-looking vertically merged cell for consecutive equal labels.

    Matplotlib's Table does not implement rowspan.  Suppress every internal
    horizontal boundary in the grouped column, then place exactly one label
    vertically centered over the resulting block.  The other columns retain
    every row and every source measurement.  This changes *only* presentation.
    """
    if not rendered:
        return
    fig.canvas.draw()  # finalize actual cell coordinates before positioning labels
    blocks: list[tuple[int, int, str]] = []
    start = 1
    for row in range(2, len(rendered) + 2):
        if row == len(rendered) + 1 or rendered[row - 1][group_col] != rendered[start - 1][group_col]:
            blocks.append((start, row - 1, rendered[start - 1][group_col]))
            start = row
    for first, last, label in blocks:
        for idx in range(first, last + 1):
            cell = table[(idx, group_col)]
            cell.get_text().set_text("")
            if first == last:
                cell.visible_edges = "LTRB"
            elif idx == first:
                cell.visible_edges = "TLR"
            elif idx == last:
                cell.visible_edges = "BLR"
            else:
                cell.visible_edges = "LR"
        top = table[(first, group_col)]
        bottom = table[(last, group_col)]
        # Table cell coordinates are in Axes coordinates after the canvas draw.
        x = float(top.get_x() + top.get_width() * 0.055)
        y = float((top.get_y() + top.get_height() + bottom.get_y()) / 2)
        ax.text(x, y, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=table[(first, group_col + 1)].get_text().get_fontsize(),
                color="#2a2a2a", fontweight="normal", zorder=10)


def save_table(frame: pd.DataFrame, path: Path, *, title: str, subtitle: str, group_cols: tuple[str, ...] = (),
               allow_missing: bool = False) -> None:
    """Formal-ablation-style table; vertically merge repeated Method groups.

    Values/NaNs in the data columns are never changed to improve appearance.
    Only the repeated Method labels receive a visual rowspan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(frame.columns)
    short = {
        **METHOD_LABELS,
        "seurat_k150": "K150", "seurat_k40": "K40", "spot": "Spot",
    }
    rendered: list[list[str]] = []
    for _, row in frame.iterrows():
        output = []
        for col in cols:
            v = row[col]
            if pd.isna(v):
                if not allow_missing:
                    raise ValueError(f"Cannot render missing value in formal result: row={_}, column={col}")
                output.append("—")  # Exploratory missing K10 only; never a measured zero.
            elif isinstance(v, (float, np.floating)):
                output.append(f"{v:+.4f}" if "ei" in title.lower() else f"{v:.4f}")
            else:
                output.append(short.get(str(v), str(v)))
        rendered.append(output)
    widths = [max(10, len(str(col)), max((len(r[i]) for r in rendered), default=0))
              for i, col in enumerate(cols)]
    weights = np.array([max(.95, min(2.9, w / 11.0)) for w in widths], dtype=float)
    weights /= weights.sum()
    fw = max(11.8, min(17.8, 1.95 * len(cols) + 1.5))
    fh = 1.60 + .48 * (len(frame) + 1)
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([.025, .035, .95, .79])
    ax.axis("off")
    fig.text(.025, .965, title, fontsize=20, color="#242424", va="top", fontweight="normal")
    fig.text(.025, .875, subtitle, fontsize=12, color="#666666", va="top")
    table = ax.table(cellText=rendered, colLabels=cols, colWidths=weights.tolist(),
                     cellLoc="center", loc="upper left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12 if len(cols) <= 5 else 10.8)
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#D1D1D1")
        cell.set_linewidth(.60)
        cell.PAD = .025
        cell.get_text().set_color("#2a2a2a")
        cell.get_text().set_fontweight("normal")
        if j == 0 and i > 0:
            cell.get_text().set_ha("left")
    if group_cols:
        if len(group_cols) != 1 or group_cols[0] not in cols:
            raise ValueError("Visual rowspan expects one valid group column")
        group_index = cols.index(group_cols[0])
        if group_index + 1 >= len(cols):
            raise ValueError("Group column must precede other table columns")
        _merge_group_column_visually(ax, fig, table, rendered, group_index)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)



def _triplet_columns() -> list[str]:
    from .tables import PAIRS
    return ["Method", "Input scale"] + [f"{pair}::{metric}" for pair in PAIRS
                    for metric in ("trained_delta_EI", "Keff_t", "Keff_tp")]


def _validate_triplet_layout(frame: pd.DataFrame) -> None:
    from .tables import METHODS, SCALES
    expected_rows = [(method, scale) for method in METHODS for scale in SCALES]
    if list(frame.columns) != _triplet_columns():
        raise ValueError("Expected canonical triplet-grid columns")
    if list(zip(frame["Method"], frame["Input scale"])) != expected_rows:
        raise ValueError(f"Expected {len(expected_rows)} canonical method-scale rows in order")


def save_triplet_table(frame: pd.DataFrame, path: Path, *, title: str, subtitle: str) -> None:
    """Validate scientific values, then render DeltaEI | Keff_t | Keff_tp cells."""
    _validate_triplet_layout(frame)
    rendered = frame.copy()
    for column in _triplet_columns()[2:]:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("No missing/invalid metric may be rendered")
        pattern = "{:+.4f}" if column.endswith("::trained_delta_EI") else "{:.2f}"
        rendered[column] = values.map(pattern.format)
    _save_triplet_layout(rendered, path, title=title, subtitle=subtitle)


def _save_triplet_layout(frame: pd.DataFrame, path: Path, *, title: str, subtitle: str) -> None:
    """Shared geometry for validated result text and explicitly labeled previews."""
    from matplotlib.patches import Rectangle
    from .tables import METHODS, SCALES, PAIRS
    _validate_triplet_layout(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    widths = [2.35, 1.13] + [1.22, .86, .86] * len(PAIRS)
    coords = np.r_[0, np.cumsum(widths)]
    width, count = float(coords[-1]), len(frame)
    height = 2.0 + .44 * (count + 1)
    fig, ax = plt.subplots(figsize=(15.7, height))
    fig.subplots_adjust(top=1 - 1.45 / height, bottom=.65 / height, left=.035, right=.985)
    ax.set_xlim(0, width)
    ax.set_ylim(0, count + 1)
    ax.axis("off")
    line, foreground = "#D1D1D1", "#242424"

    def cell(column, row, text="", fontsize=11, width_override=None):
        x = coords[column]
        w = widths[column] if width_override is None else width_override
        ax.add_patch(Rectangle((x, row), w, 1, facecolor="white", edgecolor=line, linewidth=.65))
        if text:
            ax.text(x + w / 2, row + .5, text, ha="center", va="center",
                    fontsize=fontsize, color=foreground)

    for column, label in enumerate(("Method", "Input scale")):
        cell(column, count, label)
    for index, pair in enumerate(PAIRS):
        start = 2 + 3 * index
        cell(start, count, pair.replace("->", "→"), fontsize=11.4,
             width_override=sum(widths[start:start + 3]))
    scales = {"spot": "Spot", "seurat_k150": "K150", "seurat_k40": "K40"}
    for index, (_, row) in enumerate(frame.iterrows()):
        y = count - index - 1
        cell(1, y, scales[str(row["Input scale"])])
        for pair_index, pair in enumerate(PAIRS):
            for metric_index, metric in enumerate(("trained_delta_EI", "Keff_t", "Keff_tp")):
                cell(2 + pair_index * 3 + metric_index, y, str(row[f"{pair}::{metric}"]),
                     fontsize=10.3 if metric_index == 0 else 9.6)
    for index, method in enumerate(METHODS):
        low = count - (index + 1) * len(SCALES)
        ax.add_patch(Rectangle((coords[0], low), widths[0], len(SCALES),
                              facecolor="white", edgecolor=line, linewidth=.65))
        ax.text(coords[0] + .12, low + len(SCALES) / 2, METHOD_LABELS[method],
                va="center", ha="left", fontsize=11.2, color=foreground)
    fig.text(.035, 1 - .22 / height, title, fontsize=18.5, color=foreground, va="top")
    fig.text(.035, 1 - .78 / height, subtitle, fontsize=9.7, color="#666666", va="top")
    fig.savefig(path.with_suffix(".png"), dpi=210, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _coordinates(data_root: Path, scale: str, time: str) -> pd.DataFrame:
    stem={"seurat_k150":"seurat150","seurat_k40":"seurat","seurat_k10":"seurat10"}[scale]
    path=Path(data_root)/scale/"heart"/f"{stem}_heart_{time}_spot_domain_map.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Real coordinate map is required: {path}")
    df=pd.read_csv(path, dtype={"spot_id":str,"domain_id":str})
    if not {"spot_id","domain_id","x","y"}<=set(df):
        raise ValueError(f"Invalid map: {path}")
    return df


def _plot_layer(ax, df: pd.DataFrame, labels: pd.Series, title: str, cmap: str="tab20") -> None:
    if labels is None:
        ax.scatter(pd.to_numeric(df.x), -pd.to_numeric(df.y), color="#B65355", s=2.0, alpha=.80, linewidths=0,rasterized=True)
    else:
        codes=pd.Categorical(labels.astype(str)).codes
        ax.scatter(pd.to_numeric(df.x), -pd.to_numeric(df.y), c=codes, cmap=cmap, s=2.0, alpha=.84, linewidths=0,rasterized=True)
    ax.set_aspect("equal",adjustable="datalim")
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_title(title,fontsize=9.5,weight="bold")


from .paper_figure_layouts import render_figure1, render_figure1_k10, render_figure2


def _stage_data(data_root:Path, stage:str, scale:str) -> pd.DataFrame:
    d=_coordinates(data_root,scale,stage)
    return d[["spot_id","domain_id","x","y"]]


def draw_natural_figure(data_root: Path, ei_grid: pd.DataFrame, time_pair: str, path: Path, *, require_k10: bool=True, draft: bool=False) -> None:
    match=ei_grid[ei_grid["Time pair"].eq(time_pair)]
    if len(match)!=1:raise ValueError("Missing measured time pair " + time_pair)
    gains={col.replace(" -> "," -> "):float(match.iloc[0][col]) for col in match.columns if "->" in col and col!="Time pair"}
    t,tp=time_pair.split("->")
    kw=dict(source_stage=t,target_stage=tp,
            source_spot=_stage_data(data_root,t,"seurat_k40"),target_spot=_stage_data(data_root,tp,"seurat_k40"),
            source_k150=_stage_data(data_root,t,"seurat_k150"),target_k150=_stage_data(data_root,tp,"seurat_k150"),
            source_k40=_stage_data(data_root,t,"seurat_k40"),target_k40=_stage_data(data_root,tp,"seurat_k40"),
            gains=gains,output_path=Path(path).with_suffix(".png"),demo=draft)
    if require_k10:
        kw.update(source_k10=_stage_data(data_root,t,"seurat_k10"),target_k10=_stage_data(data_root,tp,"seurat_k10"))
        render_figure1_k10(**kw)
    else:render_figure1(**kw)


def draw_optimal_figure(data_root: Path, input_scale: str, time_pair: str, run_dir: Path, trained_delta: float, path: Path, *, draft: bool=False) -> None:
    t,tp=time_pair.split("->")
    stage_assignments=[]
    for stage,filename in ((t,"assignments_t.csv"),(tp,"assignments_tp.csv")):
        d=_stage_data(data_root,stage,"seurat_k40")
        ass=pd.read_csv(Path(run_dir)/filename,dtype={"spot_id":str})
        if not {"spot_id","hard_cluster"} <= set(ass):raise ValueError(f"Invalid assignment file {filename}")
        if input_scale=="spot": mapped=d.spot_id.map(ass.set_index("spot_id").hard_cluster)
        else:
            lookup=_stage_data(data_root,stage,input_scale).set_index("spot_id").domain_id
            mapped=d.spot_id.map(lookup).map(ass.set_index("spot_id").hard_cluster)
        if mapped.isna().any():raise ValueError(f"Missing {mapped.isna().sum()} assignments")
        stage_assignments.append(pd.DataFrame({"spot_id":d.spot_id,"hard_cluster":mapped.to_numpy()}))
    render_figure2(source_stage=t,target_stage=tp,source_spot=_stage_data(data_root,t,"seurat_k40"),target_spot=_stage_data(data_root,tp,"seurat_k40"),source_assignments=stage_assignments[0],target_assignments=stage_assignments[1],delta_ei=trained_delta,method_label=f"{input_scale} · saved trained PIJ",output_path=Path(path).with_suffix(".png"),demo=draft)
