from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def apply_spatial_orientation(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = True,
) -> pd.DataFrame:
    """Apply one consistent spatial orientation transform to a coordinate frame."""
    missing = {x_col, y_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing coordinate columns: {sorted(missing)}")

    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

    if swap_xy:
        work[[x_col, y_col]] = work[[y_col, x_col]].to_numpy()

    if invert_x and work[x_col].notna().any():
        work[x_col] = float(work[x_col].min() + work[x_col].max()) - work[x_col]
    if invert_y and work[y_col].notna().any():
        work[y_col] = float(work[y_col].min() + work[y_col].max()) - work[y_col]
    return work


def set_equal_spatial_limits(ax: plt.Axes, frame: pd.DataFrame, pad_fraction: float = 0.04) -> None:
    x_min, x_max = float(frame["x"].min()), float(frame["x"].max())
    y_min, y_max = float(frame["y"].min()), float(frame["y"].max())
    span = max(x_max - x_min, y_max - y_min, 1.0)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    pad = span * pad_fraction
    half = span / 2.0 + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_aspect("equal", adjustable="box")


def clean_2d_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_signed(value: float, digits: int = 3, missing: str = "NA") -> str:
    if not np.isfinite(value):
        return missing
    return f"{value:+.{digits}f}"


def draw_figure_bracket(
    fig: plt.Figure,
    x0: float,
    x1: float,
    y: float,
    text: str,
    *,
    tick: float = 0.012,
    fontsize: float = 8.6,
    color: str = "#222222",
    text_side: Literal["above", "below"] = "above",
) -> None:
    line_kwargs = {"transform": fig.transFigure, "color": color, "lw": 1.15, "clip_on": False}
    fig.lines.append(Line2D([x0, x1], [y, y], **line_kwargs))
    fig.lines.append(Line2D([x0, x0], [y, y - tick], **line_kwargs))
    fig.lines.append(Line2D([x1, x1], [y, y - tick], **line_kwargs))
    text_y = y + 0.006 if text_side == "above" else y - tick - 0.006
    va = "bottom" if text_side == "above" else "top"
    fig.text((x0 + x1) / 2.0, text_y, text, ha="center", va=va, fontsize=fontsize, color=color)


def draw_axis_brace(
    ax: plt.Axes,
    x: float,
    y0: float,
    y1: float,
    text: str,
    *,
    fontsize: float = 9.2,
    color: str = "#242424",
    text_dx: float = 0.018,
    tick: float = 0.018,
) -> None:
    ax.plot([x, x], [y0, y1], color=color, lw=1.15, clip_on=False)
    ax.plot([x - tick, x], [y0, y0], color=color, lw=1.15, clip_on=False)
    ax.plot([x - tick, x], [y1, y1], color=color, lw=1.15, clip_on=False)
    ax.text(x + text_dx, (y0 + y1) / 2.0, text, ha="left", va="center", fontsize=fontsize, color=color)


def normalized_xy(
    frame: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    *,
    x_col: str = "x",
    y_col: str = "y",
    out_x: str = "nx",
    out_y: str = "ny",
) -> pd.DataFrame:
    x_min, x_max, y_min, y_max = bounds
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    scale = max(x_max - x_min, y_max - y_min, 1.0)
    work = frame.copy()
    work[out_x] = (pd.to_numeric(work[x_col], errors="coerce") - x_mid) / scale
    work[out_y] = (pd.to_numeric(work[y_col], errors="coerce") - y_mid) / scale
    return work


def spatial_bounds(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(frame["x"].min()),
        float(frame["x"].max()),
        float(frame["y"].min()),
        float(frame["y"].max()),
    )

