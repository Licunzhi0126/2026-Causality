from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def align_coords(coords: pd.DataFrame, units: Sequence[str]) -> np.ndarray:
    units = list(map(str, units))
    if coords.empty:
        return np.zeros((len(units), 2), dtype=float)
    work = coords.copy()
    work.index = work.index.astype(str)
    missing = [unit for unit in units if unit not in work.index]
    if missing:
        pad = pd.DataFrame(np.zeros((len(missing), 2), dtype=float), index=missing, columns=["x", "y"])
        work = pd.concat([work.loc[:, ["x", "y"]], pad], axis=0)
    return work.loc[units, ["x", "y"]].to_numpy(dtype=float)


def normalize_coords_pair(source_coords: np.ndarray, target_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_coords, dtype=float)
    target = np.asarray(target_coords, dtype=float)
    if source.size == 0 or target.size == 0:
        return source.copy(), target.copy()
    combined = np.vstack([source[:, :2], target[:, :2]])
    mins = np.nanmin(combined, axis=0)
    maxs = np.nanmax(combined, axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (source[:, :2] - mins) / denom, (target[:, :2] - mins) / denom
