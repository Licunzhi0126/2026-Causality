from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from mignet_ce.io.loaders import ExpressionData


@dataclass
class UnitAssignments:
    layer: str
    rows: pd.DataFrame


@dataclass
class OverlapMapping:
    lower_units: List[str]
    upper_units: List[str]
    counts: np.ndarray
    weights: np.ndarray

    def coverage_counts(self) -> np.ndarray:
        return self.counts.sum(axis=0)


def load_unit_assignments(layer: str, expression: ExpressionData, spot_domain_map: Optional[Path]) -> UnitAssignments:
    if layer == "spot":
        if "organ" in expression.obs.columns:
            spot_ids = expression.obs["organ"].astype(str).to_numpy() + "__" + pd.Index(expression.units).astype(str).to_numpy()
        else:
            spot_ids = expression.units
        rows = pd.DataFrame({"spot_id": spot_ids, "unit_id": expression.units})
        return UnitAssignments(layer=layer, rows=rows)

    if spot_domain_map is None:
        raise ValueError(f"Layer {layer} needs a spot-domain map.")
    mapping = pd.read_csv(spot_domain_map)
    if "spot_id" not in mapping.columns or "domain_id" not in mapping.columns:
        raise ValueError(f"{spot_domain_map} must contain 'spot_id' and 'domain_id'.")
    rows = mapping.loc[:, ["spot_id", "domain_id"]].copy()
    rows["spot_id"] = rows["spot_id"].astype(str)
    if "organ" in mapping.columns:
        rows["spot_id"] = mapping["organ"].astype(str) + "__" + rows["spot_id"]
    rows["unit_id"] = rows["domain_id"].astype(str)
    return UnitAssignments(layer=layer, rows=rows.loc[:, ["spot_id", "unit_id"]].drop_duplicates())


def build_overlap_mapping(
    lower: UnitAssignments,
    upper: UnitAssignments,
    lower_units: Sequence[str],
    upper_units: Sequence[str],
) -> OverlapMapping:
    lower_units = list(map(str, lower_units))
    upper_units = list(map(str, upper_units))
    lower_index = {unit: idx for idx, unit in enumerate(lower_units)}
    upper_index = {unit: idx for idx, unit in enumerate(upper_units)}

    left = lower.rows.rename(columns={"unit_id": "lower_unit"})
    right = upper.rows.rename(columns={"unit_id": "upper_unit"})
    merged = left.merge(right, on="spot_id", how="inner")
    merged = merged[merged["lower_unit"].isin(lower_index) & merged["upper_unit"].isin(upper_index)]

    counts = np.zeros((len(lower_units), len(upper_units)), dtype=float)
    if not merged.empty:
        grouped = merged.groupby(["lower_unit", "upper_unit"]).size().reset_index(name="n")
        for row in grouped.itertuples(index=False):
            counts[lower_index[str(row.lower_unit)], upper_index[str(row.upper_unit)]] = float(row.n)

    row_sums = counts.sum(axis=1, keepdims=True)
    weights = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return OverlapMapping(lower_units=lower_units, upper_units=upper_units, counts=counts, weights=weights)
