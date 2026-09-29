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
        rows = expression.obs.copy()
        rows = rows.reset_index(drop=True)
        rows["spot_id"] = spot_ids
        rows["unit_id"] = expression.units
        coords = expression.coords.reset_index(drop=True)
        for col in ("x", "y"):
            if col not in rows.columns and col in coords.columns:
                rows[col] = coords[col].to_numpy()
        return UnitAssignments(layer=layer, rows=rows)

    if spot_domain_map is None:
        raise ValueError(f"Layer {layer} needs a spot-domain map.")
    mapping = pd.read_csv(spot_domain_map)
    if "spot_id" not in mapping.columns or "domain_id" not in mapping.columns:
        raise ValueError(f"{spot_domain_map} must contain 'spot_id' and 'domain_id'.")
    rows = mapping.copy()
    rows["spot_id"] = rows["spot_id"].astype(str)
    if "organ" in mapping.columns:
        rows["spot_id"] = mapping["organ"].astype(str) + "__" + rows["spot_id"]
    rows["unit_id"] = rows["domain_id"].astype(str)
    return UnitAssignments(layer=layer, rows=rows.drop_duplicates(subset=["spot_id", "unit_id"]))


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


def build_spot_correspondence_table(
    lower: UnitAssignments,
    upper: UnitAssignments,
    stage: str,
    lower_layer: str,
    upper_layer: str,
) -> pd.DataFrame:
    left = lower.rows.rename(columns={"unit_id": "lower_unit"}).copy()
    right = upper.rows.rename(columns={"unit_id": "upper_unit"}).copy()
    merged = left.merge(right, on="spot_id", how="inner", suffixes=("_lower", "_upper"))
    merged.insert(0, "stage", str(stage))
    merged.insert(2, "lower_layer", lower_layer)
    merged.insert(3, "upper_layer", upper_layer)
    keep = ["stage", "spot_id", "lower_layer", "upper_layer", "lower_unit", "upper_unit"]
    for base in ("organ", "annotation", "x", "y"):
        for cand in (base, f"{base}_lower", f"{base}_upper"):
            if cand in merged.columns and base not in keep:
                merged[base] = merged[cand]
                keep.append(base)
                break
    extra = [col for col in merged.columns if col not in keep and not col.endswith(("_lower", "_upper"))]
    return merged.loc[:, keep + extra]


def build_overlap_edge_table(
    overlap: OverlapMapping,
    stage: str,
    lower_layer: str,
    upper_layer: str,
) -> pd.DataFrame:
    counts = np.asarray(overlap.counts, dtype=float)
    lower_sizes = counts.sum(axis=1)
    upper_sizes = counts.sum(axis=0)
    primary_upper_idx = np.argmax(counts, axis=1) if counts.size else np.array([], dtype=int)
    rows: list[dict[str, object]] = []
    for i, lower_unit in enumerate(overlap.lower_units):
        for j, upper_unit in enumerate(overlap.upper_units):
            value = float(counts[i, j])
            if value <= 0:
                continue
            denom = lower_sizes[i] + upper_sizes[j] - value
            rows.append(
                {
                    "stage": str(stage),
                    "lower_layer": lower_layer,
                    "upper_layer": upper_layer,
                    "lower_unit": lower_unit,
                    "upper_unit": upper_unit,
                    "overlap_spot_count": value,
                    "lower_unit_spot_count": float(lower_sizes[i]),
                    "upper_unit_spot_count": float(upper_sizes[j]),
                    "lower_to_upper_weight": value / lower_sizes[i] if lower_sizes[i] > 0 else 0.0,
                    "upper_to_lower_weight": value / upper_sizes[j] if upper_sizes[j] > 0 else 0.0,
                    "jaccard": value / denom if denom > 0 else 0.0,
                    "is_primary_match": bool(j == primary_upper_idx[i]),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["n_lower_units"] = len(overlap.lower_units)
    out.attrs["n_upper_units"] = len(overlap.upper_units)
    out.attrs["matched_lower_units"] = int((lower_sizes > 0).sum())
    out.attrs["matched_upper_units"] = int((upper_sizes > 0).sum())
    out.attrs["unmatched_lower_units"] = int((lower_sizes <= 0).sum())
    out.attrs["unmatched_upper_units"] = int((upper_sizes <= 0).sum())
    return out


def summarize_overlap_quality(edge_table: pd.DataFrame) -> dict[str, object]:
    if edge_table.empty:
        return {
            "n_lower_units": int(edge_table.attrs.get("n_lower_units", 0)),
            "n_upper_units": int(edge_table.attrs.get("n_upper_units", 0)),
            "n_overlap_edges": 0,
            "matched_lower_units": int(edge_table.attrs.get("matched_lower_units", 0)),
            "matched_upper_units": int(edge_table.attrs.get("matched_upper_units", 0)),
            "unmatched_lower_units": int(edge_table.attrs.get("unmatched_lower_units", 0)),
            "unmatched_upper_units": int(edge_table.attrs.get("unmatched_upper_units", 0)),
            "mean_primary_weight": 0.0,
            "median_primary_weight": 0.0,
            "mean_jaccard_of_primary": 0.0,
            "ambiguous_lower_units": 0,
            "low_confidence_lower_units": 0,
        }
    lower_sizes = edge_table.groupby("lower_unit")["lower_unit_spot_count"].max()
    upper_sizes = edge_table.groupby("upper_unit")["upper_unit_spot_count"].max()
    primary = edge_table[edge_table["is_primary_match"].astype(bool)].copy()
    return {
        "n_lower_units": int(edge_table.attrs.get("n_lower_units", lower_sizes.shape[0])),
        "n_upper_units": int(edge_table.attrs.get("n_upper_units", upper_sizes.shape[0])),
        "n_overlap_edges": int(edge_table.shape[0]),
        "matched_lower_units": int(edge_table.attrs.get("matched_lower_units", edge_table["lower_unit"].nunique())),
        "matched_upper_units": int(edge_table.attrs.get("matched_upper_units", edge_table["upper_unit"].nunique())),
        "unmatched_lower_units": int(edge_table.attrs.get("unmatched_lower_units", 0)),
        "unmatched_upper_units": int(edge_table.attrs.get("unmatched_upper_units", 0)),
        "mean_primary_weight": float(primary["lower_to_upper_weight"].mean()) if not primary.empty else 0.0,
        "median_primary_weight": float(primary["lower_to_upper_weight"].median()) if not primary.empty else 0.0,
        "mean_jaccard_of_primary": float(primary["jaccard"].mean()) if not primary.empty else 0.0,
        "ambiguous_lower_units": int((primary["lower_to_upper_weight"] < 0.5).sum()) if not primary.empty else 0,
        "low_confidence_lower_units": int((primary["jaccard"] < 0.2).sum()) if not primary.empty else 0,
    }
