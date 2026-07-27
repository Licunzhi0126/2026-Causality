from __future__ import annotations

"""Read and align H5AD regulatory-activity blocks for WYT RegSim.

Migration sources:
  - reference/network_only_coarse_grain/extract_grn_obs_features_v48.py
  - report: output/report/7.27 和wyt交接.pdf, sections 2.2, 2.3 and 3

Changes from the WYT source:
  - supports both ``Regulon - Gene`` and Seurat's ``Regulon...Gene`` names;
  - keeps explicit raw-to-canonical column provenance;
  - aligns spots/domains to the formal H5AD and CCI index instead of trusting
    equal matrix sizes;
  - aggregates spot regulatory activity to domains through obs["domain_id"].
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegSimFeatureBlock:
    unit_ids: list[str]
    feature_names: list[str]
    raw_column_names: list[str]
    values: np.ndarray
    unit_h5ad_path: Path
    activity_h5ad_path: Path
    metadata: dict[str, object]


def canonicalize_regulatory_column(column: object) -> str | None:
    text = str(column).strip()
    if text.startswith("Module_") and len(text) > len("Module_"):
        return text
    match = re.match(r"^Regulon(?:\s*-\s*|\.+)(.+?)\s*$", text)
    if match is None:
        return None
    name = match.group(1).strip(" .-")
    return f"Regulon::{name}" if name else None


def regulatory_column_map(columns: Sequence[object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_column in columns:
        canonical = canonicalize_regulatory_column(raw_column)
        if canonical is None:
            continue
        raw = str(raw_column)
        previous = mapping.get(canonical)
        if previous is not None and previous != raw:
            raise ValueError(
                f"Regulatory columns {previous!r} and {raw!r} both canonicalize to {canonical!r}."
            )
        mapping[canonical] = raw
    return mapping


def discover_regulatory_columns(h5ad_path: Path) -> dict[str, str]:
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"RegSim activity H5AD does not exist: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        return regulatory_column_map(list(adata.obs.columns))
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def common_regulatory_features(h5ad_paths: Sequence[Path]) -> list[str]:
    if not h5ad_paths:
        raise ValueError("At least one H5AD path is required to find common RegSim features.")
    mappings = [discover_regulatory_columns(path) for path in h5ad_paths]
    common = set(mappings[0])
    for mapping in mappings[1:]:
        common.intersection_update(mapping)
    ordered = sorted(common)
    if not ordered:
        raise ValueError(
            "No common Module_* or Regulon columns were found across RegSim H5AD inputs: "
            + ", ".join(map(str, h5ad_paths))
        )
    return ordered


def _read_obs_activity(
    h5ad_path: Path,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, str]]:
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"RegSim activity H5AD does not exist: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
        obs.index = pd.Index(adata.obs_names.astype(str), name=adata.obs_names.name)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    column_map = regulatory_column_map(list(obs.columns))
    missing = [name for name in feature_names if name not in column_map]
    if missing:
        raise ValueError(f"{path} is missing canonical RegSim columns {missing[:10]}.")
    raw_columns = [column_map[name] for name in feature_names]
    activity = obs.loc[:, raw_columns].copy()
    activity.columns = list(feature_names)
    for column in activity.columns:
        activity[column] = pd.to_numeric(activity[column], errors="coerce")
    activity = activity.fillna(0.0).astype(np.float32)
    return obs, activity, column_map


def _read_formal_units(unit_h5ad_path: Path) -> list[str]:
    path = Path(unit_h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"Formal unit H5AD does not exist: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.obs_names.astype(str).tolist()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def build_regsim_feature_block(
    *,
    unit_h5ad_path: Path,
    cci_unit_ids: Sequence[str],
    feature_names: Sequence[str],
    spots_with_domain_h5ad_path: Path | None = None,
    domain_id_column: str = "domain_id",
) -> RegSimFeatureBlock:
    formal_units = _read_formal_units(unit_h5ad_path)
    target_units = list(map(str, cci_unit_ids))
    if len(target_units) != len(set(target_units)):
        raise ValueError("CCI index contains duplicate unit IDs.")
    formal_set = set(formal_units)
    missing_formal = [unit for unit in target_units if unit not in formal_set]
    if missing_formal:
        raise ValueError(
            f"CCI units are missing from formal H5AD {unit_h5ad_path}: {missing_formal[:10]}"
        )

    activity_path = (
        Path(spots_with_domain_h5ad_path)
        if spots_with_domain_h5ad_path is not None
        else Path(unit_h5ad_path)
    )
    obs, activity, column_map = _read_obs_activity(activity_path, feature_names)
    if spots_with_domain_h5ad_path is None:
        missing_activity = [unit for unit in target_units if unit not in activity.index]
        if missing_activity:
            raise ValueError(
                f"CCI spot units are missing from RegSim H5AD {activity_path}: {missing_activity[:10]}"
            )
        aligned = activity.reindex(target_units)
        aggregation = "none_spot_level"
        source_spot_count = int(len(activity))
        source_domain_count = None
    else:
        if domain_id_column not in obs.columns:
            raise ValueError(
                f"Domain RegSim requires obs[{domain_id_column!r}] in {activity_path}."
            )
        domain_ids = obs[domain_id_column].astype(str)
        grouped = activity.assign(__domain_id=domain_ids.to_numpy()).groupby(
            "__domain_id",
            sort=False,
        ).mean()
        missing_domains = [unit for unit in target_units if unit not in grouped.index]
        if missing_domains:
            raise ValueError(
                f"CCI/formal domain IDs are missing from {activity_path} obs[{domain_id_column!r}]: "
                f"{missing_domains[:10]}"
            )
        aligned = grouped.reindex(target_units)
        aggregation = "mean_spot_activity_by_domain_id"
        source_spot_count = int(len(activity))
        source_domain_count = int(grouped.shape[0])

    values = np.array(aligned.to_numpy(dtype=np.float32), dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = 0.0
    raw_names = [str(column_map[name]) for name in feature_names]
    return RegSimFeatureBlock(
        unit_ids=target_units,
        feature_names=list(map(str, feature_names)),
        raw_column_names=raw_names,
        values=values,
        unit_h5ad_path=Path(unit_h5ad_path),
        activity_h5ad_path=activity_path,
        metadata={
            "unit_h5ad_path": str(unit_h5ad_path),
            "activity_h5ad_path": str(activity_path),
            "aggregation": aggregation,
            "domain_id_column": domain_id_column if spots_with_domain_h5ad_path is not None else None,
            "formal_unit_count": int(len(formal_units)),
            "cci_unit_count": int(len(target_units)),
            "source_spot_count": source_spot_count,
            "source_domain_count": source_domain_count,
            "feature_shape": list(values.shape),
            "raw_columns": raw_names,
            "canonical_columns": list(map(str, feature_names)),
        },
    )
