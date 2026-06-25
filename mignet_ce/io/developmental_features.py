from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.mapping import OverlapMapping

if TYPE_CHECKING:
    from mignet_ce.networks.base import NetworkContext


KNOWN_SCALAR_COLUMNS = ("pseudotime", "sr", "potency_score")


@dataclass
class DevelopmentalFeatureTable:
    values: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _feature_path(root: Path, layer: str, organ: str, stage: str) -> Path:
    return Path(root) / layer / f"{organ}_{stage}_features.csv"


def _velocity_sort_key(column: str) -> tuple[int, str]:
    match = re.search(r"velocity_(\d+)$", str(column))
    if match:
        return int(match.group(1)), str(column)
    return 10**9, str(column)


def velocity_columns(values: pd.DataFrame) -> list[str]:
    return sorted([str(col) for col in values.columns if str(col).startswith("velocity_")], key=_velocity_sort_key)


def require_columns(values: pd.DataFrame, columns: Sequence[str], method_name: str) -> None:
    missing = [column for column in columns if column not in values.columns]
    if missing:
        raise ValueError(
            f"{method_name} requires developmental feature column(s) {missing}; "
            f"available columns are {list(values.columns)}."
        )


def select_first_available_column(values: pd.DataFrame, candidates: Sequence[str], method_name: str) -> str:
    for column in candidates:
        if column in values.columns:
            return column
    raise ValueError(
        f"{method_name} requires one of developmental feature columns {list(candidates)}; "
        f"available columns are {list(values.columns)}."
    )


def _read_feature_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "unit_id" not in df.columns:
        raise ValueError(f"{path} must contain a 'unit_id' column.")
    df = df.copy()
    df["unit_id"] = df["unit_id"].astype(str)
    duplicated = df["unit_id"][df["unit_id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"{path} contains duplicated unit_id values, for example {duplicated[:5].tolist()}.")

    feature_columns = [
        column
        for column in df.columns
        if column in KNOWN_SCALAR_COLUMNS or str(column).startswith("velocity_")
    ]
    values = df.loc[:, ["unit_id"] + feature_columns].set_index("unit_id")
    for column in feature_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def _select_spot_feature_ids(mapping: pd.DataFrame, feature_index: set[str]) -> pd.Series:
    raw_ids = mapping["spot_id"].astype(str)
    if "organ" not in mapping.columns:
        return raw_ids
    prefixed_ids = mapping["organ"].astype(str) + "__" + raw_ids
    return pd.Series(
        [
            raw if raw in feature_index else prefixed if prefixed in feature_index else raw
            for raw, prefixed in zip(raw_ids, prefixed_ids)
        ],
        index=mapping.index,
    )


def _aggregate_spot_features_to_domains(
    spot_values: pd.DataFrame,
    spot_domain_map: Path,
    aggregation: str,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    if not spot_domain_map.exists():
        raise FileNotFoundError(f"Cannot aggregate spot-level developmental features; missing {spot_domain_map}.")
    mapping = pd.read_csv(spot_domain_map)
    if "spot_id" not in mapping.columns or "domain_id" not in mapping.columns:
        raise ValueError(f"{spot_domain_map} must contain 'spot_id' and 'domain_id'.")
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be one of ['mean', 'median'].")

    work = mapping.loc[:, [column for column in ("spot_id", "domain_id", "organ") if column in mapping.columns]].copy()
    work["domain_id"] = work["domain_id"].astype(str)
    work["_feature_unit_id"] = _select_spot_feature_ids(work, set(map(str, spot_values.index)))
    merged = work.merge(spot_values, left_on="_feature_unit_id", right_index=True, how="left")
    feature_columns = list(spot_values.columns)
    grouped = merged.groupby("domain_id", sort=False)[feature_columns]
    if aggregation == "median":
        aggregated = grouped.median()
    else:
        aggregated = grouped.mean()
    warnings: list[str] = []
    missing_spot_rows = int(merged[feature_columns].isna().all(axis=1).sum()) if feature_columns else 0
    if missing_spot_rows:
        warnings.append(f"{missing_spot_rows} spot rows had no developmental feature match before domain aggregation.")
    return (
        aggregated,
        {
            "aggregated_from_spot": True,
            "spot_domain_map": str(spot_domain_map),
            "aggregation": aggregation,
            "spot_rows": int(len(work)),
            "domain_rows": int(aggregated.shape[0]),
            "missing_spot_rows": missing_spot_rows,
        },
        warnings,
    )


def _format_missing_message(path: Path, missing_units: Sequence[str], missing_values: int) -> str:
    preview = list(missing_units[:10])
    suffix = f", for example {preview}" if preview else ""
    return f"{path} has missing developmental feature values: missing_units={len(missing_units)}{suffix}, missing_values={missing_values}."


def _align_and_apply_missing_policy(
    values: pd.DataFrame,
    units: Sequence[str],
    policy: str,
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    if policy not in {"error", "impute_mean", "ignore"}:
        raise ValueError("policy must be one of ['error', 'impute_mean', 'ignore'].")
    units = list(map(str, units))
    present = set(map(str, values.index))
    missing_units = [unit for unit in units if unit not in present]
    aligned = values.reindex(units)
    missing_values = int(aligned.isna().sum().sum())
    warnings: list[str] = []
    metadata = {
        "aligned_units": len(units),
        "missing_units": len(missing_units),
        "missing_values": missing_values,
        "missing_feature_policy": policy,
    }
    if policy == "error" and (missing_units or missing_values):
        raise ValueError(_format_missing_message(source_path, missing_units, missing_values))
    if policy == "impute_mean" and missing_values:
        means = aligned.mean(axis=0, skipna=True).fillna(0.0)
        aligned = aligned.fillna(means)
        metadata["imputed_values"] = missing_values
        warnings.append(f"Imputed {missing_values} missing developmental feature values with column means.")
    elif policy == "ignore" and (missing_units or missing_values):
        warnings.append(_format_missing_message(source_path, missing_units, missing_values))
    return aligned.astype(float), metadata, warnings


def load_developmental_features_for_layer(
    *,
    development_feature_root: Path,
    data_root: Path,
    layer: str,
    organ: str,
    stage: str,
    units: Sequence[str],
    aggregation: str = "mean",
    missing_policy: str = "error",
    spot_domain_map: Path | None = None,
) -> DevelopmentalFeatureTable:
    root = Path(development_feature_root)
    direct_path = _feature_path(root, layer, organ, str(stage))
    source_path = direct_path
    metadata: dict[str, object] = {
        "layer": layer,
        "organ": organ,
        "stage": str(stage),
        "feature_aggregation": aggregation,
        "missing_feature_policy": missing_policy,
    }
    warnings: list[str] = []

    if direct_path.exists():
        values = _read_feature_csv(direct_path)
        metadata.update({"feature_path": str(direct_path), "aggregated_from_spot": False})
    elif layer != "spot":
        spot_path = _feature_path(root, "spot", organ, str(stage))
        source_path = spot_path
        if not spot_path.exists():
            raise FileNotFoundError(f"Missing developmental feature file. Tried {direct_path} and {spot_path}.")
        if spot_domain_map is None:
            spot_domain_map = LayerDataResolver(Path(data_root)).paths(layer, organ, str(stage)).spot_domain_map
        spot_values = _read_feature_csv(spot_path)
        values, aggregation_metadata, aggregation_warnings = _aggregate_spot_features_to_domains(
            spot_values,
            spot_domain_map=Path(spot_domain_map),
            aggregation=aggregation,
        )
        metadata.update({"feature_path": str(spot_path), **aggregation_metadata})
        warnings.extend(aggregation_warnings)
    else:
        raise FileNotFoundError(f"Missing developmental feature file {direct_path}.")

    aligned, missing_metadata, missing_warnings = _align_and_apply_missing_policy(
        values=values,
        units=units,
        policy=missing_policy,
        source_path=source_path,
    )
    metadata.update(missing_metadata)
    metadata["feature_columns"] = list(aligned.columns)
    warnings.extend(missing_warnings)
    if warnings:
        metadata["warnings"] = list(warnings)
    return DevelopmentalFeatureTable(values=aligned, metadata=metadata, warnings=warnings)


def _nan_aware_overlap_mean(values: pd.DataFrame, overlap: OverlapMapping, target_units: Sequence[str]) -> pd.DataFrame:
    arr = values.to_numpy(dtype=float)
    counts = np.asarray(overlap.counts, dtype=float)
    if arr.shape[0] != counts.shape[0]:
        raise ValueError(f"Lower developmental feature rows {arr.shape[0]} do not match overlap rows {counts.shape[0]}.")
    if arr.shape[1] == 0:
        return pd.DataFrame(index=list(map(str, target_units)), columns=list(values.columns), dtype=float)
    finite = np.isfinite(arr)
    weighted = counts.T @ np.where(finite, arr, 0.0)
    denom = counts.T @ finite.astype(float)
    out = np.divide(weighted, denom, out=np.full_like(weighted, np.nan, dtype=float), where=denom > 0)
    return pd.DataFrame(out, index=list(map(str, overlap.upper_units)), columns=list(values.columns)).reindex(
        list(map(str, target_units))
    )


def load_developmental_features_for_pij(
    context: "NetworkContext",
    cfg: TemporalRunConfig,
    time_index: int,
    space: str,
) -> DevelopmentalFeatureTable:
    if cfg.development_feature_root is None:
        raise ValueError(f"{cfg.effective_pij_method()} requires development_feature_root.")
    if space not in {"lower", "upper"}:
        raise ValueError("space must be one of ['lower', 'upper'].")

    stage = str(context.time_points[time_index])
    resolver = LayerDataResolver(cfg.data_root)
    native_units = context.feature_alignment_space == "native_units"
    if space == "upper":
        layer = context.pair.upper_layer
        paths = resolver.paths(layer, context.organ, stage)
        units = context.upper_units_by_time[time_index] if native_units else context.stable_upper_units
        table = load_developmental_features_for_layer(
            development_feature_root=cfg.development_feature_root,
            data_root=cfg.data_root,
            layer=layer,
            organ=context.organ,
            stage=stage,
            units=units,
            aggregation=cfg.pij_feature_aggregation,
            missing_policy=cfg.pij_missing_feature_policy,
            spot_domain_map=paths.spot_domain_map,
        )
        if native_units:
            table.metadata.update(
                {
                    "space": "upper",
                    "aligned_to": "native_units",
                    "upper_layer": layer,
                }
            )
        return table

    layer = context.pair.lower_layer
    paths = resolver.paths(layer, context.organ, stage)
    lower_table = load_developmental_features_for_layer(
        development_feature_root=cfg.development_feature_root,
        data_root=cfg.data_root,
        layer=layer,
        organ=context.organ,
        stage=stage,
        units=context.lower_units_by_time[time_index],
        aggregation=cfg.pij_feature_aggregation,
        missing_policy=cfg.pij_missing_feature_policy,
        spot_domain_map=paths.spot_domain_map,
    )
    if native_units:
        lower_table.metadata.update(
            {
                "space": "lower",
                "aligned_to": "native_units",
                "lower_layer": layer,
                "overlap_aggregation": "none",
            }
        )
        return lower_table
    aggregated = _nan_aware_overlap_mean(
        lower_table.values,
        overlap=context.overlaps[time_index],
        target_units=context.stable_upper_units,
    )
    aligned, missing_metadata, missing_warnings = _align_and_apply_missing_policy(
        values=aggregated,
        units=context.stable_upper_units,
        policy=cfg.pij_missing_feature_policy,
        source_path=Path(str(lower_table.metadata.get("feature_path", cfg.development_feature_root))),
    )
    metadata = dict(lower_table.metadata)
    metadata.update(
        {
            "space": "lower",
            "aligned_to": "stable_upper_units",
            "lower_layer": layer,
            "overlap_aggregation": "spot_count_weighted_mean",
            **missing_metadata,
        }
    )
    warnings = list(lower_table.warnings) + list(missing_warnings)
    if warnings:
        metadata["warnings"] = warnings
    return DevelopmentalFeatureTable(values=aligned, metadata=metadata, warnings=warnings)
