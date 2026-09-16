from __future__ import annotations

"""Formal downstream analysis orchestration.

This module never imports visualization. It writes the complete analysis
contract to disk; figure code consumes those persisted tables separately.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .deltaei_contract import build_deltaei_contract_table
from .determinism_degeneracy.analysis import build_unified_ei_tables
from .dynamic_closure.analysis import (
    build_cross_representation_consistency,
    build_unified_closure_table,
)
from .fate_path.analysis import build_unified_fate_paths
from .grn_cci.analysis import build_unified_mechanism_table
from .mappings import load_all_mapping_records
from .null_model.analysis import (
    build_unified_matched_null,
    summarize_unified_matched_null,
)
from .preparation import ensure_full_unified_inputs
from .reporting import audit_unified_analysis_outputs, unified_run_manifest
from .spatial.analysis import (
    build_unified_effective_states,
    build_unified_spatial_metrics,
    build_unified_spatial_spots,
)


UNIFIED_TABLE_FILES = {
    "metrics": "01_metrics.csv",
    "states": "02_states.csv",
    "spatial_spots": "03_spatial_spots.csv",
    "closure": "04_dynamical_closure.csv",
    "spatial": "05_spatial_state_metrics.csv",
    "effective": "06_effective_states.csv",
    "mechanism": "07_grn_cci_mechanism.csv",
    "null_distribution": "08_matched_random_null_distribution.csv",
    "null_summary": "09_matched_random_null_summary.csv",
    "consistency": "10_cross_representation_consistency.csv",
    "fate": "11_macro_fate_paths.csv",
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_unified_analysis_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the persisted analysis contract without recomputing any metric."""

    root = Path(results_dir).resolve()
    tables_dir = root / "tables"
    return {
        name: pd.read_csv(tables_dir / filename)
        for name, filename in UNIFIED_TABLE_FILES.items()
    }


def run_unified_downstream_analysis(config) -> dict[str, Path]:
    """Compute the formal five-representation, soft-native analysis tables."""

    cfg = config.normalized()
    cfg.validate()
    output_dir = cfg.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Formal output directory is not empty: {output_dir}. "
            "Use a new output directory; existing downstream results are never overwritten."
        )
    tables_dir = output_dir / "tables"
    audit_dir = output_dir / "audit"

    ensure_full_unified_inputs(cfg)
    records = load_all_mapping_records(cfg)
    expected_records = len(cfg.mapping_names) * len(cfg.adjacent_pairs)
    if len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} mapping records, loaded {len(records)}"
        )

    metrics, states = build_unified_ei_tables(cfg, records)
    deltaei_contract = build_deltaei_contract_table(cfg, records, metrics, fail=True)
    null_distribution = build_unified_matched_null(cfg, records)
    tables = {
        "metrics": metrics,
        "states": states,
        "spatial_spots": build_unified_spatial_spots(cfg, records),
        "closure": build_unified_closure_table(cfg, records),
        "spatial": build_unified_spatial_metrics(cfg, records),
        "effective": build_unified_effective_states(cfg, records),
        "mechanism": build_unified_mechanism_table(cfg, records),
        "null_distribution": null_distribution,
        "null_summary": summarize_unified_matched_null(null_distribution),
        "consistency": build_cross_representation_consistency(cfg, records),
        "fate": build_unified_fate_paths(cfg, records),
    }
    for name, frame in tables.items():
        _write_csv(frame, tables_dir / UNIFIED_TABLE_FILES[name])

    deltaei_contract_path = audit_dir / "deltaei_contract.csv"
    _write_csv(deltaei_contract, deltaei_contract_path)
    checks = audit_unified_analysis_outputs(
        cfg,
        tables,
        deltaei_contract=deltaei_contract,
    )
    checks_path = audit_dir / "analysis_validation_checks.csv"
    _write_csv(checks, checks_path)
    manifest = unified_run_manifest(
        cfg,
        tables,
        deltaei_contract=deltaei_contract,
    )
    manifest["validation_passed"] = bool(checks["passed"].all())
    manifest["table_files"] = UNIFIED_TABLE_FILES
    manifest_path = audit_dir / "analysis_manifest.json"
    _write_json(manifest, manifest_path)
    if not bool(checks["passed"].all()):
        failures = checks.loc[~checks["passed"], ["check", "detail"]].to_dict(orient="records")
        raise RuntimeError(f"Unified downstream analysis audit failed: {failures}")
    return {
        "output_dir": output_dir,
        "tables_dir": tables_dir,
        "validation": checks_path,
        "deltaei_contract": deltaei_contract_path,
        "manifest": manifest_path,
    }
