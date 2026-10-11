"""Write perturbation provenance tables without recomputing causal metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_FIELDS = (
    "EI_micro_fixed", "EI_macro_best_checkpoint", "delta_EI_best_checkpoint",
    "I_available", "I_retained", "closure_quality", "Keff_source",
    "Keff_target", "hardK_t", "hardK_tp", "best_epoch",
)


def collect_summary(training_root: Path) -> dict[str, object]:
    """Read and validate metrics emitted by the formal two-stage trainer."""
    summary_path = Path(training_root) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = [key for key in SUMMARY_FIELDS if key not in summary]
    if missing:
        raise ValueError(f"Two-stage summary lacks required fields: {missing}")
    if abs(float(summary["delta_EI_best_checkpoint"]) - (float(summary["EI_macro_best_checkpoint"]) - float(summary["EI_micro_fixed"]))) > 1e-5:
        raise ValueError("Two-stage delta_EI_best_checkpoint is inconsistent with EI values.")
    if float(summary["I_available"]) > 1e-12 and abs(float(summary["closure_quality"]) - float(summary["I_retained"]) / float(summary["I_available"])) > 1e-5:
        raise ValueError("Two-stage closure_quality is inconsistent with retained information.")
    # Preserve formal names and expose the T/T2 spelling used by perturbation tables.
    return {**summary, "Keff_t": summary["Keff_source"], "Keff_tp": summary["Keff_target"]}


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: object) -> object:
    """Convert filesystem paths in provenance records to portable strings."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_tables(root: Path, records: Iterable[dict[str, object]]) -> tuple[Path, Path]:
    rows = list(records)
    root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    result_path = root / "perturbation_results.csv"
    results.to_csv(result_path, index=False)
    # The stable generic name is consumed by the P6–P8 analysis; retain the
    # historical filename so existing P1–P5 notebooks keep working.
    results.to_csv(root / "all_conditions.csv", index=False)
    qc_columns = [column for column in results.columns if column.startswith(("same_", "mirror_", "untouched_", "nonnegative_", "finite_", "no_self_", "selected_", "region_", "node_region_coverage", "edge_region_coverage", "overall_", "topology_", "budget_", "support_", "total_weight_", "modified_", "weight_transfer_"))]
    qc_path = root / "topology_qc.csv"
    results.loc[:, [column for column in ("time_pair", "pair_id", "strength", "direction") if column in results] + qc_columns].to_csv(qc_path, index=False)
    results.loc[:, [column for column in ("time_pair", "pair_id", "strength", "direction") if column in results] + qc_columns].to_csv(root / "symmetry_qc.csv", index=False)
    return result_path, qc_path


