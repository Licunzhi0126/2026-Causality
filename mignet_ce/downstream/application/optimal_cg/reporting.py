"""Read runner-owned metrics and write only application-level summaries."""

from __future__ import annotations

import json
from pathlib import Path


def collect_summary(training_root: Path) -> dict[str, object]:
    summary = json.loads((training_root / "summary.json").read_text(encoding="utf-8"))
    required = ("EI_micro_fixed", "EI_macro_best_checkpoint", "delta_EI_best_checkpoint", "I_available", "I_retained", "closure_quality", "Keff_source", "Keff_target", "best_epoch")
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"WYT summary lacks required fields: {missing}")
    if abs(float(summary["delta_EI_best_checkpoint"]) - (float(summary["EI_macro_best_checkpoint"]) - float(summary["EI_micro_fixed"]))) > 1e-5:
        raise ValueError("WYT delta_EI_best_checkpoint is inconsistent with its EI values.")
    if float(summary["I_available"]) > 1e-12 and abs(float(summary["closure_quality"]) - float(summary["I_retained"]) / float(summary["I_available"])) > 1e-5:
        raise ValueError("WYT closure_quality is inconsistent with retained/available information.")
    return summary


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

