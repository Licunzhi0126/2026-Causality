from __future__ import annotations

import json
from pathlib import Path

from mignet_ce.downstream.perturbation.reporting import collect_summary, write_json


def test_write_json_converts_nested_paths_to_strings(tmp_path: Path) -> None:
    report = tmp_path / "result.json"
    training_root = tmp_path / "training"

    write_json(report, {"training": {"global_training": {"rows": [{"training_root": training_root}]}}})

    assert json.loads(report.read_text(encoding="utf-8"))["training"]["global_training"]["rows"][0]["training_root"] == str(training_root)


def test_collect_summary_requires_complete_consistent_global_metrics(tmp_path: Path) -> None:
    write_json(tmp_path / "summary.json", {"EI_micro_fixed": 1.0, "EI_macro_best_checkpoint": 3.0, "delta_EI_best_checkpoint": 2.0, "I_available": 2.0, "I_retained": 1.0, "closure_quality": 0.5, "Keff_source": 2.0, "Keff_target": 2.0, "hardK_t": 2, "hardK_tp": 2, "best_epoch": 4})
    summary = collect_summary(tmp_path)
    assert summary["best_epoch"] == 4
    assert summary["Keff_t"] == 2.0
