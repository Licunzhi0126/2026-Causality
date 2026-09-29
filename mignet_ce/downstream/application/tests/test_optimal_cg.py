from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from mignet_ce.downstream.application.optimal_cg.reporting import collect_summary
from mignet_ce.downstream.application.optimal_cg.maturity_bridge import resolve_maturity
from mignet_ce.downstream.application.optimal_cg.trainer_bridge import REQUIRED_OUTPUTS, run_training
from mignet_ce.downstream.application.optimal_cg.validation import validate_maturity


def test_maturity_requires_exact_spot_coverage(tmp_path: Path) -> None:
    path = tmp_path / "maturity.csv"
    pd.DataFrame({"spot_id": ["a"], "maturity": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly cover"):
        validate_maturity(path, ["a", "b"])


def test_runner_bridge_uses_existing_two_stage_cli(tmp_path: Path) -> None:
    out_dir = tmp_path / "training"
    inputs = {name: tmp_path / f"{name}.txt" for name in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "index_t", "index_tp", "grn_t", "grn_tp")}
    for path in inputs.values():
        path.touch()
    maturity_t, maturity_tp = tmp_path / "mt.csv", tmp_path / "mtp.csv"
    maturity_t.touch(); maturity_tp.touch()
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        out_dir.mkdir()
        for name in REQUIRED_OUTPUTS:
            (out_dir / name).write_text("{}" if name == "summary.json" else "", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
    record = run_training(python="python", inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, out_dir=out_dir, runner=runner)
    assert "maturity_cci_grn_two_stage" in record["command"]
    assert record["command"][record["command"].index("--k") + 1] == "40"


def test_runner_summary_consistency(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {"EI_micro_fixed": 1.0, "EI_macro_best_checkpoint": 1.5, "delta_EI_best_checkpoint": 0.5, "I_available": 2.0, "I_retained": 1.0, "closure_quality": 0.5, "Keff_source": 3.0, "Keff_target": 4.0, "best_epoch": 12}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert collect_summary(tmp_path)["best_epoch"] == 12


def test_maturity_proxy_uses_numeric_stage_labels(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    _, _, record = resolve_maturity(
        source_sample="source", target_sample="target",
        h5ad_t=tmp_path / "source.h5ad", h5ad_tp=tmp_path / "target.h5ad",
        output_root=tmp_path / "out", supplied_root=None, runner=runner,
    )
    command = record["command"]
    assert command[command.index("--stage") + 1] == "7"
    assert "21" in command
