from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from mignet_ce.downstream.application.cli import run_optimal_cg
from mignet_ce.downstream.application.optimal_cg.reporting import collect_summary, write_json
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
    assert "formal_two_stage" in record["command"]
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from mignet_ce.coarse_frontends.method_specs import graph_configuration
    args = build_argparser().parse_args(record["command"][2:])
    assert args.alpha_cci == .20
    assert args.epochs == 1500 and args.seed == 42
    assert graph_configuration(args.method, local_graph_mode=args.local_graph_mode)["local_graph_mode"] == "all_features"
    # Leave stage controls and Formal-specific loss overrides to the shared resolver.
    assert args.stage1_ratio is None and args.lambda_dev is None and args.lambda_align is None
    assert "group_micro_analysis" not in record
    assert record["command"][record["command"].index("--k") + 1] == "40"


@pytest.mark.parametrize("checkpoint_args, expected", [([], None), (["--checkpoint-keff-min", "8"], 8.0)])
def test_cli_forwards_checkpoint_keff_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_args: list[str], expected: float | None,
) -> None:
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser

    training_root = tmp_path / "optimal_cg" / "training"

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        args = build_argparser().parse_args(command[2:])
        assert args.k == 40
        assert args.checkpoint_keff_min == expected
        assert ("--checkpoint-keff-min" in command) == (expected is not None)
        assert args.keff_min is None
        training_root.mkdir(parents=True)
        for name in REQUIRED_OUTPUTS:
            (training_root / name).touch()
        (training_root / "summary.json").write_text(json.dumps({
            "EI_micro_fixed": 1.0, "EI_macro_best_checkpoint": 1.5, "delta_EI_best_checkpoint": 0.5,
            "best_epoch": 12, "Keff_source": 8.0, "Keff_target": 8.0,
            "I_available": 2.0, "I_retained": 1.0, "closure_quality": 0.5,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(run_optimal_cg, "validate_run", lambda **_: None)
    monkeypatch.setattr(run_optimal_cg, "run_training", partial(run_training, runner=runner))
    argv = ["--output-root", str(tmp_path), "--skip-group-micro-analysis", *checkpoint_args]
    for key in run_optimal_cg.INPUT_KEYS:
        for stage in ("t", "tp"):
            argv.extend([f"--{key}-{stage}", str(tmp_path / f"{key}_{stage}")])
    run_optimal_cg.main(argv)


def test_runner_summary_consistency(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {"EI_micro_fixed": 1.0, "EI_macro_best_checkpoint": 1.5, "delta_EI_best_checkpoint": 0.5, "I_available": 2.0, "I_retained": 1.0, "closure_quality": 0.5, "Keff_source": 3.0, "Keff_target": 4.0, "best_epoch": 12}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert collect_summary(tmp_path)["best_epoch"] == 12


def test_write_json_serializes_nested_path(tmp_path: Path) -> None:
    output = tmp_path / "provenance.json"
    training_root = tmp_path / "local" / "training"
    write_json(output, {"trainer": {"group_micro_analysis": {"rows": [{"training_root": training_root}]}}})
    assert json.loads(output.read_text(encoding="utf-8"))["trainer"]["group_micro_analysis"]["rows"][0]["training_root"] == str(training_root)


def test_maturity_is_required_without_generation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="maturity"):
        resolve_maturity(supplied_root=tmp_path)
    assert not list(tmp_path.iterdir())

