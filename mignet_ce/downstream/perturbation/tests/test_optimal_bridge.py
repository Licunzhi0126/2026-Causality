from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

from mignet_ce.downstream.perturbation import optimal_bridge


def _summary() -> dict[str, object]:
    return {"delta_EI_best_checkpoint": 2.0, "EI_micro_fixed": 1.0,
            "EI_macro_best_checkpoint": 3.0, "I_available": 1.0,
            "I_retained": 1.0, "closure_quality": 1.0, "Keff_source": 1.0,
            "Keff_target": 1.0, "best_epoch": 1}


def _inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = {name: tmp_path / f"{name}.txt" for name in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "index_t", "index_tp", "grn_t", "grn_tp")}
    for path in inputs.values():
        path.write_text("x", encoding="utf-8")
    return inputs


def _mock_global_training(calls: dict[str, object]):
    def mock_training(**kwargs):
        calls.setdefault("trained", []).append(kwargs)
        kwargs["out_dir"].mkdir(parents=True)
        for name in optimal_bridge.REQUIRED_OUTPUTS:
            (kwargs["out_dir"] / name).write_text("x", encoding="utf-8")
        return {"returncode": 0}
    return mock_training


def test_direct_state_baseline_runs_one_global_trainer_without_application(monkeypatch, tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    maturity_t, maturity_tp = tmp_path / "source.csv", tmp_path / "target.csv"
    maturity_t.write_text("unit_id,pseudotime\na,1\n", encoding="utf-8"); maturity_tp.write_text("unit_id,pseudotime\nb,1\n", encoding="utf-8")
    calls: dict[str, object] = {}
    monkeypatch.setattr(optimal_bridge, "prepare_direct_state_two_stage", lambda **kwargs: object())
    def train(prepared, config):
        calls.setdefault("trained", []).append((prepared, config))
        config.out_dir.mkdir(parents=True)
        for name in optimal_bridge.REQUIRED_OUTPUTS:
            (config.out_dir / name).write_text("x", encoding="utf-8")
        return SimpleNamespace(best_epoch=1)
    monkeypatch.setattr("wyt_deltaei_coarse_grain.train_deltaei", train)
    monkeypatch.setattr(optimal_bridge, "collect_summary", lambda path: _summary())
    result = optimal_bridge.run_direct_state_baseline(baseline_inputs=inputs, g_raw_t=np.zeros((1, 2)), g_raw_tp=np.zeros((1, 2)), source_sample="s", target_sample="t", output_root=tmp_path / "out", spot_ids_t=["a"], spot_ids_tp=["b"], k=2, device="cpu", supplied_maturity_root=tmp_path)
    assert len(calls["trained"]) == 1
    assert (tmp_path / "out" / "perturbation_provenance.json").is_file()
    assert result["summary"]["delta_EI_best_checkpoint"] == 2.0


def test_direct_state_condition_runs_one_global_trainer(monkeypatch, tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    maturity_t, maturity_tp = tmp_path / "source.csv", tmp_path / "target.csv"
    maturity_t.write_text("unit_id,pseudotime\na,1\n", encoding="utf-8"); maturity_tp.write_text("unit_id,pseudotime\nb,1\n", encoding="utf-8")
    calls: dict[str, object] = {}
    monkeypatch.setattr(optimal_bridge, "prepare_direct_state_two_stage", lambda **kwargs: object())
    def train(prepared, config):
        calls.setdefault("trained", []).append((prepared, config))
        config.out_dir.mkdir(parents=True)
        for name in optimal_bridge.REQUIRED_OUTPUTS:
            (config.out_dir / name).write_text("x", encoding="utf-8")
        return SimpleNamespace(best_epoch=1)
    monkeypatch.setattr("wyt_deltaei_coarse_grain.train_deltaei", train)
    monkeypatch.setattr(optimal_bridge, "collect_summary", lambda path: _summary())
    result = optimal_bridge.run_direct_state_condition(baseline_inputs=inputs, g_raw_t=np.zeros((1, 2)), g_raw_tp=np.zeros((1, 2)), maturity_t=maturity_t, maturity_tp=maturity_tp, output_root=tmp_path / "out", spot_ids_t=["a"], spot_ids_tp=["b"], k=2, device="cpu")
    assert len(calls["trained"]) == 1
    assert result["training_root"] == str(tmp_path / "out" / "training")


