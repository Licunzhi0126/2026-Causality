from __future__ import annotations

import pytest
def require_formal_trainer():
    try:
        from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig
    except ImportError as error:
        pytest.fail(f"UPSTREAM_IMPORT_BLOCKER: {error}")
    globals()["WYTFormalTwoStageConfig"] = WYTFormalTwoStageConfig

import json
from pathlib import Path

import pytest

from mignet_ce.downstream.perturbation import optimal_bridge, runner
from mignet_ce.downstream.perturbation.config import PerturbationConfig
from mignet_ce.downstream.perturbation.paths import ExperimentPaths
from mignet_ce.downstream.perturbation.reporting import write_json


def _cache(tmp_path):
    require_formal_trainer()
    paths = ExperimentPaths(tmp_path, "test_pair")
    training = paths.baseline / "training"
    training.mkdir(parents=True)
    for name in optimal_bridge.REQUIRED_OUTPUTS:
        (training / name).write_text("original", encoding="utf-8")
    record = {"method": "formal_two_stage", "loss_protocol": "formal_two_stage_main",
              "k": 2, "device": "cpu", "local_graph_mode": "all_features", "perturbation_macro_schema": optimal_bridge.PERTURBATION_MACRO_SCHEMA, "frontend_options": {}, **optimal_bridge.training_config_record(
                  optimal_bridge.formal_training_config(k=2, training=training, device="cpu"))}
    return paths, record


@pytest.mark.parametrize("case", ["matching", "legacy", "protocol", "k", "config", "fingerprint",
                                  "no_snapshot", "graph_mode", "no_mode", "no_method", "no_hash", "frontend", "no_provenance", "invalid_json", "incomplete"])
def test_baseline_cache_is_strict_and_never_retrains(monkeypatch, tmp_path, case):
    paths, record = _cache(tmp_path)
    if case == "legacy":
        record["method"] = "maturity_cci_grn_two_stage"
    elif case == "protocol":
        record.pop("loss_protocol")
    elif case == "k":
        record["k"] = 3
    elif case == "config":
        record["training_config"]["lr"] = 99
    elif case == "fingerprint":
        record["training_config_sha256"] = "bad"
    elif case == "no_snapshot":
        record.pop("training_config")
    if case == "graph_mode":
        record["local_graph_mode"] = "legacy_features"
        record.update(optimal_bridge.training_config_record(optimal_bridge.formal_training_config(
            k=2, training=paths.baseline / "training", device="cpu",
            training_options={"local_graph_mode": "legacy_features"})))
    elif case in {"no_mode", "no_method", "no_hash"}:
        record.pop({"no_mode": "local_graph_mode", "no_method": "method", "no_hash": "training_config_sha256"}[case])
    elif case == "frontend":
        record["frontend_options"] = {"alpha_cci": 0.2}
    if case != "no_provenance":
        write_json(paths.baseline / "perturbation_provenance.json", record)
    if case == "invalid_json":
        (paths.baseline / "perturbation_provenance.json").write_text("{", encoding="utf-8")
    if case == "incomplete":
        # Simulate an interrupted run using an empty directory, without deleting artifacts.
        paths = ExperimentPaths(tmp_path, "interrupted")
        paths.baseline.mkdir(parents=True)
    before = {str(p): p.read_bytes() for p in paths.baseline.rglob("*") if p.is_file()}
    def forbidden(**kwargs):
        pytest.fail("cached baseline must never silently retrain")
    monkeypatch.setattr(runner, "run_direct_state_baseline", forbidden)
    monkeypatch.setattr(runner, "collect_summary", lambda path: {"cached": True})
    kwargs = dict(paths=paths, inputs={}, source_sample="s", target_sample="t",
                  config=PerturbationConfig(k=2, device="cpu"), supplied_maturity_root=None)
    if case == "matching":
        assert runner._baseline_summary(**kwargs) == {"cached": True}
    else:
        with pytest.raises(ValueError, match="--output-root"):
            runner._baseline_summary(**kwargs)
    assert before == {str(p): p.read_bytes() for p in paths.baseline.rglob("*") if p.is_file()}


def test_condition_rejects_old_training_before_preparation(monkeypatch, tmp_path):
    paths, record = _cache(tmp_path)
    write_json(paths.baseline / "perturbation_provenance.json", {**record, "method": "old"})
    def forbidden(**kwargs):
        pytest.fail("existing condition must be rejected before preparation")
    monkeypatch.setattr(optimal_bridge, "prepare_direct_state_two_stage", forbidden)
    with pytest.raises(ValueError, match="--output-root"):
        optimal_bridge.run_direct_state_condition(baseline_inputs={},
            maturity_t=tmp_path, maturity_tp=tmp_path, output_root=paths.baseline,
            spot_ids_t=[], spot_ids_tp=[], k=2, device="cpu")
