"""CQ-only scientific behavior and native artifact integration on small fixtures."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_wyt_two_stage import _prepared, _cli_args
from wyt_deltaei_coarse_grain import WYTCQOnlyConfig, train_deltaei
from wyt_deltaei_coarse_grain.cq_only_objective import (
    cq_checkpoint_eligible, cq_low_signal_checkpoint_eligible, cq_only_loss,
)


def test_cq_loss_has_no_delta_or_keff_gradient(tmp_path):
    values = {key: torch.tensor(.1, requires_grad=True)
              for key in ("align", "var", "local", "sharp", "dev", "proto", "within", "inter", "closure",
                          "I_available", "I_retained", "delta", "keff")}
    loss, _ = cq_only_loss(values, WYTCQOnlyConfig(2, tmp_path), .01)
    loss.backward()
    assert values["delta"].grad is None and values["keff"].grad is None
    assert values["I_retained"].grad < 0


@pytest.mark.parametrize("competing_fallback", [False, True])
def test_cq_only_trains_one_stage_and_exports_negative_delta(tmp_path, monkeypatch, competing_fallback):
    if competing_fallback:
        # Simulate an additional saved fallback candidate. A strict checkpoint
        # must retain priority, including when its measured DeltaEI is negative.
        import wyt_deltaei_coarse_grain.cq_only_trainer as trainer
        monkeypatch.setattr(trainer, "cq_low_signal_checkpoint_eligible", lambda **_: True)
    prepared = _prepared("single_stage_cq_only")
    # Uniform macro PIJ guarantees non-positive DeltaEI; selection must still use CQ.
    def uniform_macro(inputs):
        logits = inputs.z_macro_t @ inputs.z_macro_tp.T
        return torch.softmax(logits * 0, 1)
    from mignet_ce.metrics import effective_information
    prepared = replace(prepared, macro_pij_builder=uniform_macro,
                       micro_ei=effective_information(prepared.micro_pij.copy()))
    cfg = WYTCQOnlyConfig(2, tmp_path / "cq", hidden_dim=8, mid_dim=4, epochs=4,
                          knn_k=2, temperature=.03, seed=42, log_every=1)
    result = train_deltaei(prepared, cfg)
    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary["stage1_epochs"] == 4 and summary["stage2_epochs"] == 0
    assert summary["trained_delta_EI"] < 0
    assert {row["stage"] for row in result.metrics} == {"cq"}
    assert summary["selected_checkpoint"] == "best_cq.pt"
    assert summary["strict_checkpoint_found"] is True
    assert summary["checkpoint_fallback_used"] is False
    assert summary["checkpoint_source"] == "best_cq.pt"
    assert summary["fallback_type"] is None and summary["fallback_reason"] is None
    if competing_fallback:
        assert (result.out_dir / "best_low_signal_cq.pt").is_file()
    assert (result.out_dir / "best_cq.pt").is_file() and (result.out_dir / "best_model.pt").is_file()
    assert not (result.out_dir / "best_joint.pt").exists()
    eligible = [row for row in result.metrics if row["checkpoint_eligible"]]
    assert result.best_epoch == max(eligible, key=lambda row: row["CQ"])["epoch"]
    for name in ("S_t", "S_tp", "PIJ_micro_train", "PIJ_macro_train"):
        matrix = np.load(result.out_dir / (name + ".npy"))
        assert np.isfinite(matrix).all() and np.allclose(matrix.sum(1), 1, atol=1e-5)
    with pytest.raises(FileExistsError):
        train_deltaei(prepared, cfg)


def test_cq_low_signal_fallback_saves_and_restores_best_training_score(tmp_path):
    from mignet_ce.metrics import effective_information
    from mignet_ce.information_thresholds import closure_signal_threshold_bits
    from mignet_ce.downstream.generalization_audit.tables import induced_cq_from_saved, inspect_cq_checkpoint_selection
    prepared = _prepared("single_stage_cq_only")
    micro = .05 * prepared.micro_pij + .95 / prepared.micro_pij.shape[1]
    prepared = replace(prepared, micro_pij=micro, micro_ei=effective_information(micro.copy()))
    assert prepared.micro_ei < closure_signal_threshold_bits(2)
    result = train_deltaei(prepared, WYTCQOnlyConfig(
        2, tmp_path / "cq", hidden_dim=8, mid_dim=4, epochs=4, knn_k=2, temperature=.03,
    ))
    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert not any(row["checkpoint_eligible"] for row in result.metrics)
    candidates = [row for row in result.metrics if row["low_signal_checkpoint_eligible"]]
    best = max(candidates, key=lambda row: row["CQ"])
    assert result.best_epoch == best["epoch"]
    assert summary["CQ_training_score"] == pytest.approx(best["CQ"])
    assert summary["strict_checkpoint_found"] is False and summary["checkpoint_fallback_used"] is True
    assert summary["fallback_type"] == "low_signal_cq"
    assert summary["checkpoint_source"] == "best_low_signal_cq.pt"
    assert summary["checkpoint_signal_status"] == "low-signal"
    assert summary["checkpoint_metric_state_consistent"] is True
    assert 0 < summary["I_available"] < summary["signal_threshold_bits"]
    assert summary["closure_quality"] > summary["CQ_training_score"]
    checkpoints = [torch.load(result.out_dir / name, map_location="cpu")
                   for name in ("best_cq.pt", "best_low_signal_cq.pt", "best_model.pt")]
    for checkpoint in checkpoints:
        assert checkpoint["epoch"] == best["epoch"] and checkpoint["selection"] == "low_signal_cq"
        for key, value in checkpoints[0]["encoder"].items():
            torch.testing.assert_close(checkpoint["encoder"][key], value)
    budget = induced_cq_from_saved(result.out_dir)
    audited = inspect_cq_checkpoint_selection(summary, available=budget["I_available_bits"],
        retained=budget["I_retained_bits"], k=2, run_dir=result.out_dir)
    assert audited["checkpoint_fallback_used"] is True


def test_cq_signal_predicates_cover_reported_k40_failure():
    from mignet_ce.information_thresholds import closure_signal_threshold_bits
    values = dict(available=.048295, retained=.035565, cq=.668274,
                  signal_threshold=closure_signal_threshold_bits(40), keff_t=13.324259, keff_tp=27.326002)
    assert not cq_checkpoint_eligible(**values)
    assert cq_low_signal_checkpoint_eligible(**values)
    for key, value in (("available", 0.), ("available", float("nan")), ("retained", float("inf")),
                       ("cq", -.1), ("keff_tp", float("nan"))):
        assert not cq_low_signal_checkpoint_eligible(**{**values, key: value})
    informative = {**values, "available": .06}
    assert cq_checkpoint_eligible(**informative)
    assert not cq_low_signal_checkpoint_eligible(**informative)


def test_cq_only_failure_retains_diagnostics_without_summary(tmp_path):
    prepared = _prepared("single_stage_cq_only")
    micro = np.full_like(prepared.micro_pij, 1 / prepared.micro_pij.shape[1])
    prepared = replace(prepared, micro_pij=micro, micro_ei=0.)
    with pytest.raises(RuntimeError, match="effective-signal"):
        train_deltaei(prepared, WYTCQOnlyConfig(2, tmp_path, hidden_dim=8, mid_dim=4, epochs=2, knn_k=2))
    assert (tmp_path / "metrics.csv").is_file()
    assert not (tmp_path / "summary.json").exists()


def test_cq_only_config_resolver_and_dispatch(tmp_path, monkeypatch):
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    import wyt_deltaei_coarse_grain.cq_only_trainer as trainer
    args = build_argparser().parse_args(_cli_args("single_stage_cq_only", tmp_path))
    cfg = build_training_config(args)
    assert isinstance(cfg, WYTCQOnlyConfig) and cfg.lambda_cq == 1 and cfg.lambda_dev == .05
    sentinel = object()
    monkeypatch.setattr(trainer, "train_deltaei_cq_only", lambda *_: sentinel)
    assert train_deltaei(_prepared("single_stage_cq_only"), cfg) is sentinel
    args.ei_retain_ratio = 1.
    with pytest.raises(ValueError, match="CQ-only does not support"):
        build_training_config(args)
