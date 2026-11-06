"""CQ-only scientific behavior and native artifact integration on small fixtures."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_wyt_two_stage import _prepared, _cli_args
from wyt_deltaei_coarse_grain import WYTCQOnlyConfig, train_deltaei
from wyt_deltaei_coarse_grain.cq_only_objective import cq_only_loss


def test_cq_loss_has_no_delta_or_keff_gradient(tmp_path):
    values = {key: torch.tensor(.1, requires_grad=True)
              for key in ("align", "var", "local", "sharp", "dev", "proto", "within", "inter", "closure",
                          "I_available", "I_retained", "delta", "keff")}
    loss, _ = cq_only_loss(values, WYTCQOnlyConfig(2, tmp_path), .01)
    loss.backward()
    assert values["delta"].grad is None and values["keff"].grad is None
    assert values["I_retained"].grad < 0


def test_cq_only_trains_one_stage_and_exports_negative_delta(tmp_path):
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
    assert (result.out_dir / "best_cq.pt").is_file() and (result.out_dir / "best_model.pt").is_file()
    assert not (result.out_dir / "best_joint.pt").exists()
    eligible = [row for row in result.metrics if row["checkpoint_eligible"]]
    assert result.best_epoch == max(eligible, key=lambda row: row["CQ"])["epoch"]
    for name in ("S_t", "S_tp", "PIJ_micro_train", "PIJ_macro_train"):
        matrix = np.load(result.out_dir / (name + ".npy"))
        assert np.isfinite(matrix).all() and np.allclose(matrix.sum(1), 1, atol=1e-5)
    with pytest.raises(FileExistsError):
        train_deltaei(prepared, cfg)


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
