from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.pij.wyt_single_kl import single_kl_pij_torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, WYTTwoStageDeltaEIConfig
from wyt_deltaei_coarse_grain.two_stage_objective import (
    dynamical_closure_loss,
    information_closure_metrics,
)
from wyt_deltaei_coarse_grain.two_stage_trainer import joint_checkpoint_eligible


def _macro_pij(inputs: MacroPijInputs):
    return single_kl_pij_torch(inputs.z_macro_t, inputs.z_macro_tp)


def _prepared(method: str = "maturity_cci_grn_two_stage") -> PreparedCoarseInput:
    rng = np.random.default_rng(31)
    count_t, count_tp = 6, 7
    micro_t = rng.normal(size=(count_t, 4)).astype(np.float32)
    micro_tp = rng.normal(size=(count_tp, 4)).astype(np.float32)
    logits = micro_t @ micro_tp.T
    micro_pij = np.exp(logits - logits.max(axis=1, keepdims=True))
    micro_pij /= micro_pij.sum(axis=1, keepdims=True)
    return PreparedCoarseInput(
        method=method,
        unit_ids_t=[f"t{index}" for index in range(count_t)],
        unit_ids_tp=[f"q{index}" for index in range(count_tp)],
        network_t=sp.csr_matrix(rng.random((count_t, count_t))),
        network_tp=sp.csr_matrix(rng.random((count_tp, count_tp))),
        encoder_features_t=rng.normal(size=(count_t, 5)).astype(np.float32),
        encoder_features_tp=rng.normal(size=(count_tp, 5)).astype(np.float32),
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij,
        micro_ei=0.1,
        macro_pij_builder=_macro_pij,
        maturity_t=np.linspace(0.0, 1.0, count_t, dtype=np.float32),
        maturity_tp=np.linspace(0.0, 1.0, count_tp, dtype=np.float32),
    )


def test_training_dispatch_isolates_legacy_and_two_stage(monkeypatch, tmp_path) -> None:
    import wyt_deltaei_coarse_grain.trainer as dispatcher
    import wyt_deltaei_coarse_grain.two_stage_trainer as two_stage

    calls: list[str] = []
    monkeypatch.setattr(dispatcher, "_train_single_stage", lambda _p, _c: calls.append("legacy"))
    monkeypatch.setattr(two_stage, "train_deltaei_two_stage", lambda _p, _c: calls.append("two_stage"))
    for method in (
        "complete_combined_coarse",
        "complete_combined_coarse_maturity_cci_grn",
    ):
        dispatcher.train_deltaei(
            SimpleNamespace(method=method),
            WYTDeltaEIConfig(k=2, out_dir=tmp_path / method),
        )
    dispatcher.train_deltaei(
        SimpleNamespace(method="maturity_cci_grn_two_stage"),
        WYTTwoStageDeltaEIConfig(k=2, out_dir=tmp_path / "two_stage"),
    )
    assert calls == ["legacy", "legacy", "two_stage"]


def test_information_objectives_are_finite_and_identity_holds() -> None:
    source = torch.tensor([[0.45, 0.40, 0.15], [0.70, 0.20, 0.10], [0.10, 0.25, 0.65]])
    target = source.clone()
    micro = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    macro = torch.eye(3)
    closure = dynamical_closure_loss(micro, source, target, macro)
    information = information_closure_metrics(micro, source, target)
    assert torch.isfinite(closure)
    assert torch.isfinite(information["I_available"])
    assert torch.isfinite(information["I_retained"])
    assert torch.allclose(
        information["I_available"],
        information["I_retained"] + information["closure_leakage"],
        atol=1e-6,
    )


def test_joint_checkpoint_requires_ei_retention_and_both_keff_floors() -> None:
    common = dict(
        delta_ei=1.0,
        retained_information=1.0,
        ei_reference=1.0,
        retained_reference=1.0,
        ei_retain_ratio=0.9,
        retained_ratio=0.5,
        checkpoint_keff_min=2.0,
    )
    assert not joint_checkpoint_eligible(keff_t=1.9, keff_tp=2.1, **common)
    assert not joint_checkpoint_eligible(keff_t=2.1, keff_tp=1.9, **common)
    assert joint_checkpoint_eligible(keff_t=2.1, keff_tp=2.1, **common)


def test_two_stage_smoke_writes_independent_checkpoints_and_stochastic_outputs(tmp_path) -> None:
    from wyt_deltaei_coarse_grain.trainer import train_deltaei

    result = train_deltaei(
        _prepared(),
        WYTTwoStageDeltaEIConfig(
            k=2,
            out_dir=tmp_path,
            hidden_dim=8,
            mid_dim=4,
            epochs=4,
            stage1_ratio=0.5,
            knn_k=2,
            log_every=1,
            lambda_dev=0.05,
            ei_retain_ratio=0.01,
            retain_floor_ratio=0.01,
            checkpoint_retain_ratio=0.01,
            keff_min=1.0,
            checkpoint_keff_min=1.0,
        ),
    )
    assert (tmp_path / "best_ei.pt").exists()
    assert (tmp_path / "best_stage2_fallback.pt").exists()
    assert (tmp_path / "best_joint.pt").exists()
    assert not (tmp_path / "best_model.pt").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_joint_fallback_used"] is False
    assert summary["best_joint_selection"] == "eligible_closure_retain_keff"
    for name in ("S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy"):
        values = np.load(tmp_path / name)
        assert np.isfinite(values).all()
        assert np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    assert any(row["stage"] == "stage1" for row in result.metrics)
    assert any(row["stage"] == "stage2" for row in result.metrics)
    for row in result.metrics:
        for key in ("L_closure", "I_available", "I_retained", "Keff_t", "Keff_tp"):
            assert np.isfinite(row[key])
