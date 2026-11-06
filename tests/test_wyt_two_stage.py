from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest
import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.coarse_frontends import prepare_coarse_input
from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import prepare as prepare_original
from wyt_deltaei_coarse_grain.macro_builder import macro_matrix, pool_feature_blocks, pool_to_macro
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
        available_information=1.0,
        retained_information=1.0,
        ei_reference=1.0,
        retained_reference=1.0,
        ei_retain_ratio=0.9,
        retained_ratio=0.5,
        signal_threshold_bits=0.01,
        checkpoint_keff_min=2.0,
    )
    assert not joint_checkpoint_eligible(keff_t=1.9, keff_tp=2.1, **common)
    assert not joint_checkpoint_eligible(keff_t=2.1, keff_tp=1.9, **common)
    assert joint_checkpoint_eligible(keff_t=2.1, keff_tp=2.1, **common)


@pytest.mark.parametrize("method,policy", [("maturity_cci_grn_two_stage", "legacy_floor"),
    ("two_stage_noKeff", "iterative_no_floor"), ("two_stage_noKeff_HardEI", "iterative_no_floor")])
def test_two_stage_smoke_writes_independent_checkpoints_and_stochastic_outputs(tmp_path, method, policy) -> None:
    from wyt_deltaei_coarse_grain.trainer import train_deltaei
    from mignet_ce.metrics import effective_information

    # A synthetic differentiable macro builder guarantees positive DeltaEI so this
    # four-epoch test checks the export path without relying on scientific convergence.
    def synthetic_macro(inputs):
        logits = inputs.z_macro_t @ inputs.z_macro_tp.T
        return torch.softmax(logits + 4.0 * torch.eye(logits.shape[0], device=logits.device), dim=1)

    prepared = _prepared(method)
    micro = .05 * prepared.micro_pij + .95 / prepared.micro_pij.shape[1]
    prepared = replace(prepared, micro_pij=micro, micro_ei=effective_information(micro.copy()),
                       macro_pij_builder=synthetic_macro)

    result = train_deltaei(
        prepared,
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
            ei_retain_ratio=1.0 if method == "two_stage_noKeff_HardEI" else .85,
            retain_floor_ratio=0.01,
            checkpoint_retain_ratio=0.01,
            keff_policy=policy,
            lambda_keff=1.0 if policy == "legacy_floor" else 0.0,
            lambda_dead_usage=.02 if policy == "legacy_floor" else 0.0,
            keff_min=1.0 if policy == "legacy_floor" else None,
            checkpoint_keff_min=1.0 if policy == "legacy_floor" else None,
        ),
    )
    assert (tmp_path / "best_ei.pt").exists()
    assert (tmp_path / "best_stage2_deltaei_diagnostic.pt").exists()
    assert (tmp_path / "best_joint.pt").exists()
    assert not (tmp_path / "best_model.pt").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == method
    assert summary["keff_policy"] == policy
    if policy == "iterative_no_floor":
        assert summary["resolved_keff_min"] is None and summary["resolved_checkpoint_keff_min"] is None
        assert all(row["weighted_keff"] == row["weighted_dead_usage"] == 0 for row in result.metrics)
    assert summary["best_joint_fallback_used"] is False
    assert summary["best_joint_selection"] == "strict_retained_first"
    assert summary["delta_EI_best_checkpoint"] > 0
    for name in ("S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy"):
        values = np.load(tmp_path / name)
        assert np.isfinite(values).all()
        assert np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    assert any(row["stage"] == "stage1" for row in result.metrics)
    assert any(row["stage"] == "stage2" for row in result.metrics)
    for row in result.metrics:
        for key in ("L_closure", "I_available", "I_retained", "Keff_t", "Keff_tp"):
            assert np.isfinite(row[key])


def test_two_stage_preserves_diagnostic_without_promoting_negative_deltaei(tmp_path):
    from wyt_deltaei_coarse_grain import train_deltaei

    config = WYTTwoStageDeltaEIConfig(
        k=2, out_dir=tmp_path, hidden_dim=8, mid_dim=4, epochs=4,
        stage1_ratio=.5, knn_k=2, log_every=1, lambda_dev=.05,
        ei_retain_ratio=.01, retain_floor_ratio=.01, checkpoint_retain_ratio=.01,
        keff_min=1., checkpoint_keff_min=1.,
    )
    with pytest.raises(RuntimeError, match="no valid non-collapsed positive-DeltaEI checkpoint"):
        train_deltaei(_prepared(), config)
    assert (tmp_path / "best_ei.pt").is_file()
    assert (tmp_path / "best_stage2_deltaei_diagnostic.pt").is_file()
    assert not (tmp_path / "best_joint.pt").exists()
    assert not (tmp_path / "summary.json").exists()


def _write_stage(tmp_path, label, count, seed):
    rng = np.random.default_rng(seed)
    units = [f"{label}_{i}" for i in range(count)]
    genes = [f"g{i}" for i in range(8)]
    h5ad = tmp_path / f"{label}.h5ad"
    with h5py.File(h5ad, "w") as handle:
        handle.create_group("obs").create_dataset("_index", data=np.array(units, dtype="S"))
        handle.create_group("var").create_dataset("_index", data=np.array(genes, dtype="S"))
        handle.create_dataset("X", data=rng.poisson(4, (count, len(genes))).astype(np.float32))
        handle.create_group("obsm").create_dataset("spatial", data=rng.normal(size=(count, 2)))
    cci = tmp_path / f"{label}_CCI_total.npz"
    sp.save_npz(cci, sp.csr_matrix(rng.random((count, count)).astype(np.float32)))
    index = tmp_path / f"{label}_index.tsv"
    pd.DataFrame({"spot_id": units}).to_csv(index, sep="\t", index=False)
    grn = tmp_path / f"{label}_grn.csv"
    pd.DataFrame({"regulator": genes, "target": genes[1:] + genes[:1],
                  "weight": rng.random(len(genes)) + .1}).to_csv(grn, index=False)
    maturity = tmp_path / f"{label}_maturity.csv"
    pd.DataFrame({"spot_id": units, "maturity": np.linspace(0., 1., count)}).iloc[::-1].to_csv(maturity, index=False)
    return h5ad, cci, index, grn, maturity

def test_real_file_formats_share_input_protocol_and_differentiable_macro_builder(tmp_path):
    source = _write_stage(tmp_path, "source", 18, 5)
    target = _write_stage(tmp_path, "target", 21, 6)
    request = CoarseFrontendRequest(
        h5ad_t=source[0], h5ad_tp=target[0], cci_t=source[1], cci_tp=target[1],
        cci_index_t=source[2], cci_index_tp=target[2], grn_t=source[3], grn_tp=target[3],
        maturity_t=source[4], maturity_tp=target[4],
        grn_feature_method="legacy_nmf", cci_feature_method="laplacian_hks", alpha_cci=.20,
        mid_dim=8, nmf_max_iter=3,
    )
    original = prepare_original(request)
    for method in ("maturity_cci_grn_two_stage", "two_stage_noKeff", "two_stage_noKeff_HardEI", "single_stage_cq_only"):
        prepared = prepare_coarse_input(method, request)
        assert prepared.method == method
        for key in ("encoder_features_t", "encoder_features_tp", "micro_features_t", "micro_pij", "maturity_t"):
            np.testing.assert_array_equal(getattr(prepared, key), getattr(original, key))
        np.testing.assert_allclose(prepared.maturity_t, np.linspace(0., 1., 18), atol=1e-6)
        assert prepared.provenance["coarse_method"] == method
        left_logits, right_logits = torch.randn(18, 3, requires_grad=True), torch.randn(21, 3, requires_grad=True)
        left, right = left_logits.softmax(1), right_logits.softmax(1)
        blocks_t = {key: torch.tensor(values) for key, values in prepared.feature_blocks_t.items()}
        blocks_tp = {key: torch.tensor(values) for key, values in prepared.feature_blocks_tp.items()}
        macro = prepared.macro_pij_builder(MacroPijInputs(
            pool_to_macro(torch.tensor(prepared.micro_features_t), left),
            pool_to_macro(torch.tensor(prepared.micro_features_tp), right),
            macro_matrix(torch.tensor(prepared.network_t.toarray()), left),
            macro_matrix(torch.tensor(prepared.network_tp.toarray()), right),
            pool_feature_blocks(blocks_t, left), pool_feature_blocks(blocks_tp, right),
        ))
        assert macro.shape == (3, 3) and torch.isfinite(macro).all()
        torch.testing.assert_close(macro.sum(1), torch.ones(3))
        macro[:, 0].sum().backward()
        assert torch.isfinite(left_logits.grad).all() and torch.isfinite(right_logits.grad).all()


def _cli_args(method, out_dir):
    return ["--method", method, "--h5ad-t", "t.h5ad", "--h5ad-tp", "tp.h5ad",
            "--cci-t", "t.npz", "--cci-tp", "tp.npz", "--grn-t", "t.csv", "--grn-tp", "tp.csv",
            "--maturity-t", "mt.csv", "--maturity-tp", "mp.csv", "--k", "2", "--out-dir", str(out_dir)]


def test_cli_selects_two_stage_config_and_preserves_its_options(tmp_path, monkeypatch):
    import mignet_ce.coarse_frontends as frontends
    import wyt_deltaei_coarse_grain as core
    from scripts import run_wyt_deltaei_coarse_grain as cli

    prepared = _prepared()
    monkeypatch.setattr(frontends, "prepare_coarse_input", lambda method, request: prepared)
    captured = []

    def train(actual, config):
        assert actual is prepared
        captured.append(config)
        return SimpleNamespace(best_epoch=1, final_delta_ei=.2, out_dir=config.out_dir)

    monkeypatch.setattr(core, "train_deltaei", train)
    cli.main(_cli_args("maturity_cci_grn_two_stage", tmp_path / "run") + [
        "--lambda-retain-norm", ".3", "--lambda-retain-floor", "1.5", "--stage1-ratio", ".4",
    ])
    config, = captured
    assert isinstance(config, WYTTwoStageDeltaEIConfig)
    assert config.lambda_retain_norm == .3 and config.lambda_retain_floor == 1.5
    assert config.stage1_ratio == .4 and config.lambda_dev == .05


def test_no_keff_variants_share_configuration_except_ei_target(tmp_path):
    from dataclasses import asdict
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    configs = [build_training_config(build_argparser().parse_args(_cli_args(method, tmp_path)))
               for method in ("two_stage_noKeff", "two_stage_noKeff_HardEI")]
    assert [cfg.ei_retain_ratio for cfg in configs] == [.85, 1.0]
    values = [asdict(cfg) for cfg in configs]
    for value in values:
        value.pop("ei_retain_ratio")
        assert value["lambda_keff"] == value["lambda_dead_usage"] == 0
        assert value["keff_min"] is value["checkpoint_keff_min"] is None
    assert values[0] == values[1]


def test_no_keff_selection_keeps_ei_target_and_native_relaxed_fallback():
    from wyt_deltaei_coarse_grain.two_stage_trainer import relaxed_stage2_checkpoint_eligible
    common = dict(delta_ei=.9, available_information=.1, retained_information=.08,
                  keff_t=1., keff_tp=1., ei_reference=1., retained_reference=.1,
                  retained_ratio=.5, signal_threshold_bits=.01, checkpoint_keff_min=None)
    assert joint_checkpoint_eligible(**common, ei_retain_ratio=.85)
    assert not joint_checkpoint_eligible(**common, ei_retain_ratio=1.)
    assert relaxed_stage2_checkpoint_eligible(delta_ei=.9, available_information=.1,
        keff_t=1., keff_tp=1., signal_threshold_bits=.01, checkpoint_keff_min=None)
    assert not joint_checkpoint_eligible(**{**common, "keff_t": float("nan")}, ei_retain_ratio=.85)


@pytest.mark.parametrize("method", ["two_stage_noKeff", "two_stage_noKeff_HardEI"])
def test_no_keff_native_fallback_is_exported_and_labeled(tmp_path, monkeypatch, method):
    import wyt_deltaei_coarse_grain.two_stage_trainer as trainer
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    from mignet_ce.metrics import effective_information
    from wyt_deltaei_coarse_grain import train_deltaei
    def macro(inputs):
        logits = inputs.z_macro_t @ inputs.z_macro_tp.T
        return torch.softmax(logits + 4 * torch.eye(2), dim=1)
    prepared = _prepared(method)
    micro = .05 * prepared.micro_pij + .95 / prepared.micro_pij.shape[1]
    prepared = replace(prepared, micro_pij=micro, micro_ei=effective_information(micro.copy()), macro_pij_builder=macro)
    args = build_argparser().parse_args(_cli_args(method, tmp_path / "run") + [
        "--epochs", "4", "--stage1-ratio", ".5", "--hidden-dim", "8", "--mid-dim", "4", "--knn-k", "2"])
    cfg = build_training_config(args)
    monkeypatch.setattr(trainer, "joint_checkpoint_eligible", lambda **_: False)
    result = train_deltaei(prepared, cfg)
    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary["best_joint_fallback_used"] is True and summary["strict_checkpoint_found"] is False
    assert summary["ei_retain_ratio"] == (.85 if method == "two_stage_noKeff" else 1.)
    assert summary["selected_checkpoint"] == "best_joint.pt"
    assert summary["trained_delta_EI"] > 0 and summary["checkpoint_metric_state_consistent"] is True


@pytest.mark.parametrize("method", ["shr_method_8", "shr_method_10"])
def test_cli_rejects_retired_methods_before_reading_inputs(tmp_path, method):
    from scripts import run_wyt_deltaei_coarse_grain as cli
    with pytest.raises(SystemExit):
        cli.main(_cli_args(method, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("option", [
    "--lambda-retain", "--graph-projection-dim", "--attention-heads", "--attention-ffn-dim",
    "--attention-dropout", "--attention-beta-init", "--cross-time-beta-init",
    "--cross-time-gate-bias-init", "--cosine-eta-min",
])
def test_cli_rejects_removed_trainer_options(tmp_path, option):
    from scripts import run_wyt_deltaei_coarse_grain as cli
    with pytest.raises(SystemExit):
        cli.build_argparser().parse_args(_cli_args("maturity_cci_grn_two_stage", tmp_path) + [option, "1"])
