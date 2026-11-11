"""Regression coverage for common Stage 1 state and independent formal branches."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mignet_ce.coarse_frontends import get_coarse_method_spec
from mignet_ce.coarse_frontends.method_specs import (
    FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL,
    FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from mignet_ce.downstream.loss_ablation.formal.variants import (
    VARIANTS, STAGE1_VARIANTS, comparison_predecessors, report_name, required_stage1_variants,
    select_variants, shared_stage1_preview, variants_for_protocol,
)
from scripts.run_wyt_deltaei_coarse_grain import build_argparser
from wyt_deltaei_coarse_grain.training_config import build_training_config
from wyt_deltaei_coarse_grain.two_stage_objective import occupancy_gap_loss
from wyt_deltaei_coarse_grain.two_stage_trainer import (
    WYTFormalTwoStageConfig, checkpoint_consistency, finalize_two_stage_run,
    formal_stage1_signature, train_deltaei_two_stage,
)


def _macro(inputs):
    logits = inputs.z_macro_t @ inputs.z_macro_tp.T
    return torch.softmax(logits + 4 * torch.eye(logits.shape[0], device=logits.device), dim=1)


def _prepared():
    rng = np.random.default_rng(31)
    micro = rng.random((6, 7))
    micro = .05 * micro / micro.sum(axis=1, keepdims=True) + .95 / 7
    return PreparedCoarseInput(
        method="formal_two_stage", unit_ids_t=list("abcdef"), unit_ids_tp=list("ghijklm"),
        network_t=sp.csr_matrix(rng.random((6, 6))), network_tp=sp.csr_matrix(rng.random((7, 7))),
        encoder_features_t=rng.normal(size=(6, 5)).astype("float32"),
        encoder_features_tp=rng.normal(size=(7, 5)).astype("float32"),
        micro_features_t=rng.normal(size=(6, 4)).astype("float32"),
        micro_features_tp=rng.normal(size=(7, 4)).astype("float32"),
        micro_pij=micro.astype("float32"), micro_ei=0.0, macro_pij_builder=_macro,
        maturity_t=np.linspace(0, 1, 6, dtype="float32"), maturity_tp=np.linspace(0, 1, 7, dtype="float32"),
    )


def _prepared_spot40():
    from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL

    rng = np.random.default_rng(31)
    source_count, target_count = 48, 50
    micro = rng.random((source_count, target_count))
    micro = .05 * micro / micro.sum(1, keepdims=True) + .95 / target_count
    return replace(
        _prepared(), unit_ids_t=[str(i) for i in range(source_count)],
        unit_ids_tp=[str(i) for i in range(target_count)],
        network_t=sp.eye(source_count, format="csr"), network_tp=sp.eye(target_count, format="csr"),
        encoder_features_t=rng.normal(size=(source_count, 5)).astype("float32"),
        encoder_features_tp=rng.normal(size=(target_count, 5)).astype("float32"),
        micro_features_t=rng.normal(size=(source_count, 32)).astype("float32"),
        micro_features_tp=rng.normal(size=(target_count, 32)).astype("float32"),
        micro_pij=micro.astype("float32"), maturity_t=np.linspace(0, 1, source_count, dtype="float32"),
        maturity_tp=np.linspace(0, 1, target_count, dtype="float32"),
        provenance={"loss_protocol": FORMAL_TWO_STAGE_PROTOCOL},
    )


def test_consistency_accepts_reported_roundoff_but_rejects_wrong_or_nonfinite_metrics():
    recorded = dict(delta_EI=1.9, I_available=.02, I_retained=.004,
                    closure_quality=.2, Keff_t=14., Keff_tp=22.)
    recomputed = dict(recorded, delta_EI=1.9 + 1.0013580322265625e-5,
                      I_available=.02 + 2.7939677238464355e-8,
                      I_retained=.004 + 1.5366822481155396e-8)
    checks = checkpoint_consistency(recorded, recomputed)
    assert all(item["passed"] for item in checks.values())
    assert checks["delta_EI"]["absolute_error"] > 1e-5
    for value in (1.91, float("nan"), float("inf")):
        assert not checkpoint_consistency(recorded, dict(recomputed, delta_EI=value))["delta_EI"]["passed"]
    assert not checkpoint_consistency(recorded, dict(recomputed, I_retained=.004002))["I_retained"]["passed"]


def test_finalize_restores_failed_export_without_training_and_rejects_wrong_weights(tmp_path, monkeypatch):
    import shutil
    import wyt_deltaei_coarse_grain.two_stage_trainer as trainer

    prepared = _prepared()
    config = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / "failed", hidden_dim=8, mid_dim=4,
                                    epochs=4, stage1_ratio=.5, knn_k=2, log_every=1)
    original_check = trainer.checkpoint_consistency

    def simulate_failed_export(recorded, recomputed):
        result = original_check(recorded, recomputed)
        result["delta_EI"]["passed"] = False
        return result

    monkeypatch.setattr(trainer, "checkpoint_consistency", simulate_failed_export)
    with pytest.raises(AssertionError, match="Reloaded checkpoint metrics"):
        train_deltaei_two_stage(prepared, config)
    assert not (config.out_dir / "summary.json").exists()
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in config.out_dir.iterdir()}
    monkeypatch.setattr(trainer, "checkpoint_consistency", original_check)

    def forbidden_training(*_args, **_kwargs):
        raise AssertionError("Finalization must never optimize or backpropagate")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_training)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden_training)
    output_config = replace(config, out_dir=tmp_path / "finalized")
    result = finalize_two_stage_run(prepared, output_config, config.out_dir)
    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary["finalization_only"] and summary["optimizer_steps_this_run"] == 0
    assert summary["checkpoint_metric_state_consistent"]
    assert summary["checkpoint_consistency_policy"] == "per_metric_atol_plus_rtol_v1"
    assert len(result.metrics) == 4
    for name, value in source_hashes.items():
        assert hashlib.sha256((config.out_dir / name).read_bytes()).hexdigest() == value
    for name in ("best_ei.pt", "best_joint.pt"):
        assert (result.out_dir / name).read_bytes() == (config.out_dir / name).read_bytes()
    log = (config.out_dir / "train.log").read_text()
    assert "Effective loss weights: align=0; stage1_closure=0.5" in log
    assert "weighted_align=0.000000" in log and "FeatureAlign-DeltaEI" not in log
    assert "optimizer steps=0" in (result.out_dir / "train.log").read_text()
    with pytest.raises(ValueError, match="loss/model configuration"):
        finalize_two_stage_run(prepared, replace(output_config, out_dir=tmp_path / "wrong_config", lambda_align=1), config.out_dir)
    incomplete = tmp_path / "incomplete"
    shutil.copytree(config.out_dir, incomplete)
    history_lines = (incomplete / "metrics.csv").read_text().splitlines()
    (incomplete / "metrics.csv").write_text("\n".join(history_lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="complete Stage 1 and Stage 2 history"):
        finalize_two_stage_run(prepared, replace(config, out_dir=tmp_path / "incomplete_output"), incomplete)
    bad_source = tmp_path / "changed_weights"
    shutil.copytree(config.out_dir, bad_source)
    joint = torch.load(bad_source / "best_joint.pt", map_location="cpu", weights_only=False)
    joint["macro_net"]["out.weight"] *= 1000
    torch.save(joint, bad_source / "best_joint.pt")
    bad_output = tmp_path / "wrong_state"
    with pytest.raises(AssertionError, match="Reloaded checkpoint metrics"):
        finalize_two_stage_run(prepared, replace(config, out_dir=bad_output), bad_source)
    assert not json.loads((bad_output / "checkpoint_consistency.json").read_text())["passed"]
    assert not (bad_output / "summary.json").exists()


@pytest.mark.parametrize("protocol,step,manifest_version", [
    (FORMAL_TWO_STAGE_1B_PROTOCOL, "branch3", 3),
    (FORMAL_TWO_STAGE_GAP_PROTOCOL, "branch4", 4),
])
def test_finalize_workflow_reuses_five_runs_and_exports_six_pair_report(
    tmp_path, monkeypatch, protocol, step, manifest_version,
):
    import csv
    from pathlib import Path
    from types import SimpleNamespace
    import shutil
    from joblib.externals import cloudpickle
    from mignet_ce.downstream.loss_ablation.formal import workflow
    from mignet_ce.downstream.loss_ablation.formal.variants import TIME_PAIRS
    import wyt_deltaei_coarse_grain.two_stage_trainer as trainer

    prepared = replace(_prepared_spot40(), provenance={"loss_protocol": protocol})
    legacy_variants = variants_for_protocol(protocol)
    root = tmp_path / "server_round"
    args = SimpleNamespace(data_root=tmp_path / "data", cci_root=tmp_path / "cci", grn_root=tmp_path / "grn",
                           developmental_root=tmp_path / "maturity", seed=42, epochs=4, device="cpu",
                           output_root=root, branches=[step])
    parser = build_argparser()
    inputs = {}
    for pair in TIME_PAIRS:
        parsed = parser.parse_args(workflow.command(args, pair, legacy_variants[0], tmp_path / "unused")[2:])
        for key in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "cci_index_t", "cci_index_tp",
                    "grn_t", "grn_tp", "maturity_t", "maturity_tp"):
            path = getattr(parsed, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("training input")
            inputs[str(path)] = workflow.file_digest(path)
    base = build_training_config(parser.parse_args(workflow.command(
        args, TIME_PAIRS[0], legacy_variants[0], tmp_path / "failed_base_template")[2:]))
    original_check = trainer.checkpoint_consistency

    def failed_export(recorded, recomputed):
        result = original_check(recorded, recomputed)
        result["delta_EI"]["passed"] = False
        return result

    monkeypatch.setattr(trainer, "checkpoint_consistency", failed_export)
    with pytest.raises(AssertionError, match="Reloaded checkpoint metrics"):
        train_deltaei_two_stage(prepared, base)
    monkeypatch.setattr(trainer, "checkpoint_consistency", original_check)
    completed_base = replace(base, out_dir=tmp_path / "completed_base_template")
    finalize_two_stage_run(prepared, completed_base, base.out_dir)
    gap = next(variant for variant in legacy_variants if variant.name == step)
    gap_config = build_training_config(parser.parse_args(workflow.command(
        args, TIME_PAIRS[0], gap, tmp_path / "gap_template")[2:]))
    train_deltaei_two_stage(prepared, gap_config)
    assert "Occupancy loss: Lgap" in (gap_config.out_dir / "train.log").read_text()

    root.mkdir()
    config_file = root / "run_config.json"
    config_file.write_text(json.dumps(dict(protocol=protocol, manifest_version=manifest_version,
        input_sha256=inputs, training_code_sha256="original_training_fingerprint", epochs=4, seed=42, device="cpu")))
    old_report = root / "reports" / step / "run_manifest.json"
    old_report.parent.mkdir(parents=True)
    old_report.write_text(json.dumps({"status": "failed", "error": "previous fixed-tolerance failure"}))
    cache_dir = root / "_prepared"
    cache_dir.mkdir()
    for pair in TIME_PAIRS:
        key = "_to_".join(pair)
        cache = cache_dir / f"{key}.pkl"
        with cache.open("wb") as stream:
            cloudpickle.dump(prepared, stream)
        cache.with_suffix(".sha256").write_text(workflow.file_digest(cache))
        for variant in (legacy_variants[0], gap):
            failed = variant.name == "baseline" and pair == TIME_PAIRS[2]
            template = base.out_dir if failed else (
                completed_base.out_dir if variant.name == "baseline" else gap_config.out_dir)
            run_dir = root / "stage2" / variant.name / key / "attempt_0001"
            shutil.copytree(template, run_dir)
            saved = json.loads((run_dir / "config.json").read_text())
            (run_dir / "config.json").write_text(json.dumps(dict(saved, out_dir=str(run_dir))))
            names = ("summary.json", "best_joint.pt", "best_ei.pt", "metrics.csv", "config.json")
            context = {"status": "failed" if failed else "complete", "identity": {
                "cache_sha256": workflow.file_digest(cache),
                "source_sha256": workflow.file_digest(run_dir / "best_ei.pt")}}
            if not failed:
                context["artifacts"] = {name: workflow.file_digest(run_dir / name) for name in names}
            run_dir.with_suffix(".json").write_text(json.dumps(context))
    before = {path: workflow.file_digest(path) for path in root.rglob("*") if path.is_file()}

    def forbidden_training(*_args, **_kwargs):
        raise AssertionError("Finalization must not launch or execute training")

    monkeypatch.setattr(workflow.subprocess, "run", forbidden_training)
    monkeypatch.setattr(workflow, "_parallel", forbidden_training)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_training)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden_training)
    result_dir = workflow.finalize(args)
    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["optimizer_steps"] == 0
    assert manifest["training_code_sha256"] == "original_training_fingerprint"
    assert manifest["finalization_code_sha256"] != manifest["training_code_sha256"]
    assert [item["action"] for item in manifest["results"]].count("reused_complete") == 5
    assert [item["action"] for item in manifest["results"]].count("recomputed_checkpoint") == 1
    assert all(workflow.file_digest(path) == checksum for path, checksum in before.items())
    for name in ("information", "cluster_occupancy"):
        assert (result_dir / f"{name}.png").is_file() and (result_dir / f"{name}.pdf").is_file()
        with (result_dir / f"{name}.csv").open(newline="") as stream:
            exported = list(csv.DictReader(stream))
            assert len(exported) == 12
            if protocol == FORMAL_TWO_STAGE_GAP_PROTOCOL:
                assert any(key.startswith("delta_vs_previous_") for key in exported[0])
    next(Path(name) for name in inputs).write_text("changed training input")
    with pytest.raises(ValueError, match="training input checksum mismatch"):
        workflow.finalize(args)


def test_formal_method_defaults_and_independent_branches(tmp_path):
    parser = build_argparser()
    configs = {}
    for variant in VARIANTS:
        args = parser.parse_args([
            "--method", "formal_two_stage", "--h5ad-t", "t.h5ad", "--h5ad-tp", "tp.h5ad",
            "--cci-t", "t.npz", "--cci-tp", "tp.npz", "--k", "40", "--out-dir", str(tmp_path),
            *variant.flags,
        ])
        configs[variant.name] = build_training_config(args)
    assert len(STAGE1_VARIANTS) == 1
    base = configs["baseline"]
    assert get_coarse_method_spec("formal_two_stage").requires_maturity
    assert base.lambda_align == 0 and base.lambda_stage1_closure == .5
    assert base.lambda_keff == 0 and base.lambda_gap == 1 and base.lambda_dev == 1
    assert base.ei_loss_retain_ratio == 1 and base.ei_retain_ratio == .85
    assert get_coarse_method_spec("formal_two_stage").default_lambda_dev == 1
    assert all(getattr(base, "lambda_" + key) == 0 for key in (
        "var", "local", "proto", "dead_usage", "sharp", "within", "inter",
    ))
    assert configs["branch2b"].lambda_retain_norm == .5
    assert configs["branch2b"].lambda_retain_floor == 0 and configs["branch2b"].lambda_stage2_cq == 1
    assert configs["branch2a"].lambda_retain_floor == 0 and configs["branch2a"].lambda_stage2_cq == 0
    assert configs["branch2a"].checkpoint_retain_ratio == configs["branch2b"].checkpoint_retain_ratio == .5
    prepared = _prepared()
    reordered = replace(prepared, unit_ids_t=list(reversed(prepared.unit_ids_t)))
    assert formal_stage1_signature(prepared, base) != formal_stage1_signature(reordered, base)
    for name in ("branch2a", "branch2b"):
        assert formal_stage1_signature(prepared, configs[name]) == formal_stage1_signature(prepared, base)
    assert configs["branch2a"].lambda_dev == configs["branch2b"].lambda_dev == 1
    native = parser.parse_args([
        "--method", "maturity_cci_grn_two_stage", "--h5ad-t", "t.h5ad", "--h5ad-tp", "tp.h5ad",
        "--cci-t", "t.npz", "--cci-tp", "tp.npz", "--k", "40", "--out-dir", str(tmp_path),
    ])
    assert build_training_config(native).lambda_align == 1 and build_training_config(native).lambda_keff == 1


def test_gap_zero_empty_states_and_finite_gradient():
    for counts in ([100], [25, 25, 25, 25], [70, 10, 10, 10]):
        labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        assignment = torch.nn.functional.one_hot(labels, 40).float().requires_grad_()
        loss = occupancy_gap_loss(assignment, assignment)
        assert torch.isfinite(loss)
        if len(set(counts)) == 1:
            assert loss.item() == pytest.approx(0, abs=1e-6)
        else:
            assert loss.item() > 0
        loss.backward()
        assert torch.isfinite(assignment.grad).all()
    torch.manual_seed(42)
    logits = torch.randn(100, 40, requires_grad=True)
    assignment = logits.softmax(1)
    occupancy_gap_loss(assignment, assignment).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0


def test_shared_stage1_restores_state_and_separates_ei_floors(tmp_path):
    prepared = _prepared()
    base = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / "stage1", hidden_dim=8, mid_dim=4,
                                  epochs=4, stage1_ratio=.5, knn_k=2, log_every=1, stage1_only=True,
                                  lambda_dev=.05, ei_loss_retain_ratio=.85)
    train_deltaei_two_stage(prepared, base)
    source = base.out_dir / "best_ei.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    reference = train_deltaei_two_stage(prepared, replace(base, out_dir=tmp_path / "full", stage1_only=False))
    for name, overrides in (
        ("baseline", {}),
        ("branch2b", {"lambda_retain_floor": 0., "lambda_stage2_cq": 1.}),
        ("ei_loss_only", {"ei_loss_retain_ratio": 1.}),
    ):
        config = replace(base, out_dir=tmp_path / name, stage1_only=False, stage1_source=source, **overrides)
        result = train_deltaei_two_stage(prepared, config)
        assert result.metrics[:2] == checkpoint["stage1_history"]
        if name == "baseline":
            assert result.metrics == reference.metrics
            for row in result.metrics[:2]:
                expected = -row["delta_EI"] + .5 * row["L_closure"] + .05 * row["L_dev"] + row["L_gap"]
                assert row["loss"] == pytest.approx(expected, abs=1e-6)
        summary = json.loads((config.out_dir / "summary.json").read_text())
        assert summary["stage1_checkpoint_sha256"] == digest
        assert summary["stage1_best_epoch"] == checkpoint["epoch"]
        assert summary["ei_checkpoint_retain_ratio"] == .85
        for row in result.metrics[2:]:
            assert row["EI_checkpoint_floor"] == pytest.approx(.85 * checkpoint["best_delta_ei"])
            assert row["EI_loss_floor"] == pytest.approx(config.ei_loss_retain_ratio * checkpoint["best_delta_ei"])
            expected = .05 * row["L_dev"] + row["L_gap"] + 5 * row["L_ei_constraint"]
            expected += row["weighted_closure"] + row["weighted_retain_norm"]
            expected += row["weighted_retain_floor"] + row["weighted_stage2_cq"]
            assert row["loss"] == pytest.approx(expected, abs=1e-6)
        selected = torch.load(config.out_dir / "best_joint.pt", map_location="cpu", weights_only=False)
        assert len(selected["joint_score"]) == 3
    with pytest.raises(ValueError, match="fingerprint"):
        train_deltaei_two_stage(prepared, replace(base, out_dir=tmp_path / "mismatch", stage1_only=False,
                                                stage1_source=source, lambda_dev=1.0))


def test_staircase_losses_share_branch4_state_and_only_raise_stage2_training_ei_floor(tmp_path):
    prepared = _prepared()
    base = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / "baseline", hidden_dim=8, mid_dim=4,
                                  epochs=4, stage1_ratio=.5, knn_k=2, log_every=1,
                                  lambda_dev=.05, ei_loss_retain_ratio=.85)
    baseline = train_deltaei_two_stage(prepared, base)
    fourth = replace(base, lambda_dev=1., stage1_only=True, out_dir=tmp_path / "fourth_stage1")
    train_deltaei_two_stage(prepared, fourth)
    source = fourth.out_dir / "best_ei.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    stage1_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    results = []
    for name, ratio in (("branch4", .85), ("branch5", 1.)):
        config = replace(fourth, out_dir=tmp_path / name, stage1_only=False,
                         stage1_source=source, ei_loss_retain_ratio=ratio)
        result = train_deltaei_two_stage(prepared, config)
        results.append(result)
        assert result.metrics[:2] == checkpoint["stage1_history"]
        assert hashlib.sha256((result.out_dir / "best_ei.pt").read_bytes()).hexdigest() == stage1_hash
        for row in result.metrics:
            expected = row["L_dev"] + row["L_gap"]
            if row["stage"] == "stage1":
                expected += -row["delta_EI"] + .5 * row["L_closure"]
            else:
                expected += row["weighted_closure"] + row["weighted_retain_norm"] + row["weighted_retain_floor"]
                floor_loss = max(0, ratio * checkpoint["best_delta_ei"] - row["delta_EI"]) ** 2
                expected += 5 * floor_loss
                assert row["EI_loss_floor"] == pytest.approx(ratio * checkpoint["best_delta_ei"])
                assert row["EI_checkpoint_floor"] == pytest.approx(.85 * checkpoint["best_delta_ei"])
            assert row["weighted_keff"] == 0
            assert row["loss"] == pytest.approx(expected, abs=1e-6)
    assert results[0].metrics[2]["delta_EI"] == results[1].metrics[2]["delta_EI"]
    assert formal_stage1_signature(prepared, base) != formal_stage1_signature(prepared, fourth)
    for row in baseline.metrics[:2]:
        assert row["loss"] == pytest.approx(-row["delta_EI"] + .5 * row["L_closure"]
                                          + .05 * row["L_dev"] + row["L_gap"], abs=1e-6)


@pytest.mark.parametrize("threshold,expected_pair_weight", [(0., 1.), (1000., .1)])
def test_retain_cq_staircase_uses_same_legacy_stage1_and_exact_confirmed_losses(
    tmp_path, threshold, expected_pair_weight,
):
    prepared = replace(_prepared(), provenance={"loss_protocol": FORMAL_TWO_STAGE_PROTOCOL})
    base = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / "source", hidden_dim=8, mid_dim=4,
                                  epochs=4, stage1_ratio=.5, knn_k=2, log_every=1,
                                  low_signal_threshold_bits=threshold)
    # The promoted baseline reuses the exact Stage 1 state of historical 4+5.
    historical = replace(prepared, provenance={"loss_protocol": FORMAL_TWO_STAGE_GAP_PROTOCOL})
    train_deltaei_two_stage(historical, replace(base, stage1_only=True, ei_loss_retain_ratio=.85))
    source = base.out_dir / "best_ei.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    results = []
    for name, floor_weight, cq_weight in (("baseline", 2., 0.), ("branch2a", 0., 0.), ("branch2b", 0., 1.)):
        config = replace(base, out_dir=tmp_path / name, stage1_source=source,
                         lambda_retain_floor=floor_weight, lambda_stage2_cq=cq_weight)
        result = train_deltaei_two_stage(prepared, config)
        results.append(result)
        assert result.metrics[:2] == checkpoint["stage1_history"]
        assert hashlib.sha256((result.out_dir / "best_ei.pt").read_bytes()).hexdigest() == digest
        for row in result.metrics:
            expected = row["L_dev"] + row["L_gap"]
            if row["stage"] == "stage1":
                expected += -row["delta_EI"] + .5 * row["L_closure"]
            else:
                retained = row["I_retained"]
                retain_norm = -retained / max(checkpoint["I_available"], 1e-8)
                floor = max(0., .5 - retained / max(checkpoint["I_retained"], 1e-8)) ** 2
                cq = retained / max(row["I_available"], 1e-8)
                ei_loss = max(0., checkpoint["best_delta_ei"] - row["delta_EI"]) ** 2
                expected += expected_pair_weight * (.1 * row["L_closure"] + .5 * retain_norm
                                                   + floor_weight * floor + cq_weight * (1 - cq))
                expected += 5 * ei_loss
                assert row["pair_weight"] == expected_pair_weight
                assert row["weighted_stage2_cq"] == pytest.approx(expected_pair_weight * cq_weight * (1 - cq))
                assert row["weighted_retain_floor"] == pytest.approx(expected_pair_weight * floor_weight * floor)
                assert row["EI_loss_floor"] == pytest.approx(checkpoint["best_delta_ei"])
                assert row["EI_checkpoint_floor"] == pytest.approx(.85 * checkpoint["best_delta_ei"])
            assert row["loss"] == pytest.approx(expected, abs=2e-6)
            assert row["weighted_keff"] == row["weighted_align"] == 0
        assert config.checkpoint_retain_ratio == .5
    assert len({result.metrics[2]["delta_EI"] for result in results}) == 1
    assert all(result.metrics[:2] == results[0].metrics[:2] for result in results)


def test_retain_cq_preview_shares_all_stage1_rows_and_compares_2b_to_2a(tmp_path):
    import csv
    from mignet_ce.downstream.loss_ablation.plots import preview_rows
    from mignet_ce.downstream.loss_ablation.tables import write_tables

    selected = select_variants()
    methods = tuple(variant.label for variant in selected)
    rows = preview_rows(("spot",), methods)
    shared_stage1_preview(rows, selected)
    write_tables(rows, tmp_path, baseline_method=methods[0], predecessors=comparison_predecessors(selected))
    for filename in ("information.csv", "cluster_occupancy.csv"):
        with (tmp_path / filename).open(newline="") as stream:
            exported = list(csv.DictReader(stream))
        stage1 = [row for row in exported if row["stage"] == "stage1" and row["method"] != methods[0]]
        assert all(float(value) == 0 for row in stage1 for key, value in row.items()
                   if key.startswith("delta_vs_baseline_") or key.startswith("delta_vs_previous_"))
        assert all(row["comparison_previous_method"] == "Branch 2A" for row in exported if row["method"] == "Branch 2B")


def test_removing_training_retain_floor_preserves_strict_checkpoint_retain_gate():
    from wyt_deltaei_coarse_grain.two_stage_trainer import joint_checkpoint_eligible

    parameters = dict(delta_ei=.9, available_information=1., retained_information=.51,
                      keff_t=40., keff_tp=40., ei_reference=1., retained_reference=1.,
                      ei_retain_ratio=.85, retained_ratio=.5, signal_threshold_bits=0., checkpoint_keff_min=10.)
    assert joint_checkpoint_eligible(**parameters)
    assert not joint_checkpoint_eligible(**dict(parameters, retained_information=.49))
    assert not joint_checkpoint_eligible(**dict(parameters, delta_ei=.84))


def test_staircase_preview_and_csv_compare_the_correct_predecessor(tmp_path):
    import csv
    from mignet_ce.downstream.loss_ablation.plots import preview_rows
    from mignet_ce.downstream.loss_ablation.tables import predecessor_differences, write_tables

    selected = select_variants(protocol=FORMAL_TWO_STAGE_GAP_PROTOCOL)
    methods = tuple(variant.label for variant in selected)
    rows = preview_rows(("spot",), methods)
    shared_stage1_preview(rows, selected)
    by_key = {(row["method"], row["stage"], row["source_time"], row["target_time"]): row for row in rows}
    for row in rows:
        if row["method"] == "Branch 4+5" and row["stage"] == "stage1":
            fourth = by_key["Branch 4", row["stage"], row["source_time"], row["target_time"]]
            assert {key: value for key, value in row.items() if key != "method"} == {
                key: value for key, value in fourth.items() if key != "method"}
    parents = comparison_predecessors(selected)
    comparisons = predecessor_differences(rows, parents)
    fifth_first = next(row for row in comparisons if row["method"] == "Branch 4+5" and row["stage"] == "stage1")
    assert fifth_first["delta_vs_previous_delta_ei_bits"] == 0
    write_tables(rows, tmp_path, baseline_method=methods[0], predecessors=parents)
    with (tmp_path / "information.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    fifth = next(row for row in exported if row["method"] == "Branch 4+5" and row["stage"] == "stage2")
    assert fifth["comparison_previous_method"] == "Branch 4"
    assert float(fifth["delta_vs_baseline_delta_ei_bits"]) == pytest.approx(.064)
    assert float(fifth["delta_vs_previous_delta_ei_bits"]) == pytest.approx(.032)
    with pytest.raises(ValueError, match="Missing staircase predecessor"):
        predecessor_differences([row for row in rows if row["method"] != "Branch 4"], parents)


def test_protocol_profiles_preserve_independent_historical_branch5(tmp_path):
    parser = build_argparser()
    configs = {}
    for protocol in (FORMAL_TWO_STAGE_1B_PROTOCOL, FORMAL_TWO_STAGE_GAP_PROTOCOL):
        variant = next(item for item in variants_for_protocol(protocol) if item.name == "branch5")
        args = parser.parse_args(["--method", "formal_two_stage", "--h5ad-t", "t", "--h5ad-tp", "tp",
                                  "--cci-t", "t", "--cci-tp", "tp", "--k", "40", "--out-dir", str(tmp_path),
                                  *variant.flags])
        configs[protocol] = (variant, build_training_config(args))
    old_variant, old = configs[FORMAL_TWO_STAGE_1B_PROTOCOL]
    new_variant, new = configs[FORMAL_TWO_STAGE_GAP_PROTOCOL]
    assert old_variant.stage1_group == "baseline" and old.lambda_dev == .05
    assert old.lambda_keff == 1 and old.lambda_gap == 0
    assert new_variant.stage1_group == "branch4" and new.lambda_dev == 1
    assert new.lambda_keff == 0 and new.lambda_gap == 1


def test_legacy_baseline_defaults_are_frozen_after_promoting_dev_and_ei(tmp_path):
    parser = build_argparser()
    assert get_coarse_method_spec("formal_two_stage").default_lambda_dev == 1
    for protocol in (FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL, FORMAL_TWO_STAGE_GAP_PROTOCOL):
        baseline = variants_for_protocol(protocol)[0]
        args = parser.parse_args(["--method", "formal_two_stage", "--h5ad-t", "t", "--h5ad-tp", "tp",
                                  "--cci-t", "t", "--cci-tp", "tp", "--k", "40", "--out-dir", str(tmp_path),
                                  *baseline.flags])
        config = build_training_config(args)
        assert config.lambda_dev == .05 and config.ei_loss_retain_ratio == .85
        assert config.ei_retain_ratio == .85 and config.lambda_stage2_cq == 0
        assert config.lambda_keff == (0 if protocol == FORMAL_TWO_STAGE_GAP_PROTOCOL else 1)
        assert config.lambda_gap == (1 if protocol == FORMAL_TWO_STAGE_GAP_PROTOCOL else 0)


def test_branch3_replaces_keff_in_both_stages_and_accepts_old_1b_state(tmp_path):
    from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL

    prepared = replace(_prepared(), provenance={"loss_protocol": FORMAL_TWO_STAGE_PROTOCOL})
    base = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / "baseline", hidden_dim=8, mid_dim=4,
                                  epochs=4, stage1_ratio=.5, knn_k=2, log_every=1,
                                  lambda_keff=1, lambda_gap=0, lambda_dev=.05, ei_loss_retain_ratio=.85)
    reference = train_deltaei_two_stage(prepared, base)
    legacy = replace(prepared, provenance={"loss_protocol": "formal_two_stage_v1"})
    legacy_config = replace(base, out_dir=tmp_path / "old_1b", stage1_only=True)
    train_deltaei_two_stage(legacy, legacy_config)
    resumed = train_deltaei_two_stage(prepared, replace(
        base, out_dir=tmp_path / "reused_1b", stage1_source=legacy_config.out_dir / "best_ei.pt",
    ))
    assert resumed.metrics == reference.metrics
    gap_config = replace(base, out_dir=tmp_path / "gap", lambda_keff=0, lambda_gap=1)
    gap = train_deltaei_two_stage(prepared, gap_config)
    for row in gap.metrics:
        expected = .05 * row["L_dev"] + row["L_gap"]
        if row["stage"] == "stage1":
            expected += -row["delta_EI"] + .5 * row["L_closure"]
        else:
            expected += row["weighted_closure"] + row["weighted_retain_norm"]
            expected += row["weighted_retain_floor"] + 5 * row["L_ei_constraint"]
        assert row["loss"] == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match="fingerprint"):
        train_deltaei_two_stage(prepared, replace(gap_config, out_dir=tmp_path / "wrong_shared_stage1",
                                                stage1_source=legacy_config.out_dir / "best_ei.pt"))


def test_six_pair_chains_advance_without_stage_barrier(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from threading import Barrier, Event, Lock
    from mignet_ce.downstream.loss_ablation.formal import workflow

    all_started = Barrier(6)
    release = Event()
    state_lock = Lock()
    active = peak = finished_other_stage1 = 0
    immediate_transition = []

    def execute(job):
        nonlocal active, peak, finished_other_stage1
        with state_lock:
            active += 1
            peak = max(peak, active)
        try:
            if job["stage1"]:
                all_started.wait(timeout=10)
                if job["number"] != 0:
                    assert release.wait(timeout=10)
                    with state_lock:
                        finished_other_stage1 += 1
                job["parent"].mkdir(parents=True)
                (job["parent"] / "best_ei.pt").write_bytes(b"state")
            elif job["number"] == 0:
                with state_lock:
                    immediate_transition.append(finished_other_stage1 == 0)
                release.set()
            return job["parent"]
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(workflow, "_execute", execute)
    jobs = []
    for number in range(6):
        jobs.append({"key": number, "stage1_lock": Lock(), "stage1_ready": Future(),
                     "stage1_job": {"stage1": True, "number": number, "parent": tmp_path / str(number)},
                     "stage2_job": {"stage1": False, "number": number, "command": [],
                                    "parent": tmp_path / f"stage2_{number}"}})
    assert len(workflow._parallel(jobs, workers=6)) == 6
    assert peak == 6 and immediate_transition == [True]


@pytest.mark.parametrize("source_protocol,target_protocol,source_variant", [
    (FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL, "branch1b"),
    (FORMAL_TWO_STAGE_1B_PROTOCOL, FORMAL_TWO_STAGE_GAP_PROTOCOL, "branch3"),
    (FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL, "branch5"),
])
def test_previous_baseline_import_checks_inputs_configuration_and_artifacts(
    tmp_path, source_protocol, target_protocol, source_variant,
):
    from types import SimpleNamespace
    import shutil
    from joblib.externals import cloudpickle
    from mignet_ce.downstream.loss_ablation.formal import workflow
    from mignet_ce.downstream.loss_ablation.formal.variants import TIME_PAIRS

    args = SimpleNamespace(data_root=tmp_path / "data", cci_root=tmp_path / "cci", grn_root=tmp_path / "grn",
                           developmental_root=tmp_path / "maturity", seed=42, epochs=4, device="cpu",
                           reuse_baseline_from=tmp_path / "old_round")
    prepared = replace(_prepared_spot40(), provenance={"loss_protocol": target_protocol})
    baseline = variants_for_protocol(target_protocol)[0]
    parsed = build_argparser().parse_args(workflow.command(args, TIME_PAIRS[0], baseline, tmp_path / "template")[2:])
    config = build_training_config(parsed)
    train_deltaei_two_stage(replace(prepared, provenance={"loss_protocol": source_protocol}), config)
    root = args.reuse_baseline_from
    root.mkdir()
    plan = dict(protocol=target_protocol, input_sha256={"fixture": "sha"}, epochs=4, seed=42, device="cpu")
    (root / "run_config.json").write_text(json.dumps(dict(plan, protocol=source_protocol)))
    cache = tmp_path / "new_prepared.pkl"
    with cache.open("wb") as stream:
        cloudpickle.dump(prepared, stream)
    names = ("summary.json", "best_ei.pt", "best_joint.pt", "metrics.csv", "config.json")
    for pair in TIME_PAIRS:
        run_dir = root / "stage2" / source_variant / "_to_".join(pair) / "attempt_0001"
        shutil.copytree(config.out_dir, run_dir)
        context = {"status": "complete", "artifacts": {name: workflow.file_digest(run_dir / name) for name in names}}
        run_dir.with_suffix(".json").write_text(json.dumps(context))
    caches = {pair: (cache, workflow.file_digest(cache)) for pair in TIME_PAIRS}
    results, audit = workflow._verified_baseline_runs(args, plan, caches)
    assert len(results) == 3 and len(audit["runs"]) == 3
    assert audit["source_variant"] == source_variant and audit["target_protocol"] == target_protocol
    with pytest.raises(ValueError, match="inputs or training settings"):
        workflow._verified_baseline_runs(args, dict(plan, seed=43), caches)
    run_dir = results["baseline", TIME_PAIRS[0]]["stage2_run"]
    saved = json.loads((run_dir / "config.json").read_text())
    (run_dir / "config.json").write_text(json.dumps(dict(saved, lambda_stage1_closure=0)))
    context = json.loads(run_dir.with_suffix(".json").read_text())
    context["artifacts"]["config.json"] = workflow.file_digest(run_dir / "config.json")
    run_dir.with_suffix(".json").write_text(json.dumps(context))
    with pytest.raises(ValueError, match="loss/model configuration"):
        workflow._verified_baseline_runs(args, plan, caches)
    (run_dir / "config.json").write_text(json.dumps(saved))
    context["artifacts"]["config.json"] = workflow.file_digest(run_dir / "config.json")
    run_dir.with_suffix(".json").write_text(json.dumps(context))
    (run_dir / "best_joint.pt").write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        workflow._verified_baseline_runs(args, plan, caches)


def test_round_cli_defaults_to_1b_vs_gap_and_six_workers(tmp_path, monkeypatch):
    from mignet_ce.downstream.loss_ablation.formal import workflow
    from mignet_ce.downstream.loss_ablation.formal.__main__ import main

    captured = []
    monkeypatch.setattr(workflow, "run", lambda args: captured.append(args))
    argv = ["--" + key + "-root" for key in ("data", "cci", "grn", "developmental", "output")]
    argv = [item for key in argv for item in (key, str(tmp_path))]
    assert main(argv) == 0
    assert captured[0].workers == 6 and captured[0].branches == ["branch2a", "branch2b"]
    with pytest.raises(SystemExit):
        main([*argv, "--workers", "7"])


def test_finalize_cli_requires_only_existing_output_root_and_uses_recovery_route(tmp_path, monkeypatch):
    from mignet_ce.downstream.loss_ablation.formal import workflow
    from mignet_ce.downstream.loss_ablation.formal.__main__ import main

    captured = []
    monkeypatch.setattr(workflow, "finalize", lambda args: captured.append(args))
    assert main(["--finalize-only", "--output-root", str(tmp_path)]) == 0
    assert len(captured) == 1 and captured[0].device is None
    assert captured[0].branches is None  # The saved protocol resolves historical defaults.
    for extra in (["--resume"], ["--preview"]):
        with pytest.raises(SystemExit):
            main(["--finalize-only", "--output-root", str(tmp_path), *extra])


def test_workflow_preserves_attempts_and_reuses_verified_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from mignet_ce.downstream.loss_ablation.formal import workflow

    calls = []

    def subprocess_run(command, **kwargs):
        assert command.count("--out-dir") == 1
        run_dir = Path(command[command.index("--out-dir") + 1])
        run_dir.mkdir()
        for name in ("stage1_summary.json", "best_ei.pt", "metrics.csv", "config.json"):
            (run_dir / name).write_text("test artifact")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workflow.subprocess, "run", subprocess_run)
    job = dict(parent=tmp_path / "job", root=tmp_path, stage1=True,
               command=["python", "runner.py", "--out-dir", "unused"],
               cache_sha256="cache", source_sha256=None)
    first = workflow._execute(job)
    assert workflow._execute(job) == first and len(calls) == 1
    (first / "metrics.csv").write_text("damaged artifact")
    second = workflow._execute(job)
    assert second != first and first.exists() and len(calls) == 2


def test_native_cli_loads_shared_cache_and_stage1_checkpoint(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    from joblib.externals import cloudpickle

    def macro(inputs):
        logits = inputs.z_macro_t @ inputs.z_macro_tp.T
        return torch.softmax(logits + 4 * torch.eye(logits.shape[0], device=logits.device), dim=1)

    provenance = {name: str(tmp_path / name) for name in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "grn_t", "grn_tp")}
    cache = tmp_path / "prepared.pkl"
    with cache.open("wb") as stream:
        cloudpickle.dump(replace(_prepared(), macro_pij_builder=macro, provenance=provenance), stream)
    arguments = ["--method", "formal_two_stage", "--k", "2", "--epochs", "4", "--stage1-ratio", ".5",
                 "--hidden-dim", "8", "--mid-dim", "4", "--knn-k", "2", "--prepared-cache", str(cache),
                 "--maturity-t", "test_maturity_t", "--maturity-tp", "test_maturity_tp"]
    for name, value in provenance.items():
        arguments.extend(("--" + name.replace("_", "-"), value))
    repo = Path(__file__).resolve().parents[1]
    runner = [sys.executable, "-B", str(repo / "scripts/run_wyt_deltaei_coarse_grain.py")]
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    for extra in (
        ["--out-dir", str(stage1), "--stage1-only"],
        ["--out-dir", str(stage2), "--stage1-source", str(stage1 / "best_ei.pt"),
         "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1"],
    ):
        result = subprocess.run([*runner, *arguments, *extra], cwd=repo, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((stage2 / "summary.json").read_text())
    assert summary["stage1_checkpoint_sha256"] == hashlib.sha256((stage1 / "best_ei.pt").read_bytes()).hexdigest()


def test_full_workflow_schedules_only_spot_and_exports_all_branches(tmp_path, monkeypatch):
    import csv
    from pathlib import Path
    from types import SimpleNamespace
    import mignet_ce.coarse_frontends as frontends
    from mignet_ce.downstream.loss_ablation.formal import workflow
    from mignet_ce.downstream.loss_ablation.formal.variants import TIME_PAIRS

    args = SimpleNamespace(data_root=tmp_path / "data", cci_root=tmp_path / "cci",
                           grn_root=tmp_path / "grn", developmental_root=tmp_path / "maturity",
                           output_root=tmp_path / "results", seed=42, epochs=1500,
                           device="cuda", workers=6, resume=False)
    parser = build_argparser()
    for pair in TIME_PAIRS:
        parsed = parser.parse_args(workflow.command(args, pair, VARIANTS[0], Path("unused"))[2:])
        for key in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "cci_index_t", "cci_index_tp",
                    "grn_t", "grn_tp", "maturity_t", "maturity_tp"):
            path = getattr(parsed, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture input")
    monkeypatch.setattr(frontends, "prepare_coarse_input", lambda *_: _prepared())
    batches = []
    created = []

    def execute(job):
        run_dir = job["parent"] / "attempt_0001"
        if run_dir.is_dir():
            return run_dir
        created.append(job["stage1"])
        run_dir.mkdir(parents=True)
        (run_dir / "best_ei.pt").write_bytes(b"shared stage1" + job["parent"].name.encode())
        if job["stage1"]:
            (run_dir / "stage1_summary.json").write_text("{}")
        if not job["stage1"]:
            parsed = parser.parse_args(job["command"][2:])
            assert parsed.method == "formal_two_stage" and parsed.k == 40
            assert "seurat" not in str(parsed.h5ad_t)
            source = parsed.stage1_source
            (run_dir / "best_ei.pt").write_bytes(source.read_bytes())
            (run_dir / "best_joint.pt").touch()
            metrics = dict(I_available=.04, I_retained=.02, closure_quality=.5,
                           hardK_t=30, hardK_tp=28, KeffHard_t=25., KeffHard_tp=24.)
            with (run_dir / "metrics.csv").open("w", newline="") as stream:
                row = dict(metrics, stage="stage1", epoch=20, delta_EI=2.)
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            summary = dict(metrics, stage1_best_epoch=20, best_epoch=800,
                           checkpoint_metric_state_consistent=True, selected_checkpoint="best_joint.pt",
                           delta_EI_best_checkpoint=1.9, best_joint_selection="strict_retained_first",
                           best_joint_fallback_used=False, stage1_checkpoint_sha256=workflow.file_digest(source))
            (run_dir / "summary.json").write_text(json.dumps(summary))
        return run_dir

    def parallel(jobs, workers):
        assert workers == 6
        assert len({job["key"][0] for job in jobs}) >= 2
        batches.append(jobs)
        return {job["key"]: workflow._execute_pair(job) for job in jobs}

    monkeypatch.setattr(workflow, "_execute", execute)
    monkeypatch.setattr(workflow, "_parallel", parallel)
    workflow.run(args)
    first_report = args.output_root / "reports" / "branch2a_branch2b"
    assert [len(batch) for batch in batches] == [9]
    assert created.count(True) == 3 and created.count(False) == 9
    assert {job["key"][0] for job in batches[0]} == {"baseline", "branch2a", "branch2b"}
    for pair in TIME_PAIRS:
        base = next(job for job in batches[0] if job["key"] == ("baseline", pair))
        a = next(job for job in batches[0] if job["key"] == ("branch2a", pair))
        b = next(job for job in batches[0] if job["key"] == ("branch2b", pair))
        assert base["stage1_ready"] is a["stage1_ready"] is b["stage1_ready"]
        assert base["stage1_job"]["parent"] == a["stage1_job"]["parent"] == b["stage1_job"]["parent"]
    assert json.loads((first_report / "run_manifest.json").read_text())["workers"] == 6
    with (first_report / "information.csv").open(newline="") as stream:
        first_rows = list(csv.DictReader(stream))
    assert len(first_rows) == 18
    assert {row["method"] for row in first_rows} == {"Baseline (4+5)", "Branch 2A", "Branch 2B"}
    assert "delta_vs_baseline_cq" in first_rows[0]
    assert "delta_vs_previous_cq" in first_rows[0]

    args.resume = True
    args.branches = ["branch2a"]
    workflow.run(args)
    assert len(batches[-1]) == 6
    assert created.count(True) == 3 and created.count(False) == 9
    assert (first_report / "information.png").is_file()
    second_report = args.output_root / "reports" / "branch2a"
    assert json.loads((second_report / "run_manifest.json").read_text())["status"] == "complete"

    args.branches = [variant.name for variant in VARIANTS]
    workflow.run(args)
    assert len(batches[-1]) == 9
    assert created.count(True) == 3 and created.count(False) == 9
    all_report = args.output_root / "reports" / report_name(VARIANTS)
    for filename in ("information.csv", "cluster_occupancy.csv"):
        with (all_report / filename).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 18
    assert (all_report / "information.pdf").is_file()
    assert (all_report / "cluster_occupancy.png").is_file()
    args.seed = 43
    with pytest.raises(ValueError, match="settings differ"):
        workflow.run(args)
    args.seed = 42
    args.branches = ["branch2a"]

    def fail_parallel(*_):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(workflow, "_parallel", fail_parallel)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        workflow.run(args)
    failed = args.output_root / "reports" / "branch2a" / "run_manifest.json"
    assert json.loads(failed.read_text())["status"] == "failed"
    assert json.loads((first_report / "run_manifest.json").read_text())["status"] == "complete"

    original_digest = workflow._training_code_digest()
    monkeypatch.setattr(workflow, "_training_code_digest", lambda: "updated-entrypoint")
    with pytest.raises(ValueError, match="settings differ"):
        workflow.run(args)
    monkeypatch.setattr(workflow, "_training_code_digest", lambda: original_digest)
    args.output_root = tmp_path / "failed_run"
    args.resume = False
    with pytest.raises(RuntimeError, match="simulated training failure"):
        workflow.run(args)
    monkeypatch.setattr(workflow, "_training_code_digest", lambda: "updated-entrypoint")
    args.resume = True
    with pytest.raises(RuntimeError, match="simulated training failure"):
        workflow.run(args)
    updated = json.loads((args.output_root / "run_config.json").read_text())
    assert updated["training_code_sha256"] == "updated-entrypoint"
    backup = args.output_root / "run_config.previous_0001.json"
    assert json.loads(backup.read_text())["training_code_sha256"] == original_digest


def test_real_frontend_cache_uses_nested_grn_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from joblib.externals import cloudpickle
    from test_wyt_two_stage import _write_stage
    from mignet_ce.coarse_frontends import CoarseFrontendRequest, prepare_coarse_input
    from scripts import run_wyt_deltaei_coarse_grain as cli
    import wyt_deltaei_coarse_grain as core

    source = _write_stage(tmp_path, "source", 18, 5)
    target = _write_stage(tmp_path, "target", 21, 6)
    request = CoarseFrontendRequest(
        h5ad_t=source[0], h5ad_tp=target[0], cci_t=source[1], cci_tp=target[1],
        cci_index_t=source[2], cci_index_tp=target[2], grn_t=source[3], grn_tp=target[3],
        maturity_t=source[4], maturity_tp=target[4],
        grn_feature_method="legacy_nmf", cci_feature_method="laplacian_hks", alpha_cci=.20,
        mid_dim=8, nmf_max_iter=3,
    )
    prepared = prepare_coarse_input("formal_two_stage", request)
    assert "grn_t" not in prepared.provenance
    assert prepared.provenance["stage_t"]["grn_path"] == str(source[3].resolve())
    cache = tmp_path / "real_prepared.pkl"
    with cache.open("wb") as stream:
        cloudpickle.dump(prepared, stream)
    captured = []

    def train(loaded, config):
        captured.append(loaded)
        np.testing.assert_array_equal(loaded.micro_pij, prepared.micro_pij)
        return SimpleNamespace(out_dir=config.out_dir, best_epoch=1, final_delta_ei=.1)

    monkeypatch.setattr(core, "train_deltaei", train)
    arguments = ["--method", "formal_two_stage", "--k", "2", "--out-dir", str(tmp_path / "run"),
                 "--prepared-cache", str(cache), "--stage1-only"]
    for suffix, values in (("t", source), ("tp", target)):
        for key, path in zip(("h5ad", "cci", "cci-index", "grn", "maturity"), values):
            arguments.extend((f"--{key}-{suffix}", str(path)))
    cli.main(arguments)
    assert len(captured) == 1
    wrong = list(arguments)
    wrong[wrong.index("--grn-t") + 1] = str(target[3])
    with pytest.raises(ValueError, match="does not match --grn-t"):
        cli.main(wrong)


def test_failed_child_surfaces_log_tail_and_stops_later_methods(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mignet_ce.downstream.loss_ablation.formal import workflow

    def fail(command, **kwargs):
        kwargs["stdout"].write("Traceback\nValueError: actual child error\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(workflow.subprocess, "run", fail)
    job = dict(parent=tmp_path / "baseline", root=tmp_path, stage1=True,
               command=["python", "runner.py", "--out-dir", "unused"],
               cache_sha256="cache", source_sha256=None)
    with pytest.raises(RuntimeError, match="actual child error"):
        workflow._execute(job)
    assert json.loads((job["parent"] / "attempt_0001.json").read_text())["status"] == "failed"


def test_round_selection_keeps_baseline_and_stage1_dependencies():
    first = select_variants()
    assert [variant.name for variant in first] == ["baseline", "branch2a", "branch2b"]
    assert [variant.name for variant in required_stage1_variants(first)] == ["baseline"]
    later = select_variants(["branch2b", "branch2a", "branch2a"])
    assert [variant.name for variant in later] == ["baseline", "branch2a", "branch2b"]
    assert [variant.name for variant in required_stage1_variants(later)] == ["baseline"]
    assert [variant.name for variant in select_variants(["branch2b"])] == ["baseline", "branch2a", "branch2b"]
    assert [variant.name for variant in select_variants(protocol=FORMAL_TWO_STAGE_GAP_PROTOCOL)] == ["baseline", "branch4", "branch5"]
    assert [variant.name for variant in select_variants(protocol=FORMAL_TWO_STAGE_1B_PROTOCOL)] == ["baseline", "branch3"]
    with pytest.raises(ValueError, match="Unknown formal branches"):
        select_variants(["unknown"])


def test_baseline_differences_match_stage_pair_and_endpoint(tmp_path):
    import csv
    from mignet_ce.downstream.loss_ablation.plots import _metric_text, preview_rows
    from mignet_ce.downstream.loss_ablation.tables import baseline_differences, write_tables

    rows = preview_rows(("spot",), ("Formal baseline", "Branch 1A"))
    baseline = rows[0]
    branch = next(row for row in rows if row["method"] == "Branch 1A" and row["stage"] == baseline["stage"]
                  and row["source_time"] == baseline["source_time"] and row["target_time"] == baseline["target_time"])
    branch.update(delta_ei_bits=baseline["delta_ei_bits"] + .1,
                  cq=baseline["cq"] - .08,
                  hard_k_source=baseline["hard_k_source"] + 2,
                  hard_k_target=baseline["hard_k_target"] - 1,
                  keff_hard_source=baseline["keff_hard_source"] + 1.25,
                  keff_hard_target=baseline["keff_hard_target"] - .6)
    values = baseline_differences(rows, "Formal baseline")
    comparison = values[rows.index(branch)]
    assert comparison["delta_vs_baseline_delta_ei_bits"] == pytest.approx(.1)
    assert comparison["delta_vs_baseline_cq"] == pytest.approx(-.08)
    assert _metric_text(comparison, "delta_ei_bits", False, relative=True) == "+0.1000"
    assert _metric_text(comparison, "cq", False, relative=True) == "-0.0800"
    assert _metric_text(comparison, "hard_k", True, relative=True) == "+2 / -1"
    assert _metric_text(comparison, "keff_hard", True, relative=True) == "+1.25 / -0.60"
    assert _metric_text(values[0], "cq", False, relative=True) == "0.0000"
    assert "delta_vs_baseline_cq" not in baseline
    write_tables(rows, tmp_path, baseline_method="Formal baseline")
    with (tmp_path / "information.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert float(exported[rows.index(branch)]["cq"]) == branch["cq"]
    assert float(exported[rows.index(branch)]["delta_vs_baseline_cq"]) == pytest.approx(-.08)
    with pytest.raises(ValueError, match="Missing baseline"):
        baseline_differences([row for row in rows if row is not baseline], "Formal baseline")
    with pytest.raises(ValueError, match="Duplicate baseline"):
        baseline_differences([*rows, baseline], "Formal baseline")


def test_training_fingerprint_excludes_reports_but_covers_model(tmp_path, monkeypatch):
    from mignet_ce.downstream.loss_ablation.formal import workflow

    runner = tmp_path / "scripts/run_wyt_deltaei_coarse_grain.py"
    model = tmp_path / "wyt_deltaei_coarse_grain/model.py"
    frontend = tmp_path / "mignet_ce/coarse_frontends/formal_two_stage.py"
    plot = tmp_path / "mignet_ce/downstream/loss_ablation/plots.py"
    for path in (runner, model, frontend, plot):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first")
    monkeypatch.setattr(workflow, "REPO", tmp_path)
    monkeypatch.setattr(workflow, "RUNNER", runner)
    before = workflow._training_code_digest()
    plot.write_text("presentation change")
    assert workflow._training_code_digest() == before
    model.write_text("model change")
    assert workflow._training_code_digest() != before
