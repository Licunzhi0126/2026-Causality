"""Behavioral coverage for domain-owned scheduling and shared training state."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mignet_ce.downstream.domain_visualization import workflow
from mignet_ce.downstream.domain_visualization.metrics import read_run
from mignet_ce.downstream.domain_visualization.variants import VARIANTS, TIME_PAIRS
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.training_config import build_training_config
from wyt_deltaei_coarse_grain.two_stage_trainer import (
    WYTFormalTwoStageConfig, formal_stage1_signature, train_deltaei_two_stage,
)


def _args(tmp_path):
    return SimpleNamespace(data_root=tmp_path / "data", cci_root=tmp_path / "cci", grn_root=tmp_path / "grn",
                           developmental_root=tmp_path / "maturity", seed=42, epochs=1500, device="cpu", workers=6,
                           experiment="single_factor")


def _macro(inputs):
    logits = inputs.z_macro_t @ inputs.z_macro_tp.T
    return torch.softmax(logits + 4 * torch.eye(logits.shape[0], device=logits.device), dim=1)


def _prepared():
    def macro(inputs):
        # A local function is serialized by value for the actual subprocess test.
        logits = inputs.z_macro_t @ inputs.z_macro_tp.T
        return torch.softmax(logits + 4 * torch.eye(logits.shape[0], device=logits.device), dim=1)

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
        micro_pij=micro.astype("float32"), micro_ei=0.0, macro_pij_builder=macro,
        maturity_t=np.linspace(0, 1, 6, dtype="float32"), maturity_tp=np.linspace(0, 1, 7, dtype="float32"),
        provenance={"loss_protocol": "formal_two_stage_main"},
    )


def test_only_intended_factors_change_and_stage1_fingerprints_match(tmp_path):
    args = _args(tmp_path)
    configs = [build_training_config(workflow.parsed_command(args, TIME_PAIRS[0], variant)) for variant in VARIANTS]
    base = configs[0]
    expected = ({"lambda_stage1_closure", "lambda_closure"}, {"lambda_dev"}, {"lambda_ei_constraint"},
                {"lambda_retain_norm"}, {"lambda_retain_floor"})
    for config, changed in zip(configs[1:], expected):
        assert {name for name in vars(base) if getattr(base, name) != getattr(config, name)} == changed
    signatures = [formal_stage1_signature(_prepared(), config) for config in configs]
    assert len(set(signatures)) == 3
    assert signatures[0] == signatures[3] == signatures[4] == signatures[5]
    for config in configs:
        assert config.lambda_gap == config.ei_loss_retain_ratio == 1
        assert config.ei_retain_ratio == .85 and config.retain_floor_ratio == config.checkpoint_retain_ratio == .5


def test_pair_tasks_share_once_and_continue_without_global_barrier(tmp_path, monkeypatch):
    args = _args(tmp_path)
    caches = {pair: (tmp_path / f"cache_{index}.pkl", "same") for index, pair in enumerate(TIME_PAIRS)}
    jobs = workflow.build_jobs(args, tmp_path, caches)
    assert len(jobs) == 18
    assert len({id(job["stage1_ready"]) for job in jobs}) == 9
    for job in jobs:
        for stage in ("stage1_job", "stage2_job"):
            job[stage]["parent"].mkdir(parents=True, exist_ok=True)
    blocked = Event()
    other_started = Event()
    fast_stage2 = Event()
    guard = Lock()
    calls = []

    def execute(job):
        with guard:
            calls.append(job["parent"])
        if job["stage1"] and job["parent"].parts[-2:] == ("closure", "11.5_to_12.5"):
            other_started.set()
            assert blocked.wait(10)
        out = job["parent"] / "attempt_0001"
        out.mkdir(parents=True)
        (out / "best_ei.pt").write_bytes(b"shared state")
        if not job["stage1"]:
            assert "--stage1-source" in job["command"]
            (out / "summary.json").write_text(json.dumps({"stage1_checkpoint_sha256": job["source_sha256"]}))
            fast_stage2.set()
        return out

    monkeypatch.setattr(workflow, "_execute", execute)
    monkeypatch.setattr(workflow, "FINAL_ARTIFACTS", ("summary.json", "best_ei.pt"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        slow = pool.submit(workflow._execute_pair, jobs[3])
        assert other_started.wait(5)
        baseline = pool.submit(workflow._execute_pair, jobs[0])
        branch = pool.submit(workflow._execute_pair, jobs[9])
        try:
            assert fast_stage2.wait(5), "An unrelated Stage 1 blocked ready Stage 2 tasks"
            first, second = baseline.result(5), branch.result(5)
            assert first["stage1_run"] == second["stage1_run"]
            assert calls.count(jobs[0]["stage1_job"]["parent"]) == 1
        finally:
            blocked.set()
        slow.result(5)


def test_failed_attempts_are_preserved_and_complete_artifacts_are_checked(tmp_path, monkeypatch):
    parent = tmp_path / "runs"
    job = {"parent": parent, "stage1": True, "command": ["python", "runner", "--out-dir", "unused"],
           "cache_sha256": "input", "source_sha256": None}
    monkeypatch.setattr(workflow, "STAGE1_ARTIFACTS", ("best_ei.pt",))
    calls = []

    def child(command, **kwargs):
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir()
        calls.append(out)
        (out / "best_ei.pt").write_bytes(b"weights")
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(workflow.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="Training exited"):
        workflow._execute(job)
    failed = (parent / "attempt_0001.json").read_bytes()
    second = workflow._execute(job)
    assert second.name == "attempt_0002" and workflow._execute(job) == second
    assert len(calls) == 2 and (parent / "attempt_0001.json").read_bytes() == failed
    (second / "best_ei.pt").write_bytes(b"corrupt")
    assert workflow._execute(job).name == "attempt_0003"


def test_real_short_training_of_all_six_versions_uses_correct_stage1_and_exports_metrics(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prepared = _prepared()
        configs = {variant.name: WYTFormalTwoStageConfig(
            k=2, out_dir=tmp_path / variant.name, hidden_dim=8, mid_dim=4,
            epochs=4, stage1_ratio=.5, knn_k=2, log_every=1, **variant.weights,
        ) for variant in VARIANTS}
        sources = {}
        for name in ("baseline", "closure", "development"):
            config = replace(configs[name], out_dir=tmp_path / "stage1" / name, stage1_only=True)
            train_deltaei_two_stage(prepared, config)
            sources[name] = config.out_dir / "best_ei.pt"
        for variant in VARIANTS:
            config = replace(configs[variant.name], stage1_source=sources[variant.stage1_group])
            train_deltaei_two_stage(prepared, config)
            row = read_run(config.out_dir, variant=variant.name, pair=TIME_PAIRS[0], k=2)
            assert np.isfinite(row["delta_ei"])
            assert (config.out_dir / "best_ei.pt").read_bytes() == sources[variant.stage1_group].read_bytes()
        incompatible = replace(configs["closure"], out_dir=tmp_path / "wrong_source", stage1_source=sources["baseline"])
        with pytest.raises(ValueError, match="Stage 1"):
            train_deltaei_two_stage(prepared, incompatible)
    finally:
        torch.set_num_threads(previous_threads)


def test_actual_worker_subprocesses_accept_cache_and_write_all_selected_artifacts(tmp_path):
    from concurrent.futures import Future
    from joblib.externals import cloudpickle

    args = _args(tmp_path)
    args.epochs = 4
    parsed = workflow.parsed_command(args, TIME_PAIRS[0], VARIANTS[0])
    prepared = replace(_prepared(), provenance={
        **_prepared().provenance, **{name: str(getattr(parsed, name))
                                    for name in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "grn_t", "grn_tp")},
    })
    cache = tmp_path / "prepared.pkl"
    with cache.open("wb") as handle:
        cloudpickle.dump(prepared, handle)
    base_command = workflow.command(args, TIME_PAIRS[0], VARIANTS[0], Path("unused"))
    base_command[base_command.index("--k") + 1] = "2"
    base_command.extend(("--hidden-dim", "8", "--mid-dim", "4", "--knn-k", "2", "--log-every", "1",
                         "--stage1-ratio", ".5", "--prepared-cache", str(cache)))
    stage1 = dict(stage1=True, parent=tmp_path / "s1", command=[*base_command, "--stage1-only"],
                  cache_sha256=workflow.file_digest(cache))
    stage2 = dict(stage1=False, parent=tmp_path / "s2", command=base_command, cache_sha256=stage1["cache_sha256"])
    result = workflow._execute_pair(dict(key=("baseline", TIME_PAIRS[0]), stage1_job=stage1, stage2_job=stage2,
                                         stage1_lock=Lock(), stage1_ready=Future()))
    assert set(result["artifacts"]) == set(workflow.FINAL_ARTIFACTS)
    row = read_run(Path(result["stage2_run"]), variant="baseline", pair=TIME_PAIRS[0], k=2)
    assert row["k"] == 2 and np.isfinite(row["delta_ei"])


def test_workflow_preserves_previews_records_all_tasks_and_rejects_changed_resume(tmp_path, monkeypatch):
    from mignet_ce import coarse_frontends

    args = _args(tmp_path)
    args.output_root, args.mode = tmp_path / "experiment", "train"
    preview = args.output_root / "previews"
    preview.mkdir(parents=True)
    (preview / "keep.txt").write_text("existing preview")
    plan = {"input_sha256": {}, "stage1_groups": {item.name: item.stage1_group for item in VARIANTS}, "seed": 42}
    monkeypatch.setattr(workflow, "preflight", lambda args: plan)
    monkeypatch.setattr(coarse_frontends, "prepare_coarse_input", lambda *args: _prepared())
    calls = []

    def task(job):
        calls.append(job["key"])
        return dict(variant=job["key"][0], time_pair=list(job["key"][1]))

    monkeypatch.setattr(workflow, "_execute_pair", task)
    workflow.run(args)
    manifest = json.loads((args.output_root / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and len(set(calls)) == 18
    assert (preview / "keep.txt").read_text() == "existing preview"
    config_bytes = (args.output_root / "run_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.run(args)
    args.mode = "resume"
    monkeypatch.setattr(coarse_frontends, "prepare_coarse_input", lambda *args: pytest.fail("Resume should reuse verified input caches"))
    workflow.run(args)
    assert (args.output_root / "run_config.json").read_bytes() == config_bytes
    plan = dict(plan, seed=43)
    with pytest.raises(ValueError, match="Resume requires identical"):
        workflow.run(args)


def test_preflight_missing_formal_inputs_does_not_create_experiment_output(tmp_path):
    args = _args(tmp_path)
    args.output_root = tmp_path / "not_created"
    with pytest.raises(FileNotFoundError, match="Missing formal training inputs"):
        workflow.preflight(args)
    assert not args.output_root.exists()


def test_factorial_defaults_replace_formal_weights_and_have_four_exact_stage1_groups(tmp_path):
    from mignet_ce.downstream.domain_visualization.variants import FACTORIAL_VARIANTS
    from mignet_ce.coarse_frontends import get_coarse_method_spec

    args = _args(tmp_path)
    args.experiment = "factorial"
    variants = FACTORIAL_VARIANTS
    configs = [build_training_config(workflow.parsed_command(args, TIME_PAIRS[0], variant)) for variant in variants]
    default = WYTFormalTwoStageConfig(k=40, out_dir=Path("unused"))
    assert configs[-1] == default
    for key, value in variants[-1].weights.items():
        assert getattr(default, key) == value
    parsed = workflow.parsed_command(args, TIME_PAIRS[0], variants[-1])
    parsed.lambda_dev = None
    assert build_training_config(parsed).lambda_dev == get_coarse_method_spec("formal_two_stage").default_lambda_dev == 2
    assert default.objective_version == "formal_two_stage_main"
    signatures = [formal_stage1_signature(_prepared(), config) for config in configs]
    assert len(set(signatures)) == 4
    for start in (0, 2, 4, 6):
        assert signatures[start] == signatures[start + 1]
    caches = {pair: (tmp_path / f"cache{index}.pkl", "digest") for index, pair in enumerate(TIME_PAIRS)}
    jobs = workflow.build_jobs(args, tmp_path, caches)
    assert len(jobs) == 24 and len({id(job["stage1_ready"]) for job in jobs}) == 12


def test_all_eight_factorial_runs_match_confirmed_formulas_and_share_frozen_references(tmp_path):
    from mignet_ce.downstream.domain_visualization.variants import FACTORIAL_VARIANTS
    from mignet_ce.downstream.domain_visualization.metrics import history_diagnostics

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prepared = _prepared()
        sources = {}
        for variant in FACTORIAL_VARIANTS:
            config = WYTFormalTwoStageConfig(k=2, out_dir=tmp_path / variant.name, hidden_dim=8, mid_dim=4,
                                            epochs=4, stage1_ratio=.5, knn_k=2, log_every=1, **variant.weights)
            if variant.stage1_group not in sources:
                source_cfg = replace(config, out_dir=tmp_path / "stage1" / variant.stage1_group, stage1_only=True)
                train_deltaei_two_stage(prepared, source_cfg)
                sources[variant.stage1_group] = source_cfg.out_dir / "best_ei.pt"
            config = replace(config, stage1_source=sources[variant.stage1_group])
            result = train_deltaei_two_stage(prepared, config)
            assert (config.out_dir / "best_ei.pt").read_bytes() == config.stage1_source.read_bytes()
            for row in result.metrics:
                expected = config.lambda_dev * row["L_dev"] + row["L_gap"]
                if row["stage"] == "stage1":
                    expected += -row["delta_EI"] + config.lambda_stage1_closure * row["L_closure"]
                else:
                    expected += row["pair_weight"] * (config.lambda_closure * row["L_closure"]
                                                       + config.lambda_retain_norm * row["L_retain_norm"]
                                                       + row["L_retain_floor"])
                    expected += row["L_ei_constraint"]
                    assert row["EI_loss_floor"] == row["EI_reference"]
                    assert row["EI_checkpoint_floor"] == pytest.approx(.85 * row["EI_reference"])
                assert row["loss"] == pytest.approx(expected, abs=2e-6)
            record = read_run(config.out_dir, variant=variant.name, pair=TIME_PAIRS[0], k=2)
            assert np.isfinite(record["closure_js_mean"]) and record["dynamic_pair_weight"] in (1., .1)
            audit = history_diagnostics(config.out_dir, variant=variant.name, pair=TIME_PAIRS[0])
            assert audit["stage2_epochs"] == 2
    finally:
        torch.set_num_threads(old_threads)
