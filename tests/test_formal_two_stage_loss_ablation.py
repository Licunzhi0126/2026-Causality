"""Regression coverage for common Stage 1 state and independent formal branches."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mignet_ce.coarse_frontends import get_coarse_method_spec
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from mignet_ce.downstream.loss_ablation.formal.variants import (
    VARIANTS, STAGE1_VARIANTS, report_name, required_stage1_variants, select_variants,
)
from scripts.run_wyt_deltaei_coarse_grain import build_argparser
from wyt_deltaei_coarse_grain.training_config import build_training_config
from wyt_deltaei_coarse_grain.two_stage_objective import occupancy_gap_loss
from wyt_deltaei_coarse_grain.two_stage_trainer import (
    WYTFormalTwoStageConfig, formal_stage1_signature, train_deltaei_two_stage,
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
    assert len(STAGE1_VARIANTS) == 5
    base = configs["baseline"]
    assert get_coarse_method_spec("formal_two_stage").requires_maturity
    assert base.lambda_align == 1 and base.lambda_stage1_closure == 0
    assert base.lambda_keff == 1 and base.lambda_dev == .05
    assert all(getattr(base, "lambda_" + key) == 0 for key in (
        "var", "local", "proto", "dead_usage", "sharp", "within", "inter",
    ))
    assert configs["branch2b"].lambda_retain_norm == .5
    assert configs["branch2b"].lambda_retain_floor == 0 and configs["branch2b"].lambda_stage2_cq == 1
    assert configs["branch3"].lambda_gap == 1 and configs["branch3"].lambda_keff == 0
    assert configs["branch4"].lambda_align == 1 and configs["branch4"].lambda_dev == 1
    assert configs["branch5"].ei_loss_retain_ratio == 1 and configs["branch5"].ei_retain_ratio == .85
    prepared = _prepared()
    reordered = replace(prepared, unit_ids_t=list(reversed(prepared.unit_ids_t)))
    assert formal_stage1_signature(prepared, base) != formal_stage1_signature(reordered, base)
    for name in ("branch2a", "branch2b", "branch5"):
        assert formal_stage1_signature(prepared, configs[name]) == formal_stage1_signature(prepared, base)
    assert formal_stage1_signature(prepared, configs["branch4"]) != formal_stage1_signature(prepared, base)


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
                                  epochs=4, stage1_ratio=.5, knn_k=2, log_every=1, stage1_only=True)
    train_deltaei_two_stage(prepared, base)
    source = base.out_dir / "best_ei.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    reference = train_deltaei_two_stage(prepared, replace(base, out_dir=tmp_path / "full", stage1_only=False))
    for name, overrides in (
        ("baseline", {}),
        ("branch2b", {"lambda_retain_floor": 0., "lambda_stage2_cq": 1.}),
        ("branch5", {"ei_loss_retain_ratio": 1.}),
    ):
        config = replace(base, out_dir=tmp_path / name, stage1_only=False, stage1_source=source, **overrides)
        result = train_deltaei_two_stage(prepared, config)
        assert result.metrics[:2] == checkpoint["stage1_history"]
        if name == "baseline":
            assert result.metrics == reference.metrics
        summary = json.loads((config.out_dir / "summary.json").read_text())
        assert summary["stage1_checkpoint_sha256"] == digest
        assert summary["stage1_best_epoch"] == checkpoint["epoch"]
        assert summary["ei_checkpoint_retain_ratio"] == .85
        for row in result.metrics[2:]:
            assert row["EI_checkpoint_floor"] == pytest.approx(.85 * checkpoint["best_delta_ei"])
            assert row["EI_loss_floor"] == pytest.approx(config.ei_loss_retain_ratio * checkpoint["best_delta_ei"])
            expected = row["L_align"] + .05 * row["L_dev"] + row["L_keff"] + 5 * row["L_ei_constraint"]
            expected += row["weighted_closure"] + row["weighted_retain_norm"]
            expected += row["weighted_retain_floor"] + row["weighted_stage2_cq"]
            assert row["loss"] == pytest.approx(expected, abs=1e-6)
        selected = torch.load(config.out_dir / "best_joint.pt", map_location="cpu", weights_only=False)
        assert len(selected["joint_score"]) == 3
    with pytest.raises(ValueError, match="fingerprint"):
        train_deltaei_two_stage(prepared, replace(base, out_dir=tmp_path / "mismatch", stage1_only=False,
                                                stage1_source=source, lambda_dev=1.0))


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
                           device="cuda", workers=4, resume=False)
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

    def parallel(jobs, workers):
        assert workers == 4
        batches.append(jobs)
        results = {}
        new_jobs = 0
        for job in jobs:
            run_dir = job["parent"] / "attempt_0001"
            if run_dir.is_dir():
                results[job["key"]] = run_dir
                continue
            new_jobs += 1
            run_dir.mkdir(parents=True)
            (run_dir / "best_ei.pt").write_bytes(b"shared stage1" + str(job["key"][1]).encode())
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
            results[job["key"]] = run_dir
        created.append(new_jobs)
        return results

    monkeypatch.setattr(workflow, "_parallel", parallel)
    workflow.run(args)
    first_report = args.output_root / "reports" / "branch1a_branch1b"
    assert [len(batch) for batch in batches] == [9, 9]
    assert created == [9, 9]
    with (first_report / "information.csv").open(newline="") as stream:
        first_rows = list(csv.DictReader(stream))
    assert len(first_rows) == 18
    assert {row["method"] for row in first_rows} == {"Formal baseline", "Branch 1A", "Branch 1B"}
    assert "delta_vs_baseline_cq" in first_rows[0]

    args.resume = True
    args.branches = ["branch2a", "branch2b"]
    workflow.run(args)
    assert [len(batch) for batch in batches[-2:]] == [3, 9]
    assert created[-2:] == [0, 6]
    assert (first_report / "information.png").is_file()
    second_report = args.output_root / "reports" / "branch2a_branch2b"
    assert json.loads((second_report / "run_manifest.json").read_text())["status"] == "complete"

    args.branches = [variant.name for variant in VARIANTS]
    workflow.run(args)
    assert [len(batch) for batch in batches[-2:]] == [15, 24]
    assert created[-2:] == [6, 9]
    all_report = args.output_root / "reports" / report_name(VARIANTS)
    for filename in ("information.csv", "cluster_occupancy.csv"):
        with (all_report / filename).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 48
    assert (all_report / "information.pdf").is_file()
    assert (all_report / "cluster_occupancy.png").is_file()
    args.seed = 43
    with pytest.raises(ValueError, match="settings differ"):
        workflow.run(args)
    args.seed = 42
    args.branches = ["branch1a"]

    def fail_parallel(*_):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(workflow, "_parallel", fail_parallel)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        workflow.run(args)
    failed = args.output_root / "reports" / "branch1a" / "run_manifest.json"
    assert json.loads(failed.read_text())["status"] == "failed"
    assert json.loads((first_report / "run_manifest.json").read_text())["status"] == "complete"


def test_round_selection_keeps_baseline_and_stage1_dependencies():
    first = select_variants()
    assert [variant.name for variant in first] == ["baseline", "branch1a", "branch1b"]
    assert required_stage1_variants(first) == first
    later = select_variants(["branch2b", "branch2a", "branch2a"])
    assert [variant.name for variant in later] == ["baseline", "branch2a", "branch2b"]
    assert [variant.name for variant in required_stage1_variants(later)] == ["baseline"]
    assert [variant.name for variant in select_variants(["branch5"])] == ["baseline", "branch5"]
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
