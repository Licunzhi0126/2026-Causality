from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mignet_ce.downstream.generalization_audit import runner, tables
from scripts.run_generalization_audit import build_parser


def config(tmp_path, **overrides):
    values = dict(method="two_stage", data_root=tmp_path / "inputs",
                  developmental_root=tmp_path / "developmental", output_root=tmp_path / "reports",
                  organs=("lung", "brain"), pairs=("11.5:12.5",),
                  hierarchies=("k150_to_k40",), epochs=4, dpi=60)
    values.update(overrides)
    return runner.GeneralizationConfig(**values)


def materialize_inputs(cfg):
    for job in runner.planned_jobs(cfg):
        for raw in job["inputs"].values():
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-input-not-real-h5ad")


def saved_run(cfg, job, run_dir, *, informative=False, full_config=False):
    """A mathematically consistent persisted-result fixture, not a trained model."""
    run_dir.mkdir(parents=True)
    k = job["K"]
    micro = np.full((k + 3, k + 5), 1 / (k + 5), dtype=np.float32)
    if informative:
        micro *= .3
        micro[np.arange(k + 3), np.arange(k + 3) % k] += .7
    macro = (np.eye(k) * .8 + .2 / k).astype(np.float32)
    st = np.eye(k, dtype=np.float32)[np.arange(k + 3) % k]
    sp = np.eye(k, dtype=np.float32)[np.arange(k + 5) % k]
    for name, value in (("PIJ_micro_train", micro), ("PIJ_macro_train", macro), ("S_t", st), ("S_tp", sp)):
        np.save(run_dir / f"{name}.npy", value)
    ei_micro, _ = tables._matrix_stats(micro)
    ei_macro, _ = tables._matrix_stats(macro)
    summary = {"method": cfg.canonical_method, "K": k, "best_epoch": 2,
               "EI_micro_fixed": ei_micro, "EI_macro_best_checkpoint": ei_macro,
               "delta_EI_best_checkpoint": ei_macro - ei_micro}
    for label, assignment in (("t", st), ("tp", sp)):
        usage = assignment.mean(0).astype(float)
        usage /= usage.sum()
        summary[f"Keff_{label}"] = float(np.exp(-np.sum(usage * np.log(usage))))
    trainer = {"k": k, "epochs": cfg.epochs, "seed": cfg.seed, "checkpoint_keff_min": 1e-6}
    if full_config:
        trainer = runner.expected_trainer_config(cfg, job)
        trainer["out_dir"] = str(run_dir)
    provenance = {"h5ad_t": job["inputs"]["h5ad_t"], "h5ad_tp": job["inputs"]["h5ad_tp"],
                  "grn_feature_method": "legacy_nmf", "cci_feature_method": "laplacian_hks",
                  "alpha_cci": .20, "nmf_components": 5, "nmf_max_iter_used": cfg.nmf_max_iter}
    for side in ("t", "tp"):
        for key in ("cci", "cci_index"):
            provenance[f"{key}_{side}"] = job["inputs"][f"{key}_{side}"]
        provenance[f"stage_{side}"] = {"grn_path": job["inputs"][f"grn_{side}"]}
    input_manifest = {"maturity_provenance": {
        side: {"path": job["inputs"][f"maturity_{side}"], "id_column": "unit_id", "maturity_column": "pseudotime",
               "normalization": "minmax", "direction": "higher_is_more_mature"} for side in ("t", "tp")}}
    (run_dir / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    for name, obj in (("summary", summary), ("config", trainer),
                      ("feature_manifest", {"method": cfg.canonical_method, "provenance": provenance})):
        (run_dir / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    checkpoint = "best_joint.pt"
    (run_dir / checkpoint).touch()

    if informative:
        budget = tables.induced_cq_from_saved(run_dir)
        summary.update(I_available=budget["I_available_bits"], I_retained=budget["I_retained_bits"],
                       closure_quality=budget["CQ_induced"])
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def fast_reports(monkeypatch):
    # State/CSV validation remains real; only repeated image exports are omitted.
    monkeypatch.setattr(tables, "_render", lambda *args, **kwargs: None)


def test_explicit_hierarchy_and_method_commands(tmp_path):
    cfg = config(tmp_path, pairs=("11.5:12.5", "12.5:13.5", "11.5:13.5"), hierarchies=tuple(runner.HIERARCHIES))
    cfg.validate()
    jobs = runner.planned_jobs(cfg)
    assert len(jobs) == 18
    assert {j["organ"] for j in jobs} == {"lung", "brain"}
    assert {j["column"]: (j["input_scale"], j["K"]) for j in jobs} == {
        "Spot -> K150": ("spot", 150), "K150 -> K40": ("seurat_k150", 40), "Spot -> K40": ("spot", 40)}
    command = runner.training_command(cfg, jobs[0], tmp_path / "attempt")
    assert command[command.index("--method") + 1] == "maturity_cci_grn_two_stage"
    assert "--checkpoint-keff-min" not in command
    assert "--keff-min" not in command
    assert cfg.method_root.name == "two_stage"


@pytest.mark.parametrize("extra", [("--method=two_stage",), ("--k", "2"), ("--alpha-cci", ".8"), ("--h5ad-t", "x")])
def test_extra_args_cannot_override_identity(tmp_path, extra):
    with pytest.raises(ValueError, match="cannot override"):
        config(tmp_path, runner_extra_args=extra).validate()


def test_abbreviated_override_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="abbreviated"):
        config(tmp_path, runner_extra_args=("--meth", "two_stage")).validate()


def test_cli_defaults_and_dry_run_write_nothing(tmp_path):
    args = build_parser().parse_args(["--method", "two_stage", "--output-root", str(tmp_path / "reports")])
    assert args.organs == ["lung", "brain"]
    cfg = config(tmp_path)
    assert runner.run(cfg, dry_run=True)["status"] == "partial"
    assert not cfg.output_root.exists()


def test_streamed_ei_matches_existing_formula_and_rejects_bad_arrays():
    from mignet_ce.metrics import effective_information
    rng = np.random.default_rng(43)
    matrix = rng.random((600, 1800))
    matrix /= matrix.sum(axis=1, keepdims=True)
    actual, _ = tables._matrix_stats(matrix)
    assert actual == pytest.approx(effective_information(matrix.copy()), abs=1e-7)
    with pytest.raises(ValueError, match="row-stochastic"):
        tables._matrix_stats(np.array([[float("nan"), 0.], [.5, .5]]))


@pytest.mark.parametrize("method", ["two_stage", "maturity_cci_grn_two_stage"])
def test_independent_saved_kernel_and_keff_audit(tmp_path, method):
    cfg = config(tmp_path, method=method)
    job = runner.planned_jobs(cfg)[0]
    run_dir = cfg.method_root / "runs" / job["job_id"] / "attempt_0001"
    job["run_dir"] = str(run_dir)
    saved_run(cfg, job, run_dir)
    result = tables.inspect_training_run(job, cfg.contract(), cfg.method_root)
    assert result["trained_delta_EI"] > 0
    bad = json.loads((run_dir / "summary.json").read_text())
    bad["delta_EI_best_checkpoint"] += .1
    (run_dir / "summary.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="disagrees"):
        tables.inspect_training_run(job, cfg.contract(), cfg.method_root)


def test_oom_still_exports_real_table_and_preserves_lung_result(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    materialize_inputs(cfg)
    jobs = runner.planned_jobs(cfg)

    def execute(command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(command[command.index("--out-dir") + 1])
        job = next(j for j in jobs if j["inputs"]["h5ad_t"] == command[command.index("--h5ad-t") + 1])
        if job["organ"] == "brain":
            log.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory")
            return 1
        saved_run(cfg, job, run_dir)
        log.write_text("completed")
        return 0

    monkeypatch.setattr(runner, "execute_job", execute)
    result = runner.run(cfg)
    assert result["status"] == "partial"
    state = json.loads((cfg.method_root / "manifest.json").read_text(encoding="utf-8"))
    assert [j["status"] for j in state["jobs"]] == ["success", "oom"]
    table = cfg.method_root / "tables/table_generalization_train_deltaei"
    for suffix in (".csv", ".png", ".pdf"):
        assert table.with_suffix(suffix).stat().st_size > 0
    with table.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["K150 -> K40"] != "—"
    assert rows[1]["K150 -> K40"] == "—"
    assert "CUDA out of memory" in (cfg.method_root / "failures.csv").read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("returncode,text,status", [(137, "", "killed"), (-9, "", "killed"),
                                                     (1, "MemoryError", "oom"), (1, "bad input", "failed")])
def test_exit137_alone_does_not_prove_oom(tmp_path, returncode, text, status):
    log = tmp_path / "child.log"
    log.write_text(text)
    assert runner.classify_failure(returncode, log)[0] == status


def test_resume_reuses_audited_run_and_preserves_failed_attempts(tmp_path, monkeypatch, fast_reports):
    cfg = config(tmp_path, organs=("lung",))
    materialize_inputs(cfg)
    job = runner.planned_jobs(cfg)[0]
    calls = []

    def execute(command, log):
        run_dir = Path(command[command.index("--out-dir") + 1])
        calls.append(run_dir)
        if len(calls) == 1:
            run_dir.mkdir(parents=True)
            (run_dir / "partial.txt").write_text("keep this")
            return 1
        saved_run(cfg, job, run_dir)
        return 0

    monkeypatch.setattr(runner, "execute_job", execute)
    assert runner.run(cfg)["status"] == "partial"
    assert runner.run(cfg, mode="resume")["status"] == "complete"
    assert calls[0].name == "attempt_0001" and calls[1].name == "attempt_0002"
    assert (calls[0] / "partial.txt").read_text() == "keep this"
    assert runner.run(cfg, mode="resume")["status"] == "complete"
    assert len(calls) == 2
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.run(replace(cfg, seed=43), mode="resume")
    with pytest.raises(FileExistsError):
        runner.run(cfg)


def test_missing_inputs_and_render_without_retraining(tmp_path, monkeypatch, fast_reports):
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, "execute_job", lambda *a: pytest.fail("Missing inputs must not start training"))
    assert runner.run(cfg)["status"] == "partial"
    assert runner.run(cfg, mode="render")["status"] == "partial"
    state = json.loads((cfg.method_root / "manifest.json").read_text(encoding="utf-8"))
    assert all(j["status"] == "missing_input" for j in state["jobs"])
    assert all(row["K150 -> K40"] == "—" for row in tables.wide_rows(state))


def test_interrupt_still_records_partial_table(tmp_path, monkeypatch, fast_reports):
    cfg = config(tmp_path)
    materialize_inputs(cfg)

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "execute_job", interrupt)
    assert runner.run(cfg)["status"] == "interrupted"
    state = json.loads((cfg.method_root / "manifest.json").read_text(encoding="utf-8"))
    assert state["jobs"][0]["status"] == "interrupted"
    assert (cfg.method_root / "tables/table_generalization_train_deltaei.csv").exists()


def test_real_child_memoryerror_does_not_abort_controller(tmp_path, monkeypatch, fast_reports):
    import sys
    cfg = config(tmp_path, organs=("brain",))
    materialize_inputs(cfg)
    monkeypatch.setattr(runner, "training_command",
                        lambda *a: [sys.executable, "-c", "raise MemoryError('controlled test failure')"])
    result = runner.run(cfg)
    assert result["status"] == "partial"
    state = json.loads((cfg.method_root / "manifest.json").read_text(encoding="utf-8"))
    assert state["jobs"][0]["status"] == "oom"
    assert state["jobs"][0]["attempts"][0]["returncode"] != 0
    assert (cfg.method_root / "tables/table_generalization_train_deltaei.csv").exists()


@pytest.fixture
def fast_spot40_reports(monkeypatch):
    monkeypatch.setattr(tables, "_render_spot40", lambda *args, **kwargs: None)


def spot_config(tmp_path, **kwargs):
    return config(tmp_path, hierarchies=("spot_to_k40",), **kwargs)


def test_streamed_induced_cq_matches_existing_budget(tmp_path):
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
    rng = np.random.default_rng(6)
    p, source, target = rng.random((75, 81)), rng.random((75, 4)), rng.random((81, 4))
    for arr in (p, source, target):
        arr /= arr.sum(axis=1, keepdims=True)
    for name, arr in (("PIJ_micro_train", p), ("S_t", source), ("S_tp", target)):
        np.save(tmp_path / f"{name}.npy", arr)
    expected = information_closure_budget(p, source, target)
    actual = tables.induced_cq_from_saved(tmp_path)
    assert actual["I_available_bits"] == pytest.approx(expected["I_available_bits"], abs=1e-10)
    assert actual["I_retained_bits"] == pytest.approx(expected["I_macro_retained_bits"], abs=1e-10)
    assert actual["CQ_induced"] == pytest.approx(expected["closure_quality"], abs=1e-8)


def test_spot40_reuses_original_grid_and_other_folder_without_touching_legacy_manifest(tmp_path, monkeypatch, fast_spot40_reports):
    cfg = spot_config(tmp_path)
    jobs = runner.planned_jobs(cfg)
    old = cfg.method_root / "manifest.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"legacy_grid":18}', encoding="utf-8")
    lung = cfg.method_root / "runs" / jobs[0]["job_id"] / "attempt_0001"
    saved_run(cfg, jobs[0], lung, informative=True, full_config=True)
    external = tmp_path / "other_output"
    brain = external / "two_stage" / "old_brain_run"
    saved_run(cfg, jobs[1], brain, informative=True, full_config=True)
    monkeypatch.setattr(runner, "execute_job", lambda *a: pytest.fail("Both runs should be reused"))
    assert runner.run_spot40(cfg, reuse_roots=(external,))["status"] == "complete"
    state = json.loads((cfg.method_root / "spot40_manifest.json").read_text(encoding="utf-8"))
    assert [j["action"] for j in state["jobs"]] == ["reuse", "reuse"]
    assert old.read_text(encoding="utf-8") == '{"legacy_grid":18}'
    assert [Path(j["run_dir"]) for j in state["jobs"]] == [lung, brain]
    assert runner.run_spot40(cfg)["status"] == "complete"  # remembers the external source root


def test_spot40_rejects_wrong_settings_and_trains_only_missing_cell(tmp_path, monkeypatch, fast_spot40_reports):
    cfg = spot_config(tmp_path)
    materialize_inputs(cfg)
    jobs = runner.planned_jobs(cfg)
    old = cfg.method_root / "runs" / jobs[0]["job_id"] / "attempt_0001"
    saved_run(cfg, jobs[0], old, informative=True, full_config=True)
    bad = cfg.method_root / "runs" / jobs[1]["job_id"] / "attempt_0001"
    saved_run(cfg, jobs[1], bad, informative=True, full_config=True)
    trainer = json.loads((bad / "config.json").read_text())
    trainer["lr"] = .25
    (bad / "config.json").write_text(json.dumps(trainer))
    calls = []

    def execute(command, log):
        path = Path(command[command.index("--out-dir") + 1])
        calls.append(path)
        saved_run(cfg, jobs[1], path, informative=True, full_config=True)
        return 0

    monkeypatch.setattr(runner, "execute_job", execute)
    assert runner.run_spot40(cfg)["status"] == "complete"
    assert len(calls) == 1 and calls[0].name == "attempt_0002"
    assert (bad / "config.json").is_file()
    state = json.loads((cfg.method_root / "spot40_manifest.json").read_text(encoding="utf-8"))
    assert state["jobs"][0]["action"] == "reuse"
    assert state["jobs"][1]["action"] == "train"
    assert "lr" in state["jobs"][1]["rejected_sources"][0]["reason"]


def test_spot40_zero_results_are_partial_and_explain_every_dash(tmp_path, fast_spot40_reports, capsys):
    cfg = spot_config(tmp_path)
    assert runner.run_spot40(cfg)["status"] == "partial"
    stdout = capsys.readouterr().out
    assert "NO VERIFIED VALUES" in stdout and "MISSING" in stdout
    assert "DeltaEI 0/2; CQ 0/2" in stdout
    assert (cfg.method_root / "spot40_failures.csv").exists()


def test_spot40_dry_run_does_not_create_outputs(tmp_path):
    cfg = spot_config(tmp_path)
    assert runner.run_spot40(cfg, dry_run=True)["status"] == "partial"
    assert not cfg.output_root.exists()


def test_spot40_retrain_all_preserves_previous_arrays(tmp_path, monkeypatch, fast_spot40_reports):
    cfg = spot_config(tmp_path, organs=("lung",))
    materialize_inputs(cfg)
    job = runner.planned_jobs(cfg)[0]
    path = cfg.method_root / "runs" / job["job_id"] / "attempt_0001"
    saved_run(cfg, job, path, informative=True, full_config=True)
    original = (path / "PIJ_macro_train.npy").read_bytes()

    def execute(command, log):
        new = Path(command[command.index("--out-dir") + 1])
        assert new.name == "attempt_0002"
        saved_run(cfg, job, new, informative=True, full_config=True)
        return 0

    monkeypatch.setattr(runner, "execute_job", execute)
    assert runner.run_spot40(cfg, retrain_all=True)["status"] == "complete"
    assert (path / "PIJ_macro_train.npy").read_bytes() == original


def test_spot40_real_tables_and_render_use_verified_metrics(tmp_path, monkeypatch):
    cfg = spot_config(tmp_path, organs=("lung",))
    job = runner.planned_jobs(cfg)[0]
    path = cfg.method_root / "runs" / job["job_id"] / "attempt_0001"
    saved_run(cfg, job, path, informative=True, full_config=True)
    monkeypatch.setattr(runner, "execute_job", lambda *a: pytest.fail("No training should occur"))
    assert runner.run_spot40(cfg)["status"] == "complete"
    assert runner.run_spot40(cfg, mode="render")["status"] == "complete"
    for name in ("table_spot40_train_deltaei", "table_spot40_induced_cq", "figure_spot40_deltaei_cq"):
        for suffix in ("png", "pdf"):
            assert (cfg.method_root / "tables" / f"{name}.{suffix}").stat().st_size > 0
    with (cfg.method_root / "tables/table_spot40_induced_cq.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["11.5->12.5"] != "—"


def test_spot40_cli_selection():
    args = build_parser().parse_args(["--task", "spot40", "--method", "two_stage", "--output-root", "unused",
                                      "--reuse-root", "original", "--reuse-root", "another"])
    assert args.reuse_root == [Path("original"), Path("another")]


def test_spot40_undefined_cq_keeps_valid_delta_and_reports_reason(tmp_path, fast_spot40_reports, capsys):
    cfg = spot_config(tmp_path, organs=("lung",))
    job = runner.planned_jobs(cfg)[0]
    path = cfg.method_root / "runs" / job["job_id"] / "attempt_0001"
    saved_run(cfg, job, path, full_config=True)  # uniform micro PIJ => zero available information
    assert runner.run_spot40(cfg)["status"] == "partial"
    text = capsys.readouterr().out
    assert "DeltaEI 1/1; CQ 0/1" in text and "CQ UNDEFINED" in text
    assert "undefined_zero_available_information" in (cfg.method_root / "spot40_failures.csv").read_text(encoding="utf-8-sig")


def test_spot40_new_cohort_keeps_prior_state_and_reuses_results(tmp_path, monkeypatch, fast_spot40_reports):
    cfg = spot_config(tmp_path)
    for job in runner.planned_jobs(cfg):
        saved_run(cfg, job, cfg.method_root / "runs" / job["job_id"] / "attempt_0001",
                  informative=True, full_config=True)
    monkeypatch.setattr(runner, "execute_job", lambda *a: pytest.fail("Existing runs must be reused"))
    assert runner.run_spot40(cfg)["status"] == "complete"
    assert runner.run_spot40(replace(cfg, organs=("lung",)))["status"] == "complete"
    assert len(list((cfg.method_root / "audit/spot40_manifests").glob("*.json"))) == 1


def test_spot40_render_detects_changed_scientific_config(tmp_path, fast_spot40_reports, capsys):
    cfg = spot_config(tmp_path, organs=("lung",))
    job = runner.planned_jobs(cfg)[0]
    path = cfg.method_root / "runs" / job["job_id"] / "attempt_0001"
    saved_run(cfg, job, path, informative=True, full_config=True)
    assert runner.run_spot40(cfg)["status"] == "complete"
    actual = json.loads((path / "config.json").read_text())
    actual["lr"] = .2
    (path / "config.json").write_text(json.dumps(actual))
    assert runner.run_spot40(cfg, mode="render")["status"] == "partial"
    assert "Saved trainer lr" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["shr_method_8", "shr_method_10"])
def test_retired_generalization_methods_are_rejected(tmp_path, method):
    with pytest.raises(ValueError, match="Unsupported method"):
        config(tmp_path, method=method).validate()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--method", method, "--output-root", str(tmp_path)])


def test_resume_accepts_historical_checkpoint_description(tmp_path, fast_reports):
    cfg = config(tmp_path)
    assert runner.run(cfg)["status"] == "partial"
    manifest = cfg.method_root / "manifest.json"
    state = json.loads(manifest.read_text(encoding="utf-8"))
    state["contract"]["checkpoint_policy"] = "historical explanatory text"
    manifest.write_text(json.dumps(state), encoding="utf-8")
    assert runner.run(cfg, mode="resume")["status"] == "partial"
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.run(replace(cfg, seed=43), mode="resume")
