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


def saved_run(cfg, job, run_dir):
    """A mathematically consistent persisted-result fixture, not a trained model."""
    run_dir.mkdir(parents=True)
    k = job["K"]
    micro = np.full((k + 3, k + 5), 1 / (k + 5), dtype=np.float32)
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
    provenance = {"h5ad_t": job["inputs"]["h5ad_t"], "h5ad_tp": job["inputs"]["h5ad_tp"],
                  "grn_feature_method": "legacy_nmf", "cci_feature_method": "laplacian_hks",
                  "alpha_cci": .20, "nmf_components": 5, "nmf_max_iter_used": cfg.nmf_max_iter}
    for name, obj in (("summary", summary), ("config", trainer),
                      ("feature_manifest", {"method": cfg.canonical_method, "provenance": provenance})):
        (run_dir / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    checkpoint = "best_joint.pt" if cfg.method_tag == "two_stage" else "best_model.pt"
    (run_dir / checkpoint).touch()
    if cfg.method_tag == "shr_method_10":
        row = {"epoch": 2, "delta_EI": summary["delta_EI_best_checkpoint"], "checkpoint_eligible": "True",
               "Keff_t": summary["Keff_t"], "Keff_tp": summary["Keff_tp"]}
        with (run_dir / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)


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
    method10 = replace(cfg, method="shr_method_10")
    command = runner.training_command(method10, jobs[0], tmp_path / "attempt")
    assert command[command.index("--method") + 1] == "shr_method_10"
    assert command[command.index("--checkpoint-keff-min") + 1] == "1e-6"
    assert command[command.index("--keff-min") + 1] == "45"
    assert cfg.method_root != method10.method_root


@pytest.mark.parametrize("extra", [("--method=shr_method_10",), ("--k", "2"), ("--alpha-cci", ".8"), ("--h5ad-t", "x")])
def test_extra_args_cannot_override_identity(tmp_path, extra):
    with pytest.raises(ValueError, match="cannot override"):
        config(tmp_path, runner_extra_args=extra).validate()


def test_abbreviated_override_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="abbreviated"):
        config(tmp_path, runner_extra_args=("--meth", "shr_method_10")).validate()


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


@pytest.mark.parametrize("method", ["two_stage", "shr_method_10"])
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
