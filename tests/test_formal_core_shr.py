"""Synthetic matrices test reporting contracts; no production experiment is run."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mignet_ce.downstream.formal_core_analysis import run_pipeline
from mignet_ce.downstream.formal_core_analysis.methods import METHODS, METHOD_LABELS, expected_conditions
from mignet_ce.downstream.formal_core_analysis.tables import (
    PAIRS, SCALES, collect_optimal_runs, build_optimal_grid, build_optimal_triplet_grid, inspect_optimal_run,
)
from mignet_ce.downstream.formal_core_analysis.plots import save_triplet_table
from mignet_ce.downstream.formal_core_analysis.preview import generate_preview
from mignet_ce.downstream.analysis.dynamic_closure.analysis import effective_information


def _measurements():
    return pd.DataFrame([
        {"method": method, "input_scale": scale, "time_pair": pair,
         "trained_delta_EI": .1 + index / 1000, "Keff_t": 2., "Keff_tp": 2.,
         "CQ_display": .4, "closure_JS_direct_bits": .05}
        for index, (method, scale, pair) in enumerate(product(METHODS, SCALES, PAIRS))
    ])


def test_all_four_methods_appear_in_all_three_grids():
    assert [METHOD_LABELS[method] for method in METHODS] == [
        "Single-stage", "Two-stage", "SHR Method 8", "SHR Method 10"]
    frame = _measurements()
    assert len(frame) == expected_conditions(SCALES, PAIRS) == 36
    for grid in (build_optimal_triplet_grid(frame), build_optimal_grid(frame, "CQ_display"),
                 build_optimal_grid(frame, "closure_JS_direct_bits")):
        assert len(grid) == 12
        assert list(zip(grid["Method"], grid["Input scale"])) == list(product(METHODS, SCALES))


def test_missing_or_duplicate_shr_conditions_are_not_silently_accepted():
    frame = _measurements()
    for builder in (build_optimal_triplet_grid, lambda values: build_optimal_grid(values, "CQ_display")):
        with pytest.raises(ValueError, match="Missing"):
            builder(frame.iloc[:-1])
        with pytest.raises(ValueError, match="Duplicate|Multiple"):
            builder(pd.concat([frame, frame.iloc[[-1]]], ignore_index=True))


def test_pipeline_generates_four_method_jobs_without_running_them(tmp_path):
    args = run_pipeline.build_argparser().parse_args([
        "--data-root", str(tmp_path / "data"), "--grn-root", str(tmp_path / "grn"),
        "--cci-root", str(tmp_path / "cci"), "--developmental-root", str(tmp_path / "dev"),
        "--sensitivity-root", str(tmp_path / "sensitive"), "--output-root", str(tmp_path / "out"),
        "--resume",
    ])
    jobs = run_pipeline.make_commands(args, None)
    train_jobs = [job for job in jobs if job["stage"].startswith("train_")]
    assert len(train_jobs) == 4 and sum(job["expected_runs"] for job in train_jobs) == 36
    assert [job["command"][job["command"].index("--method") + 1] for job in train_jobs] == list(METHODS)
    assert all("--resume" in job["command"] for job in train_jobs)
    assert not (tmp_path / "out").exists()


def test_preview_has_no_scientific_values_and_preserves_existing_outputs(tmp_path):
    out = tmp_path / "preview"
    manifest = generate_preview(out)
    assert manifest["rows_per_table"] == 12
    assert manifest["expected_formal_conditions"] == 36
    assert manifest["models_trained"] is False
    assert manifest["contains_experimental_measurements"] is False
    assert not list(out.glob("*.csv")) and not list(out.glob("*.npy"))
    for name in manifest["files"]:
        assert (out / name).stat().st_size > 1000
    before = (out / "preview_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_preview(out)
    assert (out / "preview_manifest.json").read_bytes() == before


def test_formal_triplet_renderer_rejects_missing_values_or_wrong_order(tmp_path):
    frame = build_optimal_triplet_grid(_measurements())
    frame.iloc[0, 2] = np.nan
    with pytest.raises(ValueError, match="missing/invalid"):
        save_triplet_table(frame, tmp_path / "invalid", title="test", subtitle="test")
    with pytest.raises(ValueError, match="canonical"):
        save_triplet_table(frame.iloc[::-1], tmp_path / "invalid", title="test", subtitle="test")
    assert not list(tmp_path.glob("*.png"))


def _write_synthetic_checkpoints(root):
    micro = np.array([[.5, .1, .3, .1], [.1, .4, .2, .3], [.2, .2, .5, .1]])
    source = np.array([[.8, .2], [.5, .5], [.1, .9]])
    target = np.array([[.9, .1], [.6, .4], [.3, .7], [.2, .8]])
    macro = np.array([[.8, .2], [.3, .7]])

    def keff(assignment):
        usage = assignment.mean(0)
        return float(np.exp(-(usage * np.log(usage)).sum()))

    for method, scale, pair in product(METHODS, SCALES, PAIRS):
        run = root / method / scale / pair.replace("->", "_to_")
        run.mkdir(parents=True)
        for name, value in (("PIJ_micro_train", micro), ("S_t", source), ("S_tp", target), ("PIJ_macro_train", macro)):
            np.save(run / f"{name}.npy", value)
        summary = {"method": method, "input_scale": scale, "time_pair": pair,
                   "EI_macro_best_checkpoint": effective_information(macro), "EI_micro_fixed": effective_information(micro),
                   "delta_EI_best_checkpoint": effective_information(macro) - effective_information(micro),
                   "Keff_t": keff(source), "Keff_tp": keff(target), "K": 2, "synthetic_test_fixture": True}
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "feature_manifest.json").write_text(json.dumps({"provenance": {
            "cci_feature_method": "laplacian_hks", "grn_feature_method": "legacy_nmf", "alpha_cci": .20,
            "synthetic_test_fixture": True,
        }}), encoding="utf-8")


def test_checkpoint_audit_accepts_both_shr_methods_and_rejects_mismatched_metrics(tmp_path):
    _write_synthetic_checkpoints(tmp_path / "runs")
    runs, rejected = collect_optimal_runs(tmp_path / "runs")
    assert len(runs) == 36 and rejected == []
    for method in ("shr_method_8", "shr_method_10"):
        run = Path(runs[runs.method.eq(method)].iloc[0].run_dir)
        np.save(run / "PIJ_macro_train.npy", np.full((2, 2), .5))
        with pytest.raises(ValueError, match="summary mismatch"):
            inspect_optimal_run(run)


def test_formal_collector_reports_complete_only_with_all_36_conditions(tmp_path, monkeypatch):
    from mignet_ce.downstream.formal_core_analysis.__main__ import main
    import sys
    _write_synthetic_checkpoints(tmp_path / "runs")
    sensitivity = tmp_path / "sensitive"
    sensitivity.mkdir()
    (sensitivity / "manifest.json").write_text(json.dumps({
        "status": "complete", "cci_feature": "laplacian_hks", "cci_order_provenance": "synthetic_test_fixture",
    }), encoding="utf-8")
    selector = {"grn_feature": "legacy_nmf", "cci_feature": "laplacian_hks", "transition": "KL+OT", "alpha": .20}
    pd.DataFrame([{**selector, "time_pair": pair, "layer": layer, "EI_bits": .1 + index * .1}
                  for pair in PAIRS for index, layer in enumerate((*SCALES, "seurat_k10"))]).to_csv(
                      sensitivity / "alpha_layer_ei.csv", index=False)
    pd.DataFrame([{**selector, "time_pair": pair, "hierarchy": hierarchy, "CQ": .5,
                   "signal_status": "informative", "I_available_bits": .2, "I_retained_bits": .1,
                   "closure_JS_direct_bits": .05, "mapping_method": "spot_overlap_soft_contingency"}
                  for pair in PAIRS for hierarchy in (
                      "spot->seurat_k150", "spot->seurat_k40", "spot->seurat_k10", "seurat_k150->seurat_k40")]).to_csv(
                          sensitivity / "alpha_natural_cq.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["formal_core_analysis", "--sensitivity-root", str(sensitivity),
                        "--data-root", str(tmp_path), "--optimal-runs-root", str(tmp_path / "runs"),
                        "--output-root", str(tmp_path / "out"), "--skip-figures"])
    main()
    manifest = json.loads((tmp_path / "out" / "output_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["optimal_trained_runs_verified"] == manifest["expected_optimal_runs"] == 36
    assert manifest["methods"] == list(METHODS)
    for name in ("table_optimal_trained_deltaei_keff", "table_optimal_induced_cq", "table_optimal_closure_js_direct_bits"):
        assert len(pd.read_csv(tmp_path / "out" / "tables" / (name + ".csv"))) == 12
