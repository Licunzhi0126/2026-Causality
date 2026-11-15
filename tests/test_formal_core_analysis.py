"""Synthetic matrices test reporting contracts; no production experiment is run."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mignet_ce.downstream.formal_core_analysis import run_pipeline
from mignet_ce.downstream.formal_core_analysis.methods import METHODS, METHOD_LABELS, ORGANS, expected_conditions
from mignet_ce.downstream.formal_core_analysis.tables import (
    PAIRS, SCALES, OPTIMAL_TABLES, build_optimal_report_grid, collect_optimal_runs, build_optimal_grid, build_optimal_triplet_grid, inspect_optimal_run,
)
from mignet_ce.downstream.formal_core_analysis.plots import save_triplet_table
from mignet_ce.downstream.formal_core_analysis.preview import generate_preview
from mignet_ce.downstream.analysis.dynamic_closure.analysis import effective_information


def _measurements():
    return pd.DataFrame([
        {"method": method, "organ": organ, "input_scale": scale, "time_pair": pair,
         "loss_protocol":"formal_two_stage_main", "CQ_raw":.4,
         "HardK_minus_KeffHard_t":.2, "HardK_minus_KeffHard_tp":.3,
         "trained_delta_EI": .1 + index / 1000, "Keff_t": 2., "Keff_tp": 2.,
         "CQ_display": .4, "closure_JS_direct_bits": .05}
        for index, (organ, method, scale, pair) in enumerate(product(ORGANS, METHODS, SCALES, PAIRS))
    ])


def test_only_formal_method_and_all_organs_appear_in_both_reports():
    assert METHODS == ("formal_two_stage",)
    assert METHOD_LABELS["formal_two_stage"] == "Formal two-stage"
    frame = _measurements()
    assert len(frame) == expected_conditions(SCALES, PAIRS) == 27
    for metrics in OPTIMAL_TABLES.values():
        grid = build_optimal_report_grid(frame, metrics)
        assert len(grid) == 9
        assert list(zip(grid.Organ, grid["Input scale"])) == list(product(ORGANS, SCALES))
        assert list(grid.columns[2:]) == [f"{pair}::{metric}" for pair in PAIRS for metric in metrics]


def test_missing_or_duplicate_conditions_are_not_silently_accepted():
    frame = _measurements()
    for metrics in OPTIMAL_TABLES.values():
        builder = lambda values: build_optimal_report_grid(values, metrics)
        with pytest.raises(ValueError, match="Missing"):
            builder(frame.iloc[:-1])
        with pytest.raises(ValueError, match="Duplicate|Multiple"):
            builder(pd.concat([frame, frame.iloc[[-1]]], ignore_index=True))


def test_pipeline_generates_three_isolated_organ_jobs_without_running_them(tmp_path):
    args = run_pipeline.build_argparser().parse_args([
        "--data-root", str(tmp_path / "data"), "--grn-root", str(tmp_path / "grn"),
        "--cci-root", str(tmp_path / "cci"), "--developmental-root", str(tmp_path / "dev"),
        "--sensitivity-root", str(tmp_path / "sensitive"), "--output-root", str(tmp_path / "out"),
        "--resume",
    ])
    jobs = run_pipeline.make_commands(args, None)
    train_jobs = [job for job in jobs if job["stage"].startswith("train_")]
    assert len(train_jobs) == 3 and sum(job["expected_runs"] for job in train_jobs) == 27
    assert [job["command"][job["command"].index("--method") + 1] for job in train_jobs] == ["formal_two_stage"] * 3
    assert [job["organ"] for job in train_jobs] == list(ORGANS)
    assert len({job["command"][job["command"].index("--out-root") + 1] for job in train_jobs}) == 3
    assert all("--resume" in job["command"] for job in train_jobs)
    assert not (tmp_path / "out").exists()


def test_preview_has_no_scientific_values_and_preserves_existing_outputs(tmp_path):
    out = tmp_path / "preview"
    manifest = generate_preview(out)
    assert manifest["rows_per_table"] == 9
    assert manifest["expected_formal_conditions"] == 27
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
    frame = build_optimal_triplet_grid(_measurements().query("organ == 'heart'"))
    frame.iloc[0, 2] = np.nan
    with pytest.raises(ValueError, match="missing/invalid"):
        save_triplet_table(frame, tmp_path / "invalid", title="test", subtitle="test")
    with pytest.raises(ValueError, match="canonical"):
        save_triplet_table(frame.iloc[::-1], tmp_path / "invalid", title="test", subtitle="test")
    assert not list(tmp_path.glob("*.png"))


def _write_synthetic_checkpoints(root, methods=METHODS, organs=ORGANS):
    micro = np.array([[.5, .1, .3, .1], [.1, .4, .2, .3], [.2, .2, .5, .1]])
    source = np.array([[.8, .2], [.5, .5], [.1, .9]])
    target = np.array([[.9, .1], [.6, .4], [.3, .7], [.2, .8]])
    macro = np.array([[.8, .2], [.3, .7]])

    def keff(assignment):
        usage = assignment.mean(0)
        return float(np.exp(-(usage * np.log(usage)).sum()))

    for organ, method, scale, pair in product(organs, methods, SCALES, PAIRS):
        run = root / organ / method / scale / pair.replace("->", "_to_")
        run.mkdir(parents=True)
        for name, value in (("PIJ_micro_train", micro), ("S_t", source), ("S_tp", target), ("PIJ_macro_train", macro)):
            np.save(run / f"{name}.npy", value)
        summary = {"method": method, "organ":organ, "input_scale": scale, "time_pair": pair,
                   "loss_protocol":"formal_two_stage_main", "selected_checkpoint":"best_joint.pt",
                   "checkpoint_metric_state_consistent":True,
                   "EI_macro_best_checkpoint": effective_information(macro), "EI_micro_fixed": effective_information(micro),
                   "delta_EI_best_checkpoint": effective_information(macro) - effective_information(micro),
                   "Keff_t": keff(source), "Keff_tp": keff(target), "K": 2, "synthetic_test_fixture": True}
        for label, assignment in (("t", source), ("tp", target)):
            _, counts = np.unique(assignment.argmax(1), return_counts=True)
            proportions = counts / counts.sum()
            summary[f"hardK_{label}"] = len(counts)
            summary[f"KeffHard_{label}"] = float(np.exp(-(proportions * np.log(proportions)).sum()))
        (run / "config.json").write_text(json.dumps({"objective_version":"formal_two_stage_main"}), encoding="utf-8")
        (run / "best_joint.pt").write_bytes(b"synthetic-test-checkpoint")
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "feature_manifest.json").write_text(json.dumps({"provenance": {
            "cci_feature_method": "laplacian_hks", "grn_feature_method": "legacy_nmf", "alpha_cci": .20,
            "synthetic_test_fixture": True,
        }}), encoding="utf-8")


def test_checkpoint_audit_accepts_all_methods_and_rejects_mismatched_metrics(tmp_path):
    _write_synthetic_checkpoints(tmp_path / "runs")
    runs, rejected = collect_optimal_runs(tmp_path / "runs")
    assert len(runs) == 27 and rejected == []
    for method in METHODS:
        run = Path(runs[runs.method.eq(method)].iloc[0].run_dir)
        np.save(run / "PIJ_macro_train.npy", np.full((2, 2), .5))
        with pytest.raises(ValueError, match="summary mismatch"):
            inspect_optimal_run(run)


def test_formal_audits_low_signal_cq_fallback_and_rejects_false_labels(tmp_path):
    from mignet_ce.information_thresholds import closure_signal_threshold_bits
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
    _write_synthetic_checkpoints(tmp_path / "runs", methods=(*METHODS, "single_stage_cq_only"))
    run = tmp_path / "runs" / "heart" / "single_stage_cq_only" / "spot" / PAIRS[0].replace("->", "_to_")
    micro = np.load(run / "PIJ_micro_train.npy")
    micro = .05 * micro + .95 / micro.shape[1]
    np.save(run / "PIJ_micro_train.npy", micro)
    budget = information_closure_budget(micro, np.load(run / "S_t.npy"), np.load(run / "S_tp.npy"))
    available, retained = float(budget["I_available_bits"]), float(budget["I_macro_retained_bits"])
    threshold = closure_signal_threshold_bits(2)
    assert 1e-8 < available < threshold
    summary = json.loads((run / "summary.json").read_text())
    summary.update(
        EI_micro_fixed=effective_information(micro.copy()),
        delta_EI_best_checkpoint=summary["EI_macro_best_checkpoint"] - effective_information(micro.copy()),
        I_available=available, I_retained=retained, signal_threshold_bits=threshold,
        checkpoint_policy="strict_then_low_signal_cq_v1", selected_checkpoint="best_cq.pt",
        checkpoint_source="best_low_signal_cq.pt", strict_checkpoint_found=False,
        checkpoint_fallback_used=True, fallback_type="low_signal_cq",
        fallback_reason="no_strict_cq_checkpoint", checkpoint_signal_status="low-signal",
        CQ_training_score=retained / threshold,
    )
    for name in ("best_cq.pt", "best_low_signal_cq.pt"):
        (run / name).write_bytes(b"synthetic-checkpoint-placeholder")
    (run / "summary.json").write_text(json.dumps(summary))
    row = inspect_optimal_run(run)
    assert row["checkpoint_fallback_used"] is True and row["strict_checkpoint_found"] is False
    assert row["checkpoint_source"] == "best_low_signal_cq.pt"
    assert row["CQ_raw"] == pytest.approx(retained / available)
    runs, rejected = collect_optimal_runs(tmp_path / "runs")
    assert len(runs) == 27 and not rejected
    assert str(run.resolve()) not in set(runs.run_dir)  # historical methods stay out of formal reports
    for changed in ({"strict_checkpoint_found": True}, {"fallback_reason": None},
                    {"checkpoint_source": "../other.pt"}, {"checkpoint_signal_status": "informative"}):
        (run / "summary.json").write_text(json.dumps({**summary, **changed}))
        with pytest.raises(ValueError, match="CQ checkpoint"):
            inspect_optimal_run(run)


def test_formal_collector_reports_complete_only_with_all_27_conditions(tmp_path, monkeypatch):
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
    sentinel = tmp_path / "out" / "tables" / "cq_low_signal_audit_previous.csv"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("preserve historical output")
    main()
    assert sentinel.read_text() == "preserve historical output"
    manifest = json.loads((tmp_path / "out" / "output_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["optimal_trained_runs_verified"] == manifest["expected_optimal_runs"] == 27
    assert manifest["methods"] == list(METHODS)
    for name in OPTIMAL_TABLES:
        assert len(pd.read_csv(tmp_path / "out" / "tables" / (name + ".csv"))) == 9


@pytest.mark.parametrize("method", ["shr_method_8", "shr_method_10"])
def test_retired_results_are_preserved_and_excluded_from_active_collection(tmp_path, method):
    _write_synthetic_checkpoints(tmp_path / "runs")
    retired = tmp_path / "runs" / method / "historical"
    retired.mkdir(parents=True)
    summary = retired / "summary.json"
    payload = json.dumps({"method": method})
    summary.write_text(payload, encoding="utf-8")
    runs, rejected = collect_optimal_runs(tmp_path / "runs")
    assert len(runs) == 27 and rejected == []
    assert set(runs.method) == set(METHODS)
    assert summary.read_text(encoding="utf-8") == payload


def test_optimal_only_collector_and_preflight_do_not_require_sensitivity(tmp_path):
    from mignet_ce.downstream.formal_core_analysis.__main__ import main
    from mignet_ce.downstream.formal_core_analysis.preflight import inspect_inputs
    _write_synthetic_checkpoints(tmp_path / "runs")
    report = inspect_inputs(None, tmp_path, tmp_path / "runs", .2, "legacy_nmf",
                            PAIRS[0], True, stage="optimal")
    assert report["ready_for_publication_candidate"]
    assert report["trained_optimal_runs"] == 27
    assert not (tmp_path / "out").exists()
    main(["--stage", "optimal", "--data-root", str(tmp_path), "--optimal-runs-root", str(tmp_path / "runs"),
          "--output-root", str(tmp_path / "out"), "--skip-figures"])
    assert not list((tmp_path / "out" / "tables").glob("*hierarchy*"))
    inventory = pd.read_csv(tmp_path / "out" / "audit" / "trained_optimal_run_inventory.csv")
    assert set(inventory.organ) == set(ORGANS)
    assert np.allclose(inventory.HardK_minus_KeffHard_t, inventory.hardK_t - inventory.KeffHard_t)
    assert not np.allclose(inventory.HardK_minus_KeffHard_t, inventory.hardK_t - inventory.Keff_t)


@pytest.mark.parametrize("field,value", [("KeffHard_t", 1.), ("hardK_tp", 1), ("loss_protocol", "formal_two_stage_v1")])
def test_formal_run_rejects_wrong_hard_metrics_or_retired_protocol(tmp_path, field, value):
    _write_synthetic_checkpoints(tmp_path / "runs", organs=("heart",))
    run = next((tmp_path / "runs").rglob("summary.json")).parent
    summary = json.loads((run / "summary.json").read_text())
    summary[field] = value
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="occupancy|protocol"):
        inspect_optimal_run(run)


def test_report_rejects_wrong_protocol_nonfinite_and_duplicate_organs():
    frame = _measurements()
    for changes in ({"loss_protocol":"formal_two_stage_v1"}, {"CQ_raw":np.nan}):
        changed = frame.copy()
        for key, value in changes.items():
            changed.loc[0, key] = value
        with pytest.raises(ValueError):
            build_optimal_report_grid(changed, OPTIMAL_TABLES["table_optimal_deltaei_cq"])
    with pytest.raises(ValueError, match="unique"):
        build_optimal_report_grid(frame, ("CQ_raw",), organs=("heart", "heart"))


def test_all_spots_connect_to_their_actual_spatial_centroid_after_shuffled_join():
    from mignet_ce.downstream.formal_core_analysis.paper_figure_layouts import centroid_connections, _norm_xy
    n = 420
    spot = pd.DataFrame({"spot_id":[f"s{i}" for i in range(n)], "x":np.arange(n), "y":np.arange(n) % 13})
    membership = pd.DataFrame({"spot_id":spot.spot_id, "domain_id":np.arange(n) % 7}).sample(frac=1, random_state=3)
    members, centers, segments = centroid_connections(spot, membership, "domain_id", 1, 2)
    assert len(segments) == n  # No old 240/300-spot sampling cap.
    expected = _norm_xy(spot).merge(membership, on="spot_id")
    reference = expected.groupby("domain_id")[["xn", "yn"]].mean()
    assert np.allclose(centers, reference)
    assert np.allclose(segments[:, 0, 0], expected.xn)
    assert np.allclose(segments[:, 1, 0], reference.loc[expected.domain_id, "xn"])
    assert np.allclose(segments[:, 1, 1], reference.loc[expected.domain_id, "yn"] * .34 + 2)
    # A different non-nested target partition is joined independently by spot_id.
    other = membership.assign(domain_id=lambda d: d.spot_id.str[1:].astype(int) // 31)
    _, alternate, other_segments = centroid_connections(spot, other, "domain_id", 2, 3)
    assert len(other_segments) == n and len(alternate) != len(centers)
    for invalid in (membership.iloc[:-1], pd.concat([membership, membership.iloc[:1]])):
        with pytest.raises(ValueError, match="membership|identities"):
            centroid_connections(spot, invalid, "domain_id", 0, 1)


def test_rounded_brace_uses_a_continuous_curve():
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from mignet_ce.downstream.formal_core_analysis.paper_figure_layouts import _inward_brace
    fig, ax = plt.subplots()
    _inward_brace(ax, 0., 3., x=1., text="test")
    path = ax.patches[0].get_path()
    assert path.codes[0] == PlotPath.MOVETO
    assert set(path.codes[1:]) == {PlotPath.CURVE4}
    assert path.vertices[0, 1] == 0. and path.vertices[-1, 1] == 3.
    plt.close(fig)


def test_complete_figure_rendering_with_centroid_layers(tmp_path):
    """Synthetic spatial fixture exercises the full renderers, not scientific results."""
    from mignet_ce.downstream.formal_core_analysis.paper_figure_layouts import render_figure1_k10, render_figure2
    from mignet_ce.downstream.formal_core_analysis.tables import LEVELS
    spot = pd.DataFrame({"spot_id":list("abcdef"), "x":[0, 1, 2, 3, 4, 5], "y":[1, 2, 1, 0, 1, 2]})
    domains = spot.assign(domain_id=[0, 0, 1, 1, 2, 2])
    render_figure1_k10(source_stage="12.5", target_stage="13.5", source_spot=spot, target_spot=spot,
                       source_k150=domains, target_k150=domains, source_k40=domains, target_k40=domains,
                       source_k10=domains, target_k10=domains,
                       gains={name:np.nan for name, _, _ in LEVELS}, output_path=tmp_path / "figure1.png",
                       dpi=60, preview_notice="SYNTHETIC LAYOUT TEST")
    assignments = domains.rename(columns={"domain_id":"hard_cluster"})
    render_figure2(source_stage="12.5", target_stage="13.5", source_spot=spot, target_spot=spot,
                   source_assignments=assignments, target_assignments=assignments, delta_ei=np.nan,
                   method_label="synthetic layout test", output_path=tmp_path / "figure2.png", dpi=60,
                   preview_notice="SYNTHETIC LAYOUT TEST")
    for name in ("figure1", "figure2"):
        for extension in (".png", ".pdf"):
            assert (tmp_path / (name + extension)).stat().st_size > 1000


def test_pipeline_optimal_dry_run_checks_all_organ_inputs_without_natural_data(tmp_path, capsys):
    args = run_pipeline.build_argparser().parse_args([
        "--stage", "optimal", "--data-root", str(tmp_path / "data"),
        "--grn-root", str(tmp_path / "grn"), "--cci-root", str(tmp_path / "cci"),
        "--developmental-root", str(tmp_path / "dev"), "--output-root", str(tmp_path / "out"),
        "--dry-run"])
    for organ, scale, time in product(ORGANS, SCALES, run_pipeline.TIMES):
        sample = f"{run_pipeline.STEMS[scale]}_{organ}_{time}"
        paths = [args.data_root / scale / organ / f"{sample}.h5ad",
                 args.grn_root / scale / sample / "grn_edges.csv",
                 args.cci_root / scale / f"{sample}_CCI_total.npz",
                 args.cci_root / scale / f"{sample}_index.tsv",
                 args.developmental_root / scale / f"{organ}_{time}_features.csv"]
        if scale != "spot":
            paths.append(args.data_root / scale / organ / f"{sample}_spot_domain_map.csv")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic input-existence fixture")
    report = run_pipeline.inspect_inputs(args)
    assert report["missing_input_count"] == 0
    assert len(run_pipeline.maturity_files(args)) == 27
    assert not args.output_root.exists()
    args.cci_root.joinpath("spot", "spot_brain_11.5_index.tsv").rename(args.cci_root / "missing_index_fixture.tsv")
    report = run_pipeline.inspect_inputs(args)
    assert report["missing_input_count"] == 1
    assert "spot_brain_11.5_index.tsv" in report["missing_inputs"][0]


def test_multiscale_runner_accepts_formal_method_and_uses_native_formal_config(tmp_path):
    from scripts import run_multiscale_maturity_cci_grn as orchestrator
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    args = run_pipeline.build_argparser().parse_args([
        "--stage", "optimal", "--organs", "lung", "--data-root", str(tmp_path),
        "--grn-root", str(tmp_path), "--cci-root", str(tmp_path),
        "--developmental-root", str(tmp_path), "--output-root", str(tmp_path / "out")])
    jobs = run_pipeline.make_commands(args, None)
    command = jobs[0]["command"]
    parsed = orchestrator.build_argparser().parse_args(command[2:])
    assert parsed.method == "formal_two_stage" and parsed.organ == "lung"
    assert orchestrator.build_run_specs([], method=parsed.method, out_root=tmp_path,
                                         scales=[], pairs=[], k_by_scale={}, seeds=[42]) == []
    config = build_training_config(build_argparser().parse_args([
        "--method", "formal_two_stage", "--k", "40", "--out-dir", str(tmp_path / "run"),
        "--h5ad-t", "t.h5ad", "--h5ad-tp", "tp.h5ad", "--cci-t", "t.npz", "--cci-tp", "tp.npz",
        "--grn-t", "t.csv", "--grn-tp", "tp.csv", "--maturity-t", "t.csv", "--maturity-tp", "tp.csv"]))
    assert config.objective_version == "formal_two_stage_main"
    assert config.lambda_gap == 1. and config.lambda_keff == 0. and config.lambda_dev == 2.
    assert jobs[-1]["command"][-1] == "lung"
