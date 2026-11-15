"""Domain projections, signed comparisons and preservation of the spatial layout."""

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mignet_ce.downstream.domain_visualization.inputs import StageMaps, _load_projection
from mignet_ce.downstream.domain_visualization.metrics import (
    METRICS, baseline_differences, hard_occupancy, metric_lines,
)
from mignet_ce.downstream.domain_visualization.variants import VARIANTS


def _row(name, **changes):
    values = {key: .5 for key in METRICS}
    values.update(hard_k_t=5, hard_k_tp=4, **changes)
    return dict(values, variant=name, input_scale="spot", source_time="11.5", target_time="12.5", k=40)


def test_differences_are_signed_with_matching_baseline_and_zero_is_not_improvement():
    rows = baseline_differences([_row("baseline"), _row("closure", cq=.7, delta_ei=.4)])
    assert rows[1]["delta_vs_baseline_cq"] == pytest.approx(.2)
    assert rows[1]["delta_vs_baseline_delta_ei"] == pytest.approx(-.1)
    assert all(rows[0]["delta_vs_baseline_" + key] == 0 for key in METRICS)
    assert "CQ: +0.2000" in metric_lines(rows[1], baseline=False)
    assert metric_lines(rows[1], baseline=False)[0] == "ΔEI: -0.1000"
    with pytest.raises(ValueError, match="Missing baseline"):
        baseline_differences([_row("closure")])
    with pytest.raises(ValueError, match="Duplicate"):
        baseline_differences([_row("baseline"), _row("baseline")])
    with pytest.raises(ValueError, match="Missing baseline"):
        baseline_differences([_row("baseline"), dict(_row("closure"), target_time="13.5")])


def test_hard_occupancy_uses_original_input_units_and_rejects_bad_probabilities():
    soft = np.array([[.9, .1], [.8, .2], [.3, .7]])
    count, effective = hard_occupancy(soft)
    assert count == 2
    assert effective == pytest.approx(np.exp(-2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3)))
    for invalid in (np.array([[np.nan, 0]]), np.array([[.2, .2]]), np.array([[-.1, 1.1]])):
        with pytest.raises(ValueError):
            hard_occupancy(invalid)


def test_projection_preserves_spot_order_and_validates_hard_and_soft_labels(tmp_path):
    ids = pd.Index(["unit_b", "unit_a"])
    mapping = pd.Series(["unit_a", "unit_b", "unit_b"], index=["s3", "s1", "s2"])
    pd.DataFrame({"spot_id": ids, "hard_cluster": [1, 0]}).to_csv(tmp_path / "assignments_t.csv", index=False)
    np.save(tmp_path / "S_t.npy", np.array([[.1, .9], [.8, .2]]))
    result = _load_projection(tmp_path, "t", ids, mapping, 2, 1.9)
    assert result.labels.index.equals(mapping.index)
    assert result.labels.tolist() == [0, 1, 1]
    np.save(tmp_path / "S_t.npy", np.array([[.9, .1], [.8, .2]]))
    with pytest.raises(ValueError, match="Hard cluster"):
        _load_projection(tmp_path, "t", ids, mapping, 2, 1.9)


def test_six_column_plot_reuses_spatial_geometry_and_labels_comparisons(tmp_path, monkeypatch):
    from mignet_ce.downstream.domain_visualization import plots
    from mignet_ce.downstream.domain_visualization.inputs import LearnedDomains

    coordinates = pd.DataFrame({"x": [1., 2., 3.], "y": [4., 6., 5.]}, index=["a", "b", "c"])
    stages = (StageMaps("11.5", coordinates, {}), StageMaps("12.5", coordinates, {}))
    learned = LearnedDomains(pd.Series([0, 1, 0], index=coordinates.index), 1.8)
    domains = {variant.name: (learned, learned) for variant in VARIANTS}
    rows = [_row(variant.name) for variant in VARIANTS]
    captured = []

    def inspect(fig, path):
        captured.append(fig)
        texts = [text.get_text() for text in fig.texts]
        assert all(variant.label in texts for variant in VARIANTS)
        assert texts.count("Change vs Baseline") == 5 and texts.count("Absolute values") == 1
        assert len(fig.axes) == 24
        for axes in fig.axes:
            np.testing.assert_array_equal(axes.collections[0].get_offsets(), coordinates.to_numpy())
            assert axes.yaxis_inverted() and axes.get_aspect() == 1.0 and not axes.axison
        plt.close(fig)

    monkeypatch.setattr(plots, "_save", inspect)
    plots.render_variants(stages, domains, rows, tmp_path / "six")
    assert len(captured) == 1
    with pytest.raises(ValueError, match="all selected"):
        plots.render_variants(stages, {"baseline": (learned, learned)}, rows, tmp_path / "bad")


def test_domain_modules_have_no_imports_from_loss_experiments():
    root = Path(__file__).resolve().parents[1] / "mignet_ce/downstream/domain_visualization"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("loss_ablation" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "loss_ablation" not in (node.module or "")


def test_export_preserves_decimal_time_points_in_filenames(tmp_path):
    from mignet_ce.downstream.domain_visualization.plots import _save

    _save(plt.figure(figsize=(1, 1)), tmp_path / "optimal_12.5_to_13.5")
    assert (tmp_path / "optimal_12.5_to_13.5.png").is_file()
    assert (tmp_path / "optimal_12.5_to_13.5.pdf").is_file()


@pytest.mark.parametrize("experiment", ["single_factor", "factorial"])
def test_render_complete_manifest_writes_absolute_and_difference_metrics_without_training(tmp_path, monkeypatch, experiment):
    from dataclasses import asdict
    from types import SimpleNamespace
    import csv
    import h5py
    from mignet_ce.downstream.domain_visualization import optimal_pipeline, workflow
    from mignet_ce.downstream.domain_visualization.variants import TIME_PAIRS, variants_for_experiment
    from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig

    data = tmp_path / "data"
    layer = data / "spot/heart"
    layer.mkdir(parents=True)
    input_hashes = {}
    for stage in ("11.5", "12.5", "13.5"):
        path = layer / f"spot_heart_{stage}.h5ad"
        with h5py.File(path, "w") as handle:
            obs, var = handle.create_group("obs"), handle.create_group("var")
            obs.attrs["_index"] = var.attrs["_index"] = "_index"
            obs.create_dataset("_index", data=np.array([f"u{i}" for i in range(40)], dtype="S"))
            var.create_dataset("_index", data=np.array(["gene"], dtype="S"))
            handle.create_dataset("obsm/spatial", data=np.column_stack((np.arange(40), np.arange(40) % 7)))
        input_hashes[str(path)] = workflow.file_digest(path)
    root = tmp_path / "experiment"
    root.mkdir()
    plan = dict(data_root=str(data), input_sha256=input_hashes, configurations={})
    if experiment == "factorial":
        plan["experiment"] = experiment
    variants = variants_for_experiment(experiment)
    entries = []
    for index, variant in enumerate(variants):
        cfg = asdict(WYTFormalTwoStageConfig(k=40, out_dir=Path("unused"), epochs=4, **variant.weights))
        expected = {key: value for key, value in cfg.items() if key not in {"out_dir", "stage1_source", "stage1_only"}}
        plan["configurations"][variant.name] = expected
        for pair in TIME_PAIRS:
            run = root / variant.name / "_to_".join(pair)
            run.mkdir(parents=True)
            for name in workflow.FINAL_ARTIFACTS:
                (run / name).write_bytes(b"fixture")
            workflow.save_json(run / "config.json", dict(cfg, out_dir=str(run)))
            available, retained = .2 + index * .01, .1
            summary = dict(method="formal_two_stage", loss_protocol="formal_two_stage_main", K=40,
                           selected_checkpoint="best_joint.pt", checkpoint_metric_state_consistent=True,
                           best_epoch=4, best_joint_selection="strict_retained_first", best_joint_fallback_used=False,
                           delta_EI_best_checkpoint=1 + index * .1, EI_macro_best_checkpoint=1.5 + index * .1,
                           EI_micro_fixed=.5, closure_quality=retained / available, I_available=available, I_retained=retained,
                           hardK_t=40, hardK_tp=40, KeffHard_t=40., KeffHard_tp=40., Keff_t=40., Keff_tp=40.,
                           closure_js_mean=.05, closure_js_median=.04, closure_js_p90=.08, L_dev_best_checkpoint=.02,
                           stage1_best_delta_EI=1.1, stage1_reference_I_available=.3, stage1_reference_I_retained=.15,
                           signal_status_reference="informative")
            workflow.save_json(run / "summary.json", summary)
            workflow.save_json(run / "checkpoint_consistency.json", dict(passed=True, selected_epoch=4, selected_checkpoint="best_joint.pt"))
            workflow.save_json(run / "feature_manifest.json", {"provenance": {
                f"h5ad_{side}": str(layer / f"spot_heart_{stage}.h5ad") for side, stage in zip(("t", "tp"), pair)
            }})
            for side in ("t", "tp"):
                np.save(run / f"S_{side}.npy", np.eye(40))
                pd.DataFrame({"spot_id": [f"u{i}" for i in range(40)], "hard_cluster": range(40)}).to_csv(
                    run / f"assignments_{side}.csv", index=False)
            history = [dict(epoch=epoch, stage="stage1" if epoch <= 2 else "stage2",
                            L_ei_constraint=0., L_retain_floor=.1 if epoch == 3 else 0.,
                            weighted_ei_constraint=0., weighted_retain_floor=.1 if epoch == 3 else 0.,
                            weighted_dev=.04, weighted_closure=.05, weighted_retain_norm=-.2) for epoch in range(1, 5)]
            pd.DataFrame(history).to_csv(run / "metrics.csv", index=False)
            entries.append(dict(variant=variant.name, time_pair=list(pair), stage2_run=str(run),
                                artifacts={name: workflow.file_digest(run / name) for name in workflow.FINAL_ARTIFACTS}))
    workflow.save_json(root / "run_config.json", plan)
    workflow.save_json(root / "run_manifest.json", dict(schema_version=1, status="complete", results=entries,
                                                       run_config_sha256=workflow.file_digest(root / "run_config.json")))

    def forbidden(*args, **kwargs):
        raise AssertionError("Render must never train")

    monkeypatch.setattr(workflow, "run", forbidden)
    output = optimal_pipeline.render(SimpleNamespace(data_root=data, output_root=root))
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 3
    assert (output / "optimal_domains_spot_12.5_to_13.5.png").is_file()
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(variants) * 3
    expected_difference = -.6 if experiment == "factorial" else .1
    assert float(rows[3]["delta_vs_baseline_delta_ei"]) == pytest.approx(expected_difference)
    if experiment == "factorial":
        assert all(row["comparison_reference"] == "s8" for row in rows)
        assert (output / "pareto_candidates.json").is_file()
        effects = pd.read_csv(output / "factorial_effects.csv")
        assert len(effects) == 21
        activation = pd.read_csv(output / "penalty_activation.csv")
        assert len(activation) == 24 and (activation["retain_floor_active_fraction"] == .5).all()
        audit_out = optimal_pipeline.audit_existing(SimpleNamespace(history_root=root, output_root=tmp_path / "audit"))
        assert len(pd.read_csv(audit_out / "penalty_activation.csv")) == 24
    (Path(entries[0]["stage2_run"]) / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed domain result"):
        optimal_pipeline.render(SimpleNamespace(data_root=data, output_root=root))


def test_factorial_effects_recover_main_and_interaction_terms_and_s8_differences():
    from mignet_ce.downstream.domain_visualization.variants import FACTORIAL_VARIANTS, TIME_PAIRS
    from mignet_ce.downstream.domain_visualization.metrics import EXTRA_COMPARISONS, factorial_effects

    rows = []
    for variant in FACTORIAL_VARIANTS:
        w = variant.weights
        c, d, r = (1 if w["lambda_closure"] == 1 else -1,
                   1 if w["lambda_dev"] == 2 else -1,
                   1 if w["lambda_retain_norm"] == 1 else -1)
        for pair in TIME_PAIRS:
            row = dict(_row(variant.name), source_time=pair[0], target_time=pair[1])
            row.update({name: .1 for name in EXTRA_COMPARISONS})
            row.update(delta_ei=20 + 2 * c + 3 * d + 5 * r + 7 * c * d)
            rows.append(row)
    effects = factorial_effects(rows)
    first_pair = {row["term"]: row["effect_delta_ei"] for row in effects[:7]}
    assert first_pair == pytest.approx({"C": 4, "D": 6, "R": 10, "C:D": 14, "C:R": 0, "D:R": 0, "C:D:R": 0})
    differences = baseline_differences(rows, "s8")
    assert all(row["comparison_reference"] == "s8" for row in differences)
    assert all(row["delta_vs_baseline_delta_ei"] == 0 for row in differences if row["variant"] == "s8")


def test_pareto_retains_tradeoffs_without_averaging_time_pairs_or_choosing_a_winner():
    from mignet_ce.downstream.domain_visualization.variants import FACTORIAL_VARIANTS, TIME_PAIRS
    from mignet_ce.downstream.domain_visualization.metrics import pareto_report

    rows = []
    for variant in FACTORIAL_VARIANTS:
        ei, cq, js = {"s1": (1, .2, .5), "s2": (2, .3, .4), "s3": (3, .1, .4)}.get(variant.name, (.5, .05, .9))
        for pair in TIME_PAIRS:
            rows.append(dict(variant=variant.name, source_time=pair[0], target_time=pair[1],
                             delta_ei=ei, cq=cq, closure_js_mean=js))
    report = pareto_report(rows)
    candidates = {row["variant"]: row for row in report["cross_time_pair"]}
    assert "s2" in candidates["s1"]["dominated_by"]
    assert candidates["s2"]["pareto_optimal"] and candidates["s3"]["pareto_optimal"]
    assert not candidates["s8"]["pareto_optimal"]
    with pytest.raises(ValueError, match="complete"):
        pareto_report(rows[:-1])
