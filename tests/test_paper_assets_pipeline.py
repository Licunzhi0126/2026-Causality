from __future__ import annotations

import pandas as pd
import json
import numpy as np
import pytest

from mignet_ce.downstream.paper_assets.config import (
    AssetConfig, DEFAULT_LEVELS, DEFAULT_PIJ_ABLATIONS, INPUT_SCALES, K10_LEVELS,
    OPTIMAL_TIME_PAIRS, TABLE1_CHAIN_LEVELS, TABLE1_CROSS_LEVELS,
)
from mignet_ce.pij.ablation.methods import METHOD_SPECS
from mignet_ce.downstream.paper_assets.inputs import load_ablation_metrics
from mignet_ce.downstream.paper_assets.tables import (
    build_ei_ablation, build_ei_hierarchy, build_optimal_grid, build_seurat_cg_hierarchy,
    select_optimal_input_runs,
)
from mignet_ce.downstream.paper_assets.workflow import prepare_paper_tables, render_paper_assets
from mignet_ce.downstream.paper_assets import workflow as paper_workflow
from scripts import build_paper_assets as paper_assets_cli


def test_ablation_identity_uses_network_and_pij(tmp_path) -> None:
    for folder, network, value in (
        ("02_NG_BlockKL", "light_cci_grn", 0.25),
        ("03_N_KL", "light_cci", 0.75),
    ):
        directory = tmp_path / folder
        directory.mkdir()
        pd.DataFrame([{
            "network_method": network, "pij_method": "compare_N_kl", "organ": "heart",
            "lower_layer": "spot", "upper_layer": "seurat_k150",
            "time_pair": "11.5->12.5", "EI_gain": value,
        }]).to_csv(directory / "metrics.csv", index=False)
    metrics = load_ablation_metrics((tmp_path,), dict(DEFAULT_PIJ_ABLATIONS))
    assert metrics.set_index("Method").loc["NG_BlockKL", "EI_gain"] == pytest.approx(0.25)
    assert metrics.set_index("Method").loc["N_KL", "EI_gain"] == pytest.approx(0.75)


def test_k10_tables_and_matched_input_grid() -> None:
    pairs = ("11.5->12.5", "11.5->13.5", "12.5->13.5")
    levels = {**DEFAULT_LEVELS, **K10_LEVELS}
    ei_rows = [
        {
            "Method": "full_NG_KLot", "lower_layer": lower, "upper_layer": upper,
            "time_pair": pair, "EI_gain": float(level_index + pair_index / 10),
        }
        for level_index, (lower, upper) in enumerate(levels.values(), 1)
        for pair_index, pair in enumerate(pairs, 1)
    ]
    ei = pd.DataFrame(ei_rows)
    hierarchy = build_ei_hierarchy(ei, levels, pairs)
    assert hierarchy.shape == (3, 7)
    assert hierarchy.loc[1, "Spot -> K10"] == pytest.approx(6.2)
    ablation = build_ei_ablation(ei, K10_LEVELS, ("full_NG_KLot",), pairs)
    assert ablation.shape == (3, 5)

    cg = pd.DataFrame([
        {
            "macro_layer": layer, "time_pair": pair, "closure_quality": 0.8,
            "signal_status": "informative",
        }
        for layer in ("seurat_k150", "seurat_k40", "seurat_k10")
        for pair in pairs
    ])
    natural = build_seurat_cg_hierarchy(cg, pairs)
    assert list(natural.columns) == ["Time pair", "Spot -> K150", "Spot -> K40", "Spot -> K10"]

    optimized = pd.DataFrame([
        {"input_scale": scale, "time_pair": pair, "delta": float(scale_index * 10 + pair_index)}
        for scale_index, scale in enumerate(INPUT_SCALES)
        for pair_index, pair in enumerate(OPTIMAL_TIME_PAIRS)
    ])
    grid = build_optimal_grid(optimized, "delta")
    assert list(grid.columns) == ["Input scale", *OPTIMAL_TIME_PAIRS]
    assert grid.loc[2, "11.5->13.5"] == pytest.approx(21.0)


def test_optimal_selection_uses_input_specific_k_and_best_checkpoint() -> None:
    expected_k = {"spot": 40, "seurat_k150": 40, "seurat_k40": 10}
    rows = []
    for scale in INPUT_SCALES:
        for pair in OPTIMAL_TIME_PAIRS:
            rows.append({
                "method": "complete_combined_coarse_maturity_cci_grn",
                "scale": scale, "time_pair": pair, "K": expected_k[scale], "seed": 42,
                "run_dir": f"/{scale}/{pair}",
                "EI_micro_fixed": 0.4, "EI_macro_best_checkpoint": 0.9,
                "delta_EI_best_checkpoint": 0.5,
            })
    selected = select_optimal_input_runs(
        pd.DataFrame(rows), method="complete_combined_coarse_maturity_cci_grn",
        k_by_scale=expected_k, seed=42,
    )
    assert len(selected) == 9
    assert selected.groupby("input_scale")["K"].first().to_dict() == expected_k
    assert build_optimal_grid(selected, "delta_EI_best_checkpoint").shape == (3, 4)


def test_paper_workflow_builds_matched_deltaei_and_cg_tables(tmp_path) -> None:
    methods = (
        "complete_combined_coarse_maturity_cci_grn",
        "maturity_cci_grn_two_stage",
    )
    k_by_scale = {"spot": 40, "seurat_k150": 40, "seurat_k40": 10}
    expected_delta = {
        "complete_combined_coarse_maturity_cci_grn": 0.3,
        "maturity_cci_grn_two_stage": 0.4,
    }
    for method in methods:
        for scale in INPUT_SCALES:
            for pair in OPTIMAL_TIME_PAIRS:
                directory = tmp_path / "runs" / method / scale / f"K{k_by_scale[scale]}" / pair.replace("->", "_to_")
                directory.mkdir(parents=True)
                assignment = np.zeros((2, k_by_scale[scale]))
                assignment[0, 0] = 1.0
                assignment[1, 1] = 1.0
                np.save(directory / "S_t.npy", assignment)
                np.save(directory / "S_tp.npy", assignment)
                np.save(directory / "PIJ_micro_train.npy", np.eye(2))
                delta = expected_delta[method]
                (directory / "summary.json").write_text(json.dumps({
                    "method": method, "K": k_by_scale[scale], "best_epoch": 5,
                    "EI_micro_fixed": 0.5, "EI_macro_best_checkpoint": 0.5 + delta,
                    "delta_EI_best_checkpoint": delta,
                }), encoding="utf-8")
                (directory / "config.json").write_text(
                    json.dumps({"seed": 42, "k": k_by_scale[scale]}), encoding="utf-8"
                )
    cfg = AssetConfig(
        vertical_ablation_root=tmp_path / "unused_ablation",
        coarse_root=tmp_path / "unused_legacy",
        data_root=tmp_path / "unused_spatial",
        output_root=tmp_path / "paper",
        multiscale_roots=tuple(tmp_path / "runs" / method for method in methods),
        optimal_coarse_methods=methods,
    )
    prepared = prepare_paper_tables(cfg, ("table4", "table5"))
    result = render_paper_assets(cfg, ("table4", "table5"))
    delta = pd.read_csv(tmp_path / "paper" / "tables" / "table4_deltaei_by_input_scale.csv")
    cg = pd.read_csv(tmp_path / "paper" / "tables" / "table5_cg_by_input_scale.csv")
    long = pd.read_csv(tmp_path / "paper" / "tables" / "optimal_input_cg_long.csv")
    assert delta.shape == cg.shape == (3, 4)
    assert delta.iloc[:, 1:].to_numpy(float) == pytest.approx(0.3)
    assert cg.iloc[:, 1:].to_numpy(float) == pytest.approx(1.0)
    assert len(long) == 9
    assert result["tables_unchanged"] is True
    assert set(prepared["optimal_methods"]) == set(methods)
    for method in methods:
        method_root = tmp_path / "paper" / "tables" / "optimal" / method
        method_delta = pd.read_csv(method_root / "table4_deltaei_by_input_scale.csv")
        method_cg = pd.read_csv(method_root / "table5_cg_by_input_scale.csv")
        method_long = pd.read_csv(method_root / "optimal_input_cg_long.csv")
        assert method_delta.iloc[:, 1:].to_numpy(float) == pytest.approx(expected_delta[method])
        assert method_cg.iloc[:, 1:].to_numpy(float) == pytest.approx(1.0)
        assert set(method_long["method"]) == {method}
        figure_root = tmp_path / "paper" / "figures" / "optimal" / method
        assert (figure_root / "table4_deltaei_by_input_scale.png").exists()
        assert (figure_root / "table5_cg_by_input_scale.png").exists()
    assert set(result["outputs"]["optimal_methods"]) == set(methods)


def test_optimal_method_config_defaults_and_rejects_duplicates(tmp_path) -> None:
    common = dict(
        vertical_ablation_root=tmp_path,
        coarse_root=tmp_path,
        data_root=tmp_path,
        output_root=tmp_path / "paper",
    )
    cfg = AssetConfig(**common)
    assert cfg.resolved_optimal_coarse_methods() == (cfg.primary_coarse_method,)
    duplicate = AssetConfig(
        **common,
        optimal_coarse_methods=("maturity_cci_grn_two_stage", "maturity_cci_grn_two_stage"),
    )
    with pytest.raises(ValueError, match="duplicates"):
        duplicate.validate()


def test_paper_assets_cli_accepts_multiple_optimal_methods(tmp_path) -> None:
    args = paper_assets_cli.build_parser().parse_args([
        "--output-root", str(tmp_path / "paper"),
        "--primary-coarse-method", "complete_combined_coarse_maturity_cci_grn",
        "--optimal-method", "complete_combined_coarse_maturity_cci_grn",
        "--optimal-method", "maturity_cci_grn_two_stage",
    ])
    assert args.primary_coarse_method == "complete_combined_coarse_maturity_cci_grn"
    assert args.optimal_method == [
        "complete_combined_coarse_maturity_cci_grn",
        "maturity_cci_grn_two_stage",
    ]


def test_controlled_feature_ablation_emits_separate_table1a_and_table1b(tmp_path) -> None:
    pairs = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
    levels = {**TABLE1_CHAIN_LEVELS, **TABLE1_CROSS_LEVELS}
    rows = []
    for level_index, (hierarchy, (lower, upper)) in enumerate(levels.items()):
        for method_index, spec in enumerate(METHOD_SPECS):
            for pair_index, pair in enumerate(pairs):
                rows.append({
                    "method_id": spec.method_id,
                    "input_group": spec.input_group,
                    "feature_method": spec.feature_method,
                    "pij_construction": spec.pij_construction,
                    "organ": "heart",
                    "lower_layer": lower,
                    "upper_layer": upper,
                    "hierarchy": hierarchy,
                    "time_pair": pair,
                    "EI_lower": 0.1,
                    "EI_upper": 0.2,
                    "delta_EI": level_index + method_index / 10 + pair_index / 100,
                    "alpha_cci": 0.01,
                    "beta_n": 0.05,
                    "beta_l": 0.05,
                    "beta_g": 0.05,
                    "temperature": 0.8,
                    "ot_enabled": spec.use_ot,
                })
    feature_root = tmp_path / "feature_ablation"
    feature_root.mkdir()
    pd.DataFrame(rows).to_csv(feature_root / "feature_ablation_long.csv", index=False)
    cfg = AssetConfig(
        vertical_ablation_root=tmp_path / "unused_legacy_ablation",
        coarse_root=tmp_path / "unused_coarse",
        data_root=tmp_path / "unused_data",
        output_root=tmp_path / "paper",
        feature_ablation_root=feature_root,
        dpi=72,
    )
    prepare_paper_tables(cfg, ("table1", "table1_k10"))
    rendered = render_paper_assets(cfg, ("table1", "table1_k10"))

    table1a = tmp_path / "paper" / "tables" / "table1A_pij_feature_ablation_chain.csv"
    table1b = tmp_path / "paper" / "tables" / "table1B_pij_feature_ablation_cross.csv"
    assert len(pd.read_csv(table1a)) == 30
    assert len(pd.read_csv(table1b)) == 30
    for stem in (
        "table1A_pij_feature_ablation_chain",
        "table1B_pij_feature_ablation_cross",
    ):
        assert (tmp_path / "paper" / "figures" / f"{stem}.png").is_file()
        assert (tmp_path / "paper" / "figures" / f"{stem}.pdf").is_file()
    assert rendered["tables_unchanged"] is True

    args = paper_assets_cli.build_parser().parse_args([
        "--output-root", str(tmp_path / "paper_cli"),
        "--feature-ablation-root", str(feature_root),
        "--asset", "table1",
        "--asset", "table1_k10",
    ])
    assert args.feature_ablation_root == feature_root
    assert paper_assets_cli.main([
        "--stage", "prepare",
        "--output-root", str(tmp_path / "paper_cli"),
        "--feature-ablation-root", str(feature_root),
        "--asset", "table1",
        "--asset", "table1_k10",
    ]) == 0


def test_optimal_figure2_is_method_scoped_and_preserves_legacy_asset(tmp_path, monkeypatch) -> None:
    methods = (
        "complete_combined_coarse_maturity_cci_grn",
        "maturity_cci_grn_two_stage",
    )
    for index, method in enumerate(methods):
        directory = tmp_path / "runs" / method / "spot" / "K40" / "12.5_to_13.5"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps({
                "method": method,
                "K": 40,
                "best_epoch": 5,
                "EI_micro_fixed": 0.5,
                "EI_macro_best_checkpoint": 0.8 + index / 10,
                "delta_EI_best_checkpoint": 0.3 + index / 10,
            }),
            encoding="utf-8",
        )
        (directory / "config.json").write_text(
            json.dumps({"seed": 42, "k": 40}), encoding="utf-8"
        )

    rendered_methods: list[str] = []

    def fake_render_figure2(**kwargs):
        rendered_methods.append(str(kwargs["method_label"]))
        output = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"png")
        output.with_suffix(".pdf").write_bytes(b"pdf")

    monkeypatch.setattr(paper_workflow, "render_figure2", fake_render_figure2)
    monkeypatch.setattr(
        paper_workflow,
        "load_spot_coordinates",
        lambda *_args, **_kwargs: pd.DataFrame({"spot_id": ["a"], "x": [0.0], "y": [0.0]}),
    )
    monkeypatch.setattr(paper_workflow, "load_full_slice", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        paper_workflow,
        "load_assignments",
        lambda *_args, **_kwargs: pd.DataFrame({"spot_id": ["a"], "hard_cluster": ["0"]}),
    )
    cfg = AssetConfig(
        vertical_ablation_root=tmp_path / "unused_ablation",
        coarse_root=tmp_path / "unused_legacy",
        data_root=tmp_path / "unused_spatial",
        output_root=tmp_path / "paper",
        multiscale_roots=(tmp_path / "runs",),
        optimal_coarse_methods=methods,
    )
    prepared = prepare_paper_tables(cfg, ("figure2_optimal",))
    rendered = render_paper_assets(cfg, ("figure2_optimal",))
    assert set(prepared["optimal_methods"]) == set(methods)
    assert set(rendered_methods) == set(methods)
    assert set(rendered["outputs"]["optimal_methods"]) == set(methods)
    for method in methods:
        audit = json.loads(
            (tmp_path / "paper" / "audit" / "optimal" / method / "figure2_run.json").read_text(
                encoding="utf-8"
            )
        )
        assert audit["input_scale"] == "spot"
        assert audit["K"] == 40
        assert audit["time_pair"] == "12.5->13.5"
        assert audit["seed"] == 42
        figure = tmp_path / "paper" / "figures" / "optimal" / method / "figure2_optimal_coarse_12p5_to_13p5.png"
        assert figure.exists()
    assert not (tmp_path / "paper" / "audit" / "figure2_run.json").exists()
    assert not (tmp_path / "paper" / "figures" / "figure2_optimal_coarse_12p5_to_13p5.png").exists()
