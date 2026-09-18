from __future__ import annotations

import pandas as pd
import json
import numpy as np
import pytest

from mignet_ce.downstream.paper_assets.config import (
    AssetConfig, DEFAULT_LEVELS, DEFAULT_PIJ_ABLATIONS, INPUT_SCALES, K10_LEVELS, OPTIMAL_TIME_PAIRS,
)
from mignet_ce.downstream.paper_assets.inputs import load_ablation_metrics
from mignet_ce.downstream.paper_assets.tables import (
    build_ei_ablation, build_ei_hierarchy, build_optimal_grid, build_seurat_cg_hierarchy,
    select_optimal_input_runs,
)
from mignet_ce.downstream.paper_assets.workflow import prepare_paper_tables, render_paper_assets


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
            "macro_layer": layer, "time_pair": pair, "closure_quality_for_claim": 0.8,
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
    method = "complete_combined_coarse_maturity_cci_grn"
    k_by_scale = {"spot": 40, "seurat_k150": 40, "seurat_k40": 10}
    for scale in INPUT_SCALES:
        for pair in OPTIMAL_TIME_PAIRS:
            directory = tmp_path / "runs" / method / scale / f"K{k_by_scale[scale]}" / pair.replace("->", "_to_") / "seed_42"
            directory.mkdir(parents=True)
            assignment = np.zeros((2, k_by_scale[scale]))
            assignment[0, 0] = 1.0
            assignment[1, 1] = 1.0
            np.save(directory / "S_t.npy", assignment)
            np.save(directory / "S_tp.npy", assignment)
            np.save(directory / "PIJ_micro_train.npy", np.eye(2))
            (directory / "summary.json").write_text(json.dumps({
                "method": method, "K": k_by_scale[scale], "best_epoch": 5,
                "EI_micro_fixed": 0.5, "EI_macro_best_checkpoint": 0.8,
                "delta_EI_best_checkpoint": 0.3,
            }), encoding="utf-8")
    cfg = AssetConfig(
        vertical_ablation_root=tmp_path / "unused_ablation",
        coarse_root=tmp_path / "unused_legacy",
        data_root=tmp_path / "unused_spatial",
        output_root=tmp_path / "paper",
        multiscale_roots=(tmp_path / "runs",),
    )
    prepare_paper_tables(cfg, ("table4", "table5"))
    result = render_paper_assets(cfg, ("table4", "table5"))
    delta = pd.read_csv(tmp_path / "paper" / "tables" / "table4_deltaei_by_input_scale.csv")
    cg = pd.read_csv(tmp_path / "paper" / "tables" / "table5_cg_by_input_scale.csv")
    long = pd.read_csv(tmp_path / "paper" / "tables" / "optimal_input_cg_long.csv")
    assert delta.shape == cg.shape == (3, 4)
    assert delta.iloc[:, 1:].to_numpy(float) == pytest.approx(0.3)
    assert cg.iloc[:, 1:].to_numpy(float) == pytest.approx(1.0)
    assert len(long) == 9
    assert result["tables_unchanged"] is True
