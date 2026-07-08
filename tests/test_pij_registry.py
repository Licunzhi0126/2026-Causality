from __future__ import annotations

from mignet_ce.config import COMPARE_PIJ_METHODS, DEVELOPMENT_PIJ_METHODS, PIJ_METHODS, PIJ_METHOD_PRESETS
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def test_pij_registry_matches_configured_methods() -> None:
    assert set(PIJ_METHOD_REGISTRY) == PIJ_METHODS
    assert {get_pij_method(name).name for name in PIJ_METHODS} == PIJ_METHODS
    assert {"joint_nmf", "laplacian", "3dot", "slat"}.issubset(PIJ_METHODS)
    assert {"expr_ot", "energy_entropy_ot"}.issubset(PIJ_METHODS)
    assert {"pure_expression_ot"}.issubset(PIJ_METHODS)
    assert PIJ_METHOD_PRESETS["pure_expression"] == ("pure_expression_ot",)
    assert set(PIJ_METHOD_PRESETS["ot_ablation_v2"]) == {
        "sr_ot",
        "pseudotime_ot",
        "spatial_ot",
        "sr_spatial_ot",
        "pseudotime_spatial_ot",
        "sr_expression_ot",
        "pseudotime_expression_ot",
    }
    assert "velocity_ot" not in PIJ_METHOD_PRESETS["ot_ablation_v2"]
    assert PIJ_METHOD_PRESETS["ot_ablation_v3"] == (
        "energy_ot",
        "expr_pseudotime_sr_ot",
        "expr_pseudotime_sr_spatial_ot",
        "expr_pseudotime_sr_energy_ot",
    )
    assert PIJ_METHOD_PRESETS["ot_ablation_v4"] == (
        "expr_pseudotime_sr_energy_ot",
        "expr_pseudotime_sr_energy_spatial_ot",
    )
    assert "energy_ot" not in DEVELOPMENT_PIJ_METHODS
    assert {
        "expr_pseudotime_sr_ot",
        "expr_pseudotime_sr_spatial_ot",
        "expr_pseudotime_sr_energy_ot",
        "expr_pseudotime_sr_energy_spatial_ot",
    }.issubset(DEVELOPMENT_PIJ_METHODS)
    assert len(COMPARE_PIJ_METHODS) == 30
    assert set(PIJ_METHOD_PRESETS["lightcci_compare_matrix"]) == set(COMPARE_PIJ_METHODS)
    assert PIJ_METHOD_PRESETS["lightcci_main"] == ("compare_main_lap_sr_spatial_sot",)
    assert set(PIJ_METHOD_PRESETS["lightcci_all"]) == {*COMPARE_PIJ_METHODS, "compare_main_lap_sr_spatial_sot"}
    assert {
        "compare_E_cos",
        "compare_N_cos",
        "compare_L_sot",
        "compare_L_Sr_sot",
        "compare_main_lap_sr_spatial_sot",
    }.issubset(PIJ_METHOD_REGISTRY)
    assert "compare_N_cos" not in DEVELOPMENT_PIJ_METHODS
    assert "compare_L_Sr_sot" in DEVELOPMENT_PIJ_METHODS
    assert "compare_main_lap_sr_spatial_sot" in DEVELOPMENT_PIJ_METHODS
    assert set(PIJ_METHOD_PRESETS["all"]) == PIJ_METHODS
