from __future__ import annotations

from mignet_ce.config import PIJ_METHODS, PIJ_METHOD_PRESETS
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def test_pij_registry_matches_configured_methods() -> None:
    assert set(PIJ_METHOD_REGISTRY) == PIJ_METHODS
    assert {get_pij_method(name).name for name in PIJ_METHODS} == PIJ_METHODS
    assert {"joint_nmf", "laplacian", "3dot", "slat"}.issubset(PIJ_METHODS)
    assert {"expr_ot", "energy_entropy_ot"}.issubset(PIJ_METHODS)
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
    assert set(PIJ_METHOD_PRESETS["all"]) == PIJ_METHODS
