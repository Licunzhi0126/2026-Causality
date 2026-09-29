from __future__ import annotations

import pytest

from mignet_ce.config import (
    COMPARE_PIJ_METHODS,
    COST_FUSION_EXPERIMENT_PIJ_METHODS,
    COST_FUSION_NEW_PIJ_METHODS,
    DEVELOPMENT_PIJ_METHODS,
    PIJ_METHODS,
    PIJ_METHOD_PRESETS,
    TemporalRunConfig,
)
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY


def test_cost_fusion_methods_and_presets_are_registered_without_changing_old_matrix() -> None:
    assert len(COMPARE_PIJ_METHODS) == 30
    assert len(COST_FUSION_NEW_PIJ_METHODS) == 10
    assert len(COST_FUSION_EXPERIMENT_PIJ_METHODS) == 12
    assert set(COST_FUSION_NEW_PIJ_METHODS) <= PIJ_METHODS
    assert set(COST_FUSION_NEW_PIJ_METHODS) <= set(PIJ_METHOD_REGISTRY)
    assert PIJ_METHOD_PRESETS["lightcci_compare_matrix"] == COMPARE_PIJ_METHODS
    assert PIJ_METHOD_PRESETS["lightcci_cost_fusion"] == COST_FUSION_EXPERIMENT_PIJ_METHODS
    assert PIJ_METHOD_PRESETS["lightcci_cost_fusion_new_only"] == COST_FUSION_NEW_PIJ_METHODS


def test_sr_costmix_methods_require_development_features() -> None:
    sr_methods = [name for name in COST_FUSION_NEW_PIJ_METHODS if "_Sr_costmix" in name]
    assert set(sr_methods) <= DEVELOPMENT_PIJ_METHODS
    cfg = TemporalRunConfig(pij_method=sr_methods[0], network_method="light_cci")
    with pytest.raises(ValueError, match="requires development_feature_root"):
        cfg.validate()


@pytest.mark.parametrize("field", ["compare_cost_weight_l", "compare_cost_weight_e", "compare_cost_weight_sr"])
def test_cost_fusion_weights_must_be_nonnegative(field: str) -> None:
    cfg = TemporalRunConfig(
        pij_method="compare_L_euc_sot",
        network_method="light_cci",
        **{field: -0.1},
    )
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_registered_method_names_encode_their_metrics() -> None:
    for name in COST_FUSION_NEW_PIJ_METHODS:
        method_cls = PIJ_METHOD_REGISTRY[name]
        expected = "euclidean" if "_euc_" in name or name.endswith("_euc_sot") else "cosine"
        assert method_cls.vector_metric == expected
