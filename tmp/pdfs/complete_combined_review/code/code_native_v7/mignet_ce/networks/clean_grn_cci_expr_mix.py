from __future__ import annotations

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.clean_grn_cci_mix import CleanGRNCCIMixBuilder


class CleanGRNCCIExpressionMixBuilder(CleanGRNCCIMixBuilder):
    network_method = "clean_grn_cci_expr_mix"

    def _grn_build_options(self, cfg: TemporalRunConfig) -> dict[str, object]:
        return {
            "grn_source": "sample_expression_weighted",
            "expression_weight_mode": cfg.grn_expression_weight_mode,
            "expression_transform": cfg.grn_expression_transform,
            "expression_weight_floor": cfg.grn_expression_weight_floor,
            "unit_specific_fallback": cfg.unit_grn_fallback,
        }

    def _network_metadata(self, cfg: TemporalRunConfig) -> dict[str, object]:
        return {
            "intra_source": "sample_level_grn_expression_weighted",
            "inter_source": "cci_only",
            "grn_source": "sample_expression_weighted",
            "expression_weight_mode": cfg.grn_expression_weight_mode,
            "expression_transform": cfg.grn_expression_transform,
            "expression_weight_floor": float(cfg.grn_expression_weight_floor),
        }
