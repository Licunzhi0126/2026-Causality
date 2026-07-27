from __future__ import annotations

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.clean_grn_cci_mix import CleanGRNCCIMixBuilder


class UnitSpecificCleanGRNCCIMixBuilder(CleanGRNCCIMixBuilder):
    network_method = "unit_specific_clean_grn_cci_mix"

    def _grn_build_options(self, cfg: TemporalRunConfig) -> dict[str, object]:
        return {
            "grn_source": "unit_specific",
            "expression_weight_mode": cfg.grn_expression_weight_mode,
            "expression_transform": cfg.grn_expression_transform,
            "expression_weight_floor": cfg.grn_expression_weight_floor,
            "unit_specific_fallback": cfg.unit_grn_fallback,
        }

    def _network_metadata(self, cfg: TemporalRunConfig) -> dict[str, object]:
        return {
            "intra_source": "unit_specific_grn_with_configured_fallback",
            "inter_source": "cci_only",
            "grn_source": "unit_specific",
            "unit_grn_fallback": cfg.unit_grn_fallback,
            "fallback_expression_weight_mode": cfg.grn_expression_weight_mode,
            "fallback_expression_transform": cfg.grn_expression_transform,
        }
