from __future__ import annotations

from mignet_ce.networks.legacy_mixed_grn_cci import LegacyMixedGRNCCIBuilder


class LegacyInterAdditiveGRNCCIBuilder(LegacyMixedGRNCCIBuilder):
    network_method = "legacy_inter_additive_grn_cci"
    inter_influence_mode = "additive"
    inter_additive_cci_weight = 1.0
    inter_additive_grn_weight = 1.0
    inter_grn_pair_policy = "require_pair"
    include_intra_grn = True
