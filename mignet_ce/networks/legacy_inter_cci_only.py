from __future__ import annotations

from mignet_ce.networks.legacy_mixed_grn_cci import LegacyMixedGRNCCIBuilder


class LegacyInterCCIOnlyBuilder(LegacyMixedGRNCCIBuilder):
    network_method = "legacy_inter_cci_only"
    inter_influence_mode = "cci_only"
    inter_grn_pair_policy = "zero_if_missing"
    include_intra_grn = True
