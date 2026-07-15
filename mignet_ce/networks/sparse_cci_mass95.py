from __future__ import annotations

from mignet_ce.networks.light_cci import LightCCINetworkBuilder


class SparseCCIMass95NetworkBuilder(LightCCINetworkBuilder):
    network_method = "sparse_cci_mass95"
    cci_mass_keep_ratio = 0.95
