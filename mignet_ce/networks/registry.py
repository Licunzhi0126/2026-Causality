from __future__ import annotations

from typing import Dict, Type

from mignet_ce.networks.base import NetworkBuilder
from mignet_ce.networks.cross_cell_multilayer import CrossCellMultilayerBuilder
from mignet_ce.networks.legacy_mixed_grn_cci import LegacyMixedGRNCCIBuilder


NETWORK_BUILDERS: Dict[str, Type[NetworkBuilder]] = {
    "legacy_mixed_grn_cci": LegacyMixedGRNCCIBuilder,
    "cross_cell_multilayer": CrossCellMultilayerBuilder,
}


def get_network_builder(network_method: str) -> NetworkBuilder:
    try:
        builder_cls = NETWORK_BUILDERS[network_method]
    except KeyError as exc:
        raise ValueError(f"Unsupported network_method {network_method!r}. Expected one of {sorted(NETWORK_BUILDERS)}.") from exc
    return builder_cls()
