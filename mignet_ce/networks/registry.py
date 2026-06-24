from __future__ import annotations

from typing import Dict, Type

from mignet_ce.networks.base import NetworkBuilder
from mignet_ce.networks.cross_cell_multilayer import CrossCellMultilayerBuilder
from mignet_ce.networks.expression_only import ExpressionOnlyBuilder
from mignet_ce.networks.legacy_inter_additive_grn_cci import LegacyInterAdditiveGRNCCIBuilder
from mignet_ce.networks.legacy_inter_cci_only import LegacyInterCCIOnlyBuilder
from mignet_ce.networks.legacy_mixed_grn_cci import LegacyMixedGRNCCIBuilder


NETWORK_BUILDERS: Dict[str, Type[NetworkBuilder]] = {
    "legacy_mixed_grn_cci": LegacyMixedGRNCCIBuilder,
    "legacy_inter_cci_only": LegacyInterCCIOnlyBuilder,
    "legacy_inter_additive_grn_cci": LegacyInterAdditiveGRNCCIBuilder,
    "cross_cell_multilayer": CrossCellMultilayerBuilder,
    "expression_only": ExpressionOnlyBuilder,
}


def get_network_builder(network_method: str) -> NetworkBuilder:
    try:
        builder_cls = NETWORK_BUILDERS[network_method]
    except KeyError as exc:
        raise ValueError(f"Unsupported network_method {network_method!r}. Expected one of {sorted(NETWORK_BUILDERS)}.") from exc
    return builder_cls()
