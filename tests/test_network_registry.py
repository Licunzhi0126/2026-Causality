from __future__ import annotations

from mignet_ce.config import NETWORK_METHODS
from mignet_ce.networks.registry import NETWORK_BUILDERS


def test_network_registry_matches_configured_methods() -> None:
    assert NETWORK_METHODS == {"legacy_mixed_grn_cci", "cross_cell_multilayer", "expression_only"}
    assert set(NETWORK_BUILDERS) == NETWORK_METHODS
