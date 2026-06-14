from __future__ import annotations

from mignet_ce.config import NETWORK_METHODS
from mignet_ce.networks.registry import NETWORK_BUILDERS


def test_network_registry_contains_only_two_methods() -> None:
    assert NETWORK_METHODS == {"legacy_mixed_grn_cci", "cross_cell_multilayer"}
    assert set(NETWORK_BUILDERS) == {"legacy_mixed_grn_cci", "cross_cell_multilayer"}
