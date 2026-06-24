from __future__ import annotations

from mignet_ce.config import NETWORK_METHODS
from mignet_ce.networks.registry import NETWORK_BUILDERS
from scripts.check_mignet_vertical_ablation_inputs import LEGACY_NETWORK_METHODS


def test_network_registry_matches_configured_methods() -> None:
    assert NETWORK_METHODS == {
        "legacy_mixed_grn_cci",
        "legacy_inter_cci_only",
        "legacy_inter_additive_grn_cci",
        "cross_cell_multilayer",
        "expression_only",
    }
    assert set(NETWORK_BUILDERS) == NETWORK_METHODS
    assert NETWORK_BUILDERS["legacy_mixed_grn_cci"].__module__ == "mignet_ce.networks.legacy_mixed_grn_cci"
    assert NETWORK_BUILDERS["legacy_inter_cci_only"].__module__ == "mignet_ce.networks.legacy_inter_cci_only"
    assert (
        NETWORK_BUILDERS["legacy_inter_additive_grn_cci"].__module__
        == "mignet_ce.networks.legacy_inter_additive_grn_cci"
    )
    assert LEGACY_NETWORK_METHODS == {
        "legacy_mixed_grn_cci",
        "legacy_inter_cci_only",
        "legacy_inter_additive_grn_cci",
    }
