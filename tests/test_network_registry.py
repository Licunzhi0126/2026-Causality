from __future__ import annotations

from mignet_ce.config import NETWORK_METHODS
from mignet_ce.networks.registry import NETWORK_BUILDERS
from scripts.check_mignet_vertical_ablation_inputs import LEGACY_NETWORK_METHODS


def test_network_registry_matches_configured_methods() -> None:
    assert NETWORK_METHODS == {
        "legacy_mixed_grn_cci",
        "legacy_inter_cci_only",
        "legacy_inter_additive_grn_cci",
        "clean_grn_cci_mix",
        "clean_grn_cci_expr_mix",
        "clean_expression_cci_mix",
        "unit_specific_clean_grn_cci_mix",
        "cross_cell_multilayer",
        "expression_only",
        "light_cci",
        "sparse_cci_mass99",
        "sparse_cci_mass95",
    }
    assert set(NETWORK_BUILDERS) == NETWORK_METHODS
    assert NETWORK_BUILDERS["legacy_mixed_grn_cci"].__module__ == "mignet_ce.networks.legacy_mixed_grn_cci"
    assert NETWORK_BUILDERS["legacy_inter_cci_only"].__module__ == "mignet_ce.networks.legacy_inter_cci_only"
    assert NETWORK_BUILDERS["clean_grn_cci_mix"].__module__ == "mignet_ce.networks.clean_grn_cci_mix"
    assert (
        NETWORK_BUILDERS["clean_grn_cci_expr_mix"].__module__
        == "mignet_ce.networks.clean_grn_cci_expr_mix"
    )
    assert (
        NETWORK_BUILDERS["clean_expression_cci_mix"].__module__
        == "mignet_ce.networks.clean_expression_cci_mix"
    )
    assert (
        NETWORK_BUILDERS["unit_specific_clean_grn_cci_mix"].__module__
        == "mignet_ce.networks.unit_specific_clean_grn_cci_mix"
    )
    assert (
        NETWORK_BUILDERS["legacy_inter_additive_grn_cci"].__module__
        == "mignet_ce.networks.legacy_inter_additive_grn_cci"
    )
    assert NETWORK_BUILDERS["light_cci"].__module__ == "mignet_ce.networks.light_cci"
    assert NETWORK_BUILDERS["sparse_cci_mass99"].__module__ == "mignet_ce.networks.sparse_cci_mass99"
    assert NETWORK_BUILDERS["sparse_cci_mass95"].__module__ == "mignet_ce.networks.sparse_cci_mass95"
    assert LEGACY_NETWORK_METHODS == {
        "legacy_mixed_grn_cci",
        "legacy_inter_cci_only",
        "legacy_inter_additive_grn_cci",
    }
