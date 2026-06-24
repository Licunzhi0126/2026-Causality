from __future__ import annotations

import pytest

from mignet_ce.graph.builder import _resolve_inter_influence


PAIR = ("Ligand", "Receptor")
PAIR_LOOKUP = {PAIR: (2.0, 0.4)}


def test_product_mode_preserves_legacy_multiplication_and_pair_gate() -> None:
    raw_w, norm_w, influence = _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=PAIR,
        pair_lookup=PAIR_LOOKUP,
        mode="product",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="require_pair",
    )
    assert raw_w == pytest.approx(2.0)
    assert norm_w == pytest.approx(0.4)
    assert influence == pytest.approx(0.2)

    assert _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=("Missing", "Pair"),
        pair_lookup=PAIR_LOOKUP,
        mode="product",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="require_pair",
    ) == (None, None, None)


def test_cci_only_mode_keeps_edge_without_grn_pair() -> None:
    raw_w, norm_w, influence = _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=("Missing", "Pair"),
        pair_lookup=PAIR_LOOKUP,
        mode="cci_only",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="zero_if_missing",
    )
    assert raw_w is None
    assert norm_w is None
    assert influence == pytest.approx(0.5)


def test_additive_mode_uses_weighted_mean_and_configured_missing_pair_policy() -> None:
    raw_w, norm_w, influence = _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=PAIR,
        pair_lookup=PAIR_LOOKUP,
        mode="additive",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="require_pair",
    )
    assert raw_w == pytest.approx(2.0)
    assert norm_w == pytest.approx(0.4)
    assert influence == pytest.approx(0.45)

    assert _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=("Missing", "Pair"),
        pair_lookup=PAIR_LOOKUP,
        mode="additive",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="require_pair",
    ) == (None, None, None)

    raw_w, norm_w, influence = _resolve_inter_influence(
        cci_norm=0.5,
        pair_key=("Missing", "Pair"),
        pair_lookup=PAIR_LOOKUP,
        mode="additive",
        additive_cci_weight=1.0,
        additive_grn_weight=1.0,
        grn_pair_policy="zero_if_missing",
    )
    assert raw_w is None
    assert norm_w == pytest.approx(0.0)
    assert influence == pytest.approx(0.25)


def test_inter_influence_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Unsupported inter_influence_mode"):
        _resolve_inter_influence(
            cci_norm=0.5,
            pair_key=PAIR,
            pair_lookup=PAIR_LOOKUP,
            mode="unknown",
            additive_cci_weight=1.0,
            additive_grn_weight=1.0,
            grn_pair_policy="require_pair",
        )

    with pytest.raises(ValueError, match="positive sum"):
        _resolve_inter_influence(
            cci_norm=0.5,
            pair_key=PAIR,
            pair_lookup=PAIR_LOOKUP,
            mode="additive",
            additive_cci_weight=0.0,
            additive_grn_weight=0.0,
            grn_pair_policy="require_pair",
        )
