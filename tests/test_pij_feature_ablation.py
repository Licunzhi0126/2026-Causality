from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from mignet_ce.pij.ablation.config import AblationConfig
from mignet_ce.pij.ablation.costs import build_controlled_cost
from mignet_ce.pij.ablation.features import AblationFeatureProvider, PairFeatureBlocks
from mignet_ce.pij.ablation.methods import METHOD_BY_ID, METHOD_SPECS
from mignet_ce.pij.ablation.runner import FEATURE_ABLATION_COLUMNS, append_and_write
from mignet_ce.pij.ablation.transitions import transition_from_cost
from mignet_ce.pij.compare._shared.features import CompareFeatureSet
from mignet_ce.pij.compare._shared.ng_kl_ot import (
    build_canonical_ng_cost_numpy,
    build_ng_component_costs_numpy,
    canonical_ng_pij_from_cost_numpy,
)
from scripts import run_pij_feature_ablation as ablation_cli


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260922)
    return (
        rng.normal(size=(5, 4)),
        rng.normal(size=(6, 4)),
        rng.normal(size=(5, 7)),
        rng.normal(size=(6, 7)),
    )


def _blocks(
    cci_source: np.ndarray | None,
    cci_target: np.ndarray | None,
    grn_source: np.ndarray | None,
    grn_target: np.ndarray | None,
    representation: str | None,
) -> PairFeatureBlocks:
    return PairFeatureBlocks(
        cci_source=cci_source,
        cci_target=cci_target,
        grn_source=grn_source,
        grn_target=grn_target,
        cci_representation=representation,
        pairwise_cci_used=True,
        metadata={},
    )


def test_registry_contains_exactly_the_ten_controlled_rows() -> None:
    assert tuple(spec.method_id for spec in METHOD_SPECS) == (
        "N_KL", "N_KL_OT", "L_KL", "L_KL_OT", "G_KL", "G_KL_OT",
        "NG_KL", "NG_KL_OT", "LG_KL", "LG_KL_OT",
    )
    assert set(METHOD_BY_ID) == {spec.method_id for spec in METHOD_SPECS}


def test_n_g_and_full_ng_costs_match_production_rawkl() -> None:
    n_t, n_tp, g_t, g_tp = _features()
    cfg = AblationConfig()
    d_n, d_g, component_metadata = build_ng_component_costs_numpy(
        n_t, n_tp, g_t, g_tp
    )
    n_only = build_controlled_cost(_blocks(n_t, n_tp, None, None, "N"), cfg)
    g_only = build_controlled_cost(_blocks(None, None, g_t, g_tp, None), cfg)
    ng = build_controlled_cost(_blocks(n_t, n_tp, g_t, g_tp, "N"), cfg)
    production, production_metadata = build_canonical_ng_cost_numpy(
        n_t, n_tp, g_t, g_tp
    )

    np.testing.assert_allclose(n_only.cost, d_n)
    np.testing.assert_allclose(g_only.cost, d_g)
    np.testing.assert_allclose(ng.cost, production)
    assert component_metadata["component_normalization"] == "none"
    assert production_metadata["combined_scale_control"] == "none"


def test_ablation_ng_kl_ot_matches_production_transition() -> None:
    n_t, n_tp, g_t, g_tp = _features()
    cfg = AblationConfig()
    cost = build_controlled_cost(_blocks(n_t, n_tp, g_t, g_tp, "N"), cfg).cost
    expected_joint, expected_pij, _ = canonical_ng_pij_from_cost_numpy(
        cost,
        temperature=0.8,
    )
    actual = transition_from_cost(cost, use_ot=True, cfg=cfg)
    np.testing.assert_allclose(actual.raw_matrix, expected_joint)
    np.testing.assert_allclose(actual.pij, expected_pij)


def test_kl_and_kl_ot_share_cost_and_exponential_kernel() -> None:
    n_t, n_tp, g_t, g_tp = _features()
    l_t, l_tp = n_t[:, :3], n_tp[:, :3]
    cfg = AblationConfig()
    variants = (
        _blocks(n_t, n_tp, None, None, "N"),
        _blocks(l_t, l_tp, None, None, "L"),
        _blocks(None, None, g_t, g_tp, None),
        _blocks(n_t, n_tp, g_t, g_tp, "N"),
        _blocks(l_t, l_tp, g_t, g_tp, "L"),
    )
    for blocks in variants:
        kl_cost = build_controlled_cost(blocks, cfg).cost
        ot_cost = build_controlled_cost(blocks, cfg).cost
        np.testing.assert_array_equal(kl_cost, ot_cost)
        kl = transition_from_cost(kl_cost, use_ot=False, cfg=cfg)
        ot = transition_from_cost(ot_cost, use_ot=True, cfg=cfg)
        assert kl.metadata["kernel"] == ot.metadata["kernel_before_ot"]


def test_grn_features_are_the_exact_production_pairwise_block(monkeypatch) -> None:
    n_t, n_tp, g_t, g_tp = _features()
    feature_set = CompareFeatureSet(
        lower_features=[n_t, n_tp],
        upper_features=[n_t, n_tp],
        feature_names=["n"],
        metadata={"grn_block": {"enabled": True}},
        pairwise_lower_grn_features={(0, 1): (g_t, g_tp)},
        pairwise_upper_grn_features={(0, 1): (g_t, g_tp)},
    )
    calls = []

    def fake_builder(_context, _cfg, keys, *, apply_feature_weights):
        calls.append((tuple(keys), apply_feature_weights))
        return feature_set

    monkeypatch.setattr(
        "mignet_ce.pij.ablation.features.build_compare_feature_set",
        fake_builder,
    )
    provider = AblationFeatureProvider(
        SimpleNamespace(network_method="light_cci_grn"),
        SimpleNamespace(),
    )
    blocks = provider.pair_blocks(
        side="lower",
        pair=(0, 1),
        cci_representation=None,
        use_grn=True,
    )
    assert blocks.grn_source is g_t
    assert blocks.grn_target is g_tp
    assert calls == [(('N',), False)]


def test_repository_cli_locks_six_hierarchies_and_three_times(tmp_path) -> None:
    args = ablation_cli.build_argparser().parse_args(
        ["--output-root", str(tmp_path / "out")]
    )
    assert tuple(args.time_points) == ablation_cli.EXPECTED_TIME_POINTS
    assert len(ablation_cli.PAIR_PRESETS[ablation_cli.PAIR_PRESET]) == 6


def test_aggregate_writer_emits_required_schema_and_manifest(tmp_path) -> None:
    row = {
        "method_id": "N_KL", "input_group": "CCI", "feature_method": "NMF",
        "pij_construction": "KL", "organ": "heart", "lower_layer": "spot",
        "upper_layer": "seurat_k150", "hierarchy": "Spot -> K150",
        "time_pair": "11.5->12.5", "EI_lower": 0.1, "EI_upper": 0.2,
        "delta_EI": 0.1, "alpha_cci": 0.01, "beta_n": 0.05,
        "beta_l": 0.05, "beta_g": 0.05, "temperature": 0.8,
        "ot_enabled": False,
    }
    long_path, manifest_path = append_and_write([pd.DataFrame([row])], tmp_path)
    written = pd.read_csv(long_path)
    assert tuple(written.columns) == FEATURE_ABLATION_COLUMNS
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["contract"]["transition_protocol"] == "canonical_ng_rawkl_v2"
