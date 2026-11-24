"""Isolated mathematical checks; these do not claim full training success."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import torch
import pytest
from mignet_ce.downstream.perturbation.formal_macro_adapter import (
    build_direct_state_macro_pij_builder, make_projector, paired_zscore, validate_stage_coordinates)
from mignet_ce.representations.coarse_input import MacroPijInputs
from mignet_ce.pij.compare._shared import ng_kl_ot

ROOT = Path(__file__).resolve().parents[4]


def functions(path, names, namespace):
    # Execute only the actual read-only mathematical functions, bypassing package
    # initialization. No fake modules/constants or production monkeypatches.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace


def stage(seed=1):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(units=["a", "b", "c"], grn_genes=["A", "B", "C"],
        expression_grn=rng.random((3, 3)).astype("float32"),
        grn_adjacency=sp.csr_matrix(rng.random((3, 3)).astype("float32")),
        projection_reg=rng.normal(size=(3, 4)).astype("float32"),
        projection_tar=rng.normal(size=(3, 4)).astype("float32"), g_raw=np.zeros((3, 4)))


def formal_math():
    ns = {"torch": torch, "np": np}
    ns.update({name: getattr(ng_kl_ot, name) for name in (
        "canonical_ng_pij_torch", "CANONICAL_ALPHA_CCI", "CANONICAL_FEATURE_BETA_G",
        "CANONICAL_FEATURE_BETA_N", "CANONICAL_TEMPERATURE")})
    return functions(ROOT / "wyt_deltaei_coarse_grain/complete_combined.py",
        {"_pairwise_zscore_torch", "build_macro_pij_builder"}, ns)["build_macro_pij_builder"]


def test_zero_residual_matches_readonly_formal_math():
    source, target = stage(), stage(2)
    n = torch.tensor([[0.2, 0.9], [0.7, 0.1]])
    inputs = MacroPijInputs(n, n, None, None,
        {"N": n, "X": torch.tensor(source.expression_grn[:2])},
        {"N": n.flip(0), "X": torch.tensor(target.expression_grn[:2]), "DG_TP": torch.zeros(2, 4)})
    expected = formal_math()(source, target)(inputs)
    actual = build_direct_state_macro_pij_builder(source, target, delta_g_raw_tp=np.zeros((3, 4)))(inputs)
    error = (expected - actual).abs().max().item()
    print(f"isolated_zero_residual_max_error={error}")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_t1_adjacency_changes_raw_macro_g():
    source = stage()
    changed = SimpleNamespace(**{**vars(source), "grn_adjacency": source.grn_adjacency * 2})
    x = torch.tensor(source.expression_grn[:2])
    assert not torch.allclose(make_projector()(x, source, "t"), make_projector()(x, changed, "t"))
    np.testing.assert_array_equal(source.expression_grn, changed.expression_grn)


def test_real_trainer_pooling_changes_residual_and_keeps_gradients():
    ns = functions(ROOT / "wyt_deltaei_coarse_grain/macro_builder.py",
        {"pool_feature_blocks", "pool_to_macro"}, {"torch": torch, "EPS": 1e-12})
    source, target = stage(), stage(2)
    residual = torch.tensor([[1., -2., 0., 3.], [2., 1., 4., 0.], [-1., 0., 2., 1.]])
    logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.]], requires_grad=True)
    assignment = logits.softmax(1)
    blocks = {"N": torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
              "X": torch.tensor(target.expression_grn), "DG_TP": residual}
    pooled = ns["pool_feature_blocks"](blocks, assignment)
    other = ns["pool_feature_blocks"](blocks, assignment.flip(0))
    assert not torch.allclose(pooled["DG_TP"], other["DG_TP"])
    inputs = MacroPijInputs(None, None, None, None,
        {"N": pooled["N"], "X": torch.tensor(source.expression_grn[:2])}, pooled)
    value = build_direct_state_macro_pij_builder(source, target, delta_g_raw_tp=residual)(inputs)
    (value * torch.tensor([[1., 2.], [3., 5.]])).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("field", ["grn_genes", "projection_reg", "units"])
def test_coordinates_fail_fast(field):
    reference = stage()
    changed = SimpleNamespace(**vars(reference))
    setattr(changed, field, getattr(reference, field)[::-1])
    with pytest.raises(ValueError, match="coordinate mismatch"):
        validate_stage_coordinates(reference, changed)


def test_nonlegacy_residual_fails_explicitly():
    with pytest.raises(NotImplementedError):
        build_direct_state_macro_pij_builder(stage(), stage(2), feature_method="enhanced")


def test_formal_frontend_integration_availability():
    try:
        from mignet_ce.coarse_frontends.formal_two_stage import prepare
    except ImportError as error:
        pytest.fail(f"UPSTREAM_IMPORT_BLOCKER: {error}")
    assert callable(prepare)


def test_cache_rejects_previous_schema_without_shared_trainer(monkeypatch, tmp_path):
    import json
    from dataclasses import dataclass
    from mignet_ce.downstream.perturbation import optimal_bridge as bridge
    @dataclass
    class Config:
        k: int
        out_dir: Path
        device: str
        local_graph_mode: str = "all_features"
    monkeypatch.setattr(bridge, "formal_training_config", lambda **kw:
        Config(kw["k"], kw["training"], kw["device"]))
    training = tmp_path / "training"
    training.mkdir()
    for name in bridge.REQUIRED_OUTPUTS:
        (training / name).write_text("x", encoding="utf-8")
    record = {"method": "formal_two_stage", "loss_protocol": "formal_two_stage_main",
        "k": 2, "device": "cpu", "frontend_options": {}, "local_graph_mode": "all_features",
        **bridge.training_config_record(Config(2, training, "cpu"))}
    path = tmp_path / "perturbation_provenance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="perturbation_macro_schema"):
        bridge.validate_cached_training(tmp_path, k=2, device="cpu")
    record["perturbation_macro_schema"] = bridge.PERTURBATION_MACRO_SCHEMA
    path.write_text(json.dumps(record), encoding="utf-8")
    bridge.validate_cached_training(tmp_path, k=2, device="cpu")
    with pytest.raises(ValueError, match="frontend_options"):
        bridge.validate_cached_training(tmp_path, k=2, device="cpu", frontend_options={"alpha_cci": 0.2})


@pytest.mark.parametrize("condition", ["zero", "partial", "all_edges_zero"])
def test_fixed_graph_coordinates_and_actual_training_features(monkeypatch, tmp_path, condition):
    from mignet_ce.downstream.perturbation import direct_state_bridge, state_bridge
    from mignet_ce.downstream.perturbation.grn_io import read_grn
    from mignet_ce.downstream.perturbation.materialize import materialize_source_grn
    from mignet_ce.downstream.perturbation.mirror_perturbation import apply_mirrored_weight_perturbation
    from wyt_deltaei_coarse_grain import complete_combined

    keys = ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "index_t", "index_tp", "grn_t", "grn_tp")
    paths = {key: tmp_path / key for key in keys}
    for key in ("grn_t", "grn_tp"):
        paths[key].write_text("regulator,target,weight\nA,B,2\nC,D,3\nA,E,1\nD,E,0\n")
    baseline = read_grn(paths["grn_t"])
    selected = {"zero": (), "partial": ((0, 1), (2, 3)),
                "all_edges_zero": ((0, 1), (2, 3), (0, 4))}[condition]
    _, perturbed, _ = apply_mirrored_weight_perturbation(baseline, selected_edges=selected, mirror_fraction=1.)
    materialized = materialize_source_grn(baseline=baseline, perturbed=perturbed,
                                          baseline_inputs=paths, output_dir=tmp_path, pair_id="P1")
    assert materialized.graph.nodes == ("A", "B", "C", "D", "E")
    np.testing.assert_array_equal(materialized.graph.adjacency, perturbed.adjacency)

    source = ["s0", "s1", "s2"]
    target = ["t0", "t1"]
    rng = np.random.default_rng(11)
    counts = sp.csr_matrix(rng.uniform(1, 5, size=(5, 5)))
    monkeypatch.setattr(complete_combined, "read_h5ad_counts", lambda path: (
        counts, source + target, baseline.nodes, "fixture"))
    pair_data = SimpleNamespace(
        unit_ids_t=source, unit_ids_tp=target, cci_t=sp.csr_matrix(np.ones((3, 3)) - np.eye(3)),
        cci_tp=sp.csr_matrix(np.ones((2, 2)) - np.eye(2)), coords_t=None, coords_tp=None,
        index_t=paths["index_t"], index_tp=paths["index_tp"],
    )
    monkeypatch.setattr(state_bridge, "load_spot_pair", lambda request: pair_data)
    pij = np.array([[1., 0.], [0.25, 0.75], [0., 1.]])
    baseline_pij = tmp_path / "PIJ_micro_train.npy"
    np.save(baseline_pij, pij)
    for suffix, units in (("t", source), ("tp", target)):
        (tmp_path / f"assignments_{suffix}.csv").write_text("unit_id\n" + "\n".join(units))
        (tmp_path / f"maturity_{suffix}.csv").write_text(
            "unit_id,pseudotime\n" + "\n".join(f"{unit},{i}" for i, unit in enumerate(units)))
    options = {"grn_state_dim": 4, "grn_topk_targets": 1, "nmf_components": 2,
               "nmf_max_iter": 10, "mid_dim": 2}
    states = state_bridge.build_propagated_grn_states(
        baseline_inputs=paths, perturbed_grn_t=materialized.perturbed_grn,
        baseline_spot_pij=baseline_pij, frontend_options=options)
    assert states.stage_t.grn_genes == list(baseline.nodes)
    assert states.stage_t.grn_adjacency.shape == (5, 5)
    assert states.g_raw_t_perturbed.shape == (3, 4)
    assert states.g_raw_tp_perturbed.shape == (2, 4)
    np.testing.assert_array_equal(states.delta_g_raw_tp, pij.T @ states.delta_g_raw_t)
    if condition == "partial":
        # Top-k switches A's target from B to E. The zeroed C→D leaves both endpoints isolated.
        assert states.stage_t.grn_adjacency.nnz == 1
        assert states.stage_t.grn_adjacency[0, 4] == 1.
        assert np.count_nonzero(states.delta_g_raw_tp)
    if condition == "all_edges_zero":
        assert states.stage_t.grn_adjacency.nnz == 0
        np.testing.assert_array_equal(states.g_raw_t_perturbed, np.zeros((3, 4)))
        assert np.count_nonzero(states.delta_g_raw_tp)

    request = state_bridge.make_request(paths, options)
    ordinary_baseline = state_bridge.prepare_stage(request, pair_data, "t")
    np.testing.assert_allclose(states.g_raw_t_baseline, ordinary_baseline.g_raw, atol=1e-6, rtol=1e-5)
    expected = complete_combined.prepare_complete_pair(
        ordinary_baseline, states.stage_tp_baseline, nmf_components=2, nmf_max_iter=10,
        seed=request.seed, mid_dim=2)
    monkeypatch.setattr(state_bridge, "load_spot_pair", lambda request: pytest.fail("reloaded spots"))
    monkeypatch.setattr(complete_combined, "read_h5ad_counts", lambda path: pytest.fail("reloaded expression"))
    actual = direct_state_bridge.prepare_direct_state_two_stage(
        baseline_inputs=paths, maturity_t=tmp_path / "maturity_t.csv", maturity_tp=tmp_path / "maturity_tp.csv",
        states=states, frontend_options=options)
    for name, baseline_values in (
        ("encoder_features_t", expected.encoder_t), ("encoder_features_tp", expected.encoder_tp),
        ("micro_features_t", expected.micro_features_t), ("micro_features_tp", expected.micro_features_tp),
        ("micro_pij", expected.micro_pij),
    ):
        if condition == "zero":
            np.testing.assert_allclose(getattr(actual, name), baseline_values, atol=2e-6, rtol=1e-5)
        else:
            assert not np.allclose(getattr(actual, name), baseline_values)
    if condition == "zero":
        np.testing.assert_array_equal(states.delta_g_raw_tp, np.zeros((2, 4)))
        np.testing.assert_array_equal(states.g_raw_tp_perturbed, states.g_raw_tp_baseline)
    np.testing.assert_array_equal(actual.feature_blocks_tp["DG_TP"], states.delta_g_raw_tp)
    np.testing.assert_array_equal(actual.feature_blocks_t["N"], expected.n_t)
    np.testing.assert_array_equal(actual.feature_blocks_tp["X"], states.stage_tp_baseline.expression_grn)


def test_macro_builder_adds_pooled_target_residual_once(monkeypatch):
    from mignet_ce.downstream.perturbation import formal_macro_adapter as adapter
    source, target = stage(), stage(2)
    n = torch.tensor([[0.2, 0.9], [0.7, 0.1]])
    x_t, x_tp = torch.tensor(source.expression_grn[:2]), torch.tensor(target.expression_grn[:2])
    residual = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    project = adapter.make_projector()
    raw_t = project(x_t, source, "t")
    raw_tp = project(x_tp, target, "tp")
    inputs = MacroPijInputs(n, n, None, None, {"N": n, "X": x_t},
                            {"N": n.flip(0), "X": x_tp, "DG_TP": residual})
    captured = []
    monkeypatch.setattr(adapter, "paired_zscore", lambda t, tp: (captured.append((t, tp)) or (t, tp)))
    adapter.build_direct_state_macro_pij_builder(source, target, delta_g_raw_tp=residual)(inputs)
    assert len(captured) == 2
    torch.testing.assert_close(captured[1][0], raw_t)
    torch.testing.assert_close(captured[1][1], raw_tp + residual)
