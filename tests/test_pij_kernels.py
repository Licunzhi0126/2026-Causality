from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import numpy as np
import pytest

from mignet_ce.metrics import TemporalMetricsEngine, pairwise_joint_nmf, pairwise_shared_core_directed_nmf
from mignet_ce.transition.cosine import build_cosine_transition_kernel
from mignet_ce.transition.sinkhorn_3dot import build_3dot_transition_kernel
from mignet_ce.transition.slat_adapter import build_slat_transition_kernel


def assert_row_stochastic(p: np.ndarray) -> None:
    assert p.ndim == 2
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_cosine_transition_kernel_is_row_stochastic() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    p = build_cosine_transition_kernel(source, target)
    assert p.shape == (2, 3)
    assert_row_stochastic(p)


def test_3dot_transition_kernel_clips_large_topk() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(4, 3))
    target = rng.normal(size=(5, 3))
    source_coords = rng.normal(size=(4, 2))
    target_coords = rng.normal(size=(5, 2))
    p = build_3dot_transition_kernel(
        source,
        target,
        source_coords,
        target_coords,
        sim_k=100,
        dist_k=100,
        max_iter=5,
    )
    assert p.shape == (4, 5)
    assert_row_stochastic(p)


def test_3dot_transition_kernel_handles_zero_features() -> None:
    source = np.zeros((3, 2))
    target = np.zeros((2, 2))
    source_coords = np.zeros((3, 2))
    target_coords = np.ones((2, 2))
    p = build_3dot_transition_kernel(source, target, source_coords, target_coords, max_iter=0)
    assert p.shape == (3, 2)
    assert_row_stochastic(p)


def test_metrics_accept_precomputed_pij() -> None:
    engine = TemporalMetricsEngine()
    lower = [np.eye(3), np.eye(3)]
    upper = [np.eye(3), np.eye(3)]
    p = np.eye(3)
    metrics = engine.calculate_metrics_for_pairs(
        lower_feat=lower,
        upper_feat=upper,
        time_points=["a", "b"],
        pairs=[(0, 1)],
        organ="organ",
        lower_layer="lower",
        upper_layer="upper",
        pij_method="test",
        precomputed_p_lower={(0, 1): p},
        precomputed_p_upper={(0, 1): p},
    )
    assert metrics.loc[0, "pij_method"] == "test"
    assert metrics.loc[0, "EI_gain"] == pytest.approx(0.0)


def test_pairwise_joint_nmf_shapes() -> None:
    source = np.array([[1.0, 0.2, 0.0], [0.1, 0.8, 0.3], [0.0, 0.4, 1.0]])
    target = np.array([[0.8, 0.1, 0.2], [0.2, 1.0, 0.5]])

    w_source, w_target, h_matrix = pairwise_joint_nmf(source, target, n_components=2, max_iter=4, seed=7)

    assert w_source.shape == (3, 2)
    assert w_target.shape == (2, 2)
    assert h_matrix.shape == (2, 3)
    assert np.all(w_source >= 0)
    assert np.all(w_target >= 0)
    assert np.all(h_matrix >= 0)


def test_pairwise_joint_nmf_reports_column_mismatch() -> None:
    with pytest.raises(ValueError, match="identical column counts"):
        pairwise_joint_nmf(np.ones((3, 3)), np.ones((2, 4)), n_components=2, max_iter=1)


def test_pairwise_shared_core_directed_nmf_shapes_with_different_node_counts() -> None:
    source = np.eye(8, dtype=float) + 0.1
    target = np.eye(5, dtype=float) + 0.2

    u_source, v_source, u_target, v_target, core = pairwise_shared_core_directed_nmf(
        source,
        target,
        n_components=3,
        max_iter=4,
        seed=11,
    )

    assert u_source.shape == (8, 3)
    assert v_source.shape == (8, 3)
    assert u_target.shape == (5, 3)
    assert v_target.shape == (5, 3)
    assert core.shape == (3, 3)
    assert np.hstack([u_source, v_source]).shape == (8, 6)
    assert np.hstack([u_target, v_target]).shape == (5, 6)
    assert np.all(core >= 0)


def test_slat_adapter_uses_slat_edges_features_return(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    torch_mod = module("torch")
    torch_mod.manual_seed = lambda seed: calls.setdefault("manual_seed", seed)
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: calls.setdefault("cuda_seed", seed),
    )

    scslat_mod = module("scSLAT")
    model_mod = module("scSLAT.model")
    loaddata_mod = module("scSLAT.model.loaddata")
    preprocess_mod = module("scSLAT.model.preprocess")
    utils_mod = module("scSLAT.model.utils")
    scslat_mod.model = model_mod
    model_mod.loaddata = loaddata_mod
    model_mod.preprocess = preprocess_mod
    model_mod.utils = utils_mod

    def fake_cal_spatial_net(adata, **kwargs) -> None:
        adata.uns["cal_spatial_net_kwargs"] = kwargs

    def fake_load_anndatas(adatas, **kwargs):
        calls["load_kwargs"] = kwargs
        assert all("cal_spatial_net_kwargs" in adata.uns for adata in adatas)
        return ["source_edges", "target_edges"], ["source_features", "target_features"]

    class FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self.values = np.asarray(values, dtype=float)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    def fake_run_slat(features, edges, **kwargs):
        calls["features"] = features
        calls["edges"] = edges
        calls["run_kwargs"] = kwargs
        return (
            FakeTensor(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
            FakeTensor(np.array([[1.0, 0.0], [0.0, 1.0]])),
            0.25,
        )

    preprocess_mod.Cal_Spatial_Net = fake_cal_spatial_net
    loaddata_mod.load_anndatas = fake_load_anndatas
    utils_mod.run_SLAT = fake_run_slat

    p, metadata = build_slat_transition_kernel(
        np.ones((3, 2)),
        np.ones((2, 2)),
        np.zeros((3, 2)),
        np.zeros((2, 2)),
        epochs=1,
    )

    assert p.shape == (3, 2)
    assert_row_stochastic(p)
    assert calls["features"] == ["source_features", "target_features"]
    assert calls["edges"] == ["source_edges", "target_edges"]
    assert calls["load_kwargs"] == {"feature": "raw", "self_loop": True, "check_order": False}
    assert metadata["run_time_seconds"] == pytest.approx(0.25)


def test_slat_requires_real_dependencies() -> None:
    has_slat = all(importlib.util.find_spec(name) for name in ["torch", "scSLAT"])
    if has_slat:
        pytest.skip("Dependency presence is environment-specific; covered by vertical smoke runs.")
    with pytest.raises(ImportError, match="requires the real scSLAT runtime"):
        build_slat_transition_kernel(
            np.ones((3, 2)),
            np.ones((3, 2)),
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            epochs=1,
        )
