from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mignet_ce.metrics import TemporalMetricsEngine
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
