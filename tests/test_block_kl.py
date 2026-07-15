from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.pij.compare.block_kl import build_block_kl_cost
from mignet_ce.pij.compare.kl import pairwise_feature_kl


def test_block_kl_combines_independently_normalized_n_and_g_costs() -> None:
    n_source = np.array([[2.0, 0.0], [0.0, 2.0]])
    n_target = np.array([[1.5, 0.5], [0.5, 1.5], [1.0, 1.0]])
    g_source = np.array([[1.0, 3.0], [3.0, 1.0]])
    g_target = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])

    cost, metadata = build_block_kl_cost(
        n_source,
        n_target,
        g_source,
        g_target,
        weight_n=0.5,
        weight_g=0.5,
        beta_n=0.5,
        beta_g=0.5,
    )

    assert cost.shape == (2, 3)
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0.0)
    assert metadata["mode"] == "weighted_independently_normalized_block_kl"


def test_block_kl_is_invariant_to_repeating_every_g_feature_column() -> None:
    rng = np.random.default_rng(7)
    n_source = rng.normal(size=(3, 4))
    n_target = rng.normal(size=(5, 4))
    g_source = rng.normal(size=(3, 2))
    g_target = rng.normal(size=(5, 2))

    original, _ = build_block_kl_cost(n_source, n_target, g_source, g_target)
    repeated, _ = build_block_kl_cost(
        n_source,
        n_target,
        np.tile(g_source, (1, 3)),
        np.tile(g_target, (1, 3)),
    )

    assert np.allclose(repeated, original)


def test_zero_grn_weight_exactly_recovers_original_n_kl_cost() -> None:
    rng = np.random.default_rng(11)
    n_source = rng.normal(size=(3, 4))
    n_target = rng.normal(size=(5, 4))
    g_source = rng.normal(size=(3, 7))
    g_target = rng.normal(size=(5, 7))

    block_cost, metadata = build_block_kl_cost(
        n_source,
        n_target,
        g_source,
        g_target,
        weight_n=1.0,
        weight_g=0.0,
        beta_n=0.2,
        beta_g=0.2,
    )
    original = pairwise_feature_kl(n_source, n_target, beta=0.2)

    assert np.array_equal(block_cost, original)
    assert metadata["mode"] == "n_only_exact_fallback"


def test_block_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1"):
        build_block_kl_cost(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), weight_n=0.7, weight_g=0.7)
