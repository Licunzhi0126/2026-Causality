from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.pij.feature_versions.distances import (
    normalize_composition,
    pairwise_cosine,
    pairwise_hellinger,
    pairwise_js,
    pairwise_kl,
    pairwise_scalar_robust,
)
from mignet_ce.pij.feature_versions.fusion import fuse_cost_blocks, robust_normalize_cost


def test_js_is_symmetric_bounded_nonnegative_and_zero_on_identity() -> None:
    values = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 0.0, 1.0]])
    distance = pairwise_js(values, values, pseudocount=1e-8, block_size=1)
    np.testing.assert_allclose(distance, distance.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(distance), 0.0, rtol=0.0, atol=1e-12)
    assert np.all(distance >= 0.0)
    assert np.max(distance) <= np.log(2.0) + 1e-12
    assert np.all(np.isfinite(distance))


def test_hellinger_is_symmetric_bounded_and_zero_on_identity() -> None:
    values = np.array([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    distance = pairwise_hellinger(values, values, pseudocount=1e-8, block_size=1)
    np.testing.assert_allclose(distance, distance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-12)
    assert np.min(distance) >= 0.0
    assert np.max(distance) <= 1.0 + 1e-12


def test_composition_pseudocount_stabilizes_zero_rows() -> None:
    normalized = normalize_composition(np.zeros((2, 4)), pseudocount=1e-8)
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(normalized, np.full((2, 4), 0.25))


def test_new_kl_matches_direct_definition() -> None:
    source = np.array([[2.0, 0.0], [0.5, 1.5]])
    target = np.array([[1.0, 1.0], [0.0, 2.0]])
    beta = 0.4
    actual = pairwise_kl(source, target, beta=beta, block_size=1)
    source_prob = np.exp(source / beta - np.max(source / beta, axis=1, keepdims=True))
    source_prob /= source_prob.sum(axis=1, keepdims=True)
    target_prob = np.exp(target / beta - np.max(target / beta, axis=1, keepdims=True))
    target_prob /= target_prob.sum(axis=1, keepdims=True)
    expected = np.array(
        [[np.sum(p * (np.log(p) - np.log(q))) for q in target_prob] for p in source_prob]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_cosine_zero_vector_rule_is_deterministic() -> None:
    distance = pairwise_cosine(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(distance, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_scalar_robust_distance_uses_joint_iqr() -> None:
    source = np.array([[0.0], [2.0]])
    target = np.array([[1.0], [3.0]])
    q25, q75 = np.percentile([0.0, 2.0, 1.0, 3.0], [25.0, 75.0])
    expected = np.abs(source[:, 0, None] - target[None, :, 0]) / (q75 - q25 + 1e-12)
    np.testing.assert_allclose(pairwise_scalar_robust(source, target), expected)


def test_fusion_normalizes_blocks_without_weight_redistribution() -> None:
    costs = {
        "varying": np.arange(9, dtype=float).reshape(3, 3),
        "constant": np.ones((3, 3), dtype=float),
    }
    fused, metadata, normalized = fuse_cost_blocks(costs, {"varying": 0.6, "constant": 0.4})
    np.testing.assert_array_equal(normalized["constant"], np.zeros((3, 3)))
    np.testing.assert_allclose(fused, 0.6 * normalized["varying"])
    assert metadata["block_diagnostics"]["constant"]["constant_block"] is True
    assert metadata["weight_redistribution"] is False
    assert np.all(np.isfinite(fused)) and np.all(fused >= 0.0)


def test_fusion_rejects_invalid_weights_and_nonfinite_cost() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        fuse_cost_blocks({"a": np.ones((2, 2))}, {"a": 0.5})
    with pytest.raises(ValueError, match="non-finite"):
        robust_normalize_cost(np.array([[np.nan]]))
