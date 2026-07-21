from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.pij.compare._shared.cosine import pairwise_cosine_distance
from mignet_ce.pij.compare._shared.distances import (
    pairwise_euclidean_distance,
    pairwise_scalar_absolute_distance,
    pairwise_vector_distance,
    robust_normalize_cost,
)


def test_cosine_wrapper_matches_existing_implementation() -> None:
    source = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    target = np.array([[0.0, 1.0], [2.0, 0.0]])
    np.testing.assert_array_equal(
        pairwise_vector_distance(source, target, "cosine"),
        pairwise_cosine_distance(source, target),
    )


def test_pairwise_euclidean_matches_hand_calculation_and_blocks() -> None:
    source = np.array([[0.0, 0.0], [3.0, 4.0]])
    target = np.array([[0.0, 4.0], [3.0, 0.0]])
    expected = np.array([[4.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(pairwise_euclidean_distance(source, target, block_size=1), expected)


def test_cosine_and_euclidean_distinguish_vector_magnitude() -> None:
    source = np.array([[1.0, 0.0]])
    target = np.array([[3.0, 0.0]])
    assert pairwise_vector_distance(source, target, "cosine")[0, 0] == pytest.approx(0.0)
    assert pairwise_vector_distance(source, target, "euclidean")[0, 0] == pytest.approx(2.0)


def test_scalar_absolute_distance_and_column_validation() -> None:
    source = np.array([[1.0], [4.0]])
    target = np.array([[2.0], [7.0]])
    np.testing.assert_array_equal(
        pairwise_scalar_absolute_distance(source, target),
        np.array([[1.0, 6.0], [2.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="exactly one feature column"):
        pairwise_scalar_absolute_distance(np.ones((2, 2)), target)


def test_empty_distance_preserves_shape() -> None:
    result = pairwise_euclidean_distance(np.empty((0, 3)), np.ones((2, 3)))
    assert result.shape == (0, 2)


def test_robust_quantile_normalization_and_metadata() -> None:
    cost = np.arange(20, dtype=float).reshape(4, 5)
    normalized, metadata = robust_normalize_cost(cost.copy())
    q_low, q_high = np.percentile(cost, [5.0, 95.0])
    expected = np.clip((cost - q_low) / (q_high - q_low), 0.0, 1.0)
    np.testing.assert_allclose(normalized, expected)
    assert metadata["normalization_mode"] == "quantile_5_95"
    assert metadata["q_low"] == pytest.approx(q_low)
    assert metadata["q_high"] == pytest.approx(q_high)


def test_robust_normalization_minmax_fallback_and_degenerate() -> None:
    mostly_zero = np.concatenate([np.zeros(100), np.ones(1)]).reshape(1, -1)
    normalized, metadata = robust_normalize_cost(mostly_zero.copy())
    assert metadata["normalization_mode"] == "minmax_fallback"
    assert normalized[0, -1] == pytest.approx(1.0)

    constant, metadata = robust_normalize_cost(np.full((2, 3), 7.0))
    np.testing.assert_array_equal(constant, np.zeros((2, 3)))
    assert metadata["normalization_mode"] == "all_zero_degenerate"


def test_robust_normalization_empty_and_nonfinite_diagnostics() -> None:
    empty, empty_metadata = robust_normalize_cost(np.empty((0, 3)))
    assert empty.shape == (0, 3)
    assert empty_metadata["normalization_mode"] == "empty"

    normalized, metadata = robust_normalize_cost(np.array([[0.0, np.nan, np.inf, -np.inf, 2.0]]))
    assert metadata["nonfinite_count"] == 3
    assert np.isfinite(normalized).all()
    assert metadata["normalized_summary"]["nonfinite_count"] == 0
