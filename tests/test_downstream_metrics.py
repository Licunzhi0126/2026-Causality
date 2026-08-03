from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.visualization.downstream.metrics import (
    aggregate_transition_by_overlap,
    compose_transitions,
    ei_decomposition,
    mean_row_js,
    row_normalize,
)


def test_row_normalize_replaces_zero_rows_with_uniform_distribution() -> None:
    normalized = row_normalize(np.asarray([[0.0, 0.0], [1.0, 3.0]]))
    np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.25, 0.75]])


def test_ei_decomposition_identity_has_one_bit_of_ei() -> None:
    result = ei_decomposition(np.eye(2))
    assert result["H_effect"] == pytest.approx(1.0)
    assert result["H_noise"] == pytest.approx(0.0, abs=1e-10)
    assert result["EI"] == pytest.approx(1.0)


def test_compose_transitions_and_js_match_direct_product() -> None:
    first = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    second = np.asarray([[0.7, 0.3], [0.4, 0.6]])
    composed = compose_transitions([first, second])
    direct = row_normalize(first) @ row_normalize(second)
    np.testing.assert_allclose(composed, direct)
    assert mean_row_js(composed, direct) == pytest.approx(0.0, abs=1e-10)


def test_overlap_aggregation_preserves_identity_membership() -> None:
    transition = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    counts = np.eye(2)
    aggregated = aggregate_transition_by_overlap(transition, counts, counts)
    np.testing.assert_allclose(aggregated, transition)
