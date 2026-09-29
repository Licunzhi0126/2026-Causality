from __future__ import annotations

import numpy as np

from mignet_ce.downstream.analysis.metrics import entropy, row_normalize


def test_row_normalize_replaces_zero_rows_with_uniform_distribution() -> None:
    normalized = row_normalize(np.asarray([[0.0, 0.0], [1.0, 3.0]]))
    np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.25, 0.75]])


def test_entropy_supports_rowwise_probability_tables() -> None:
    values = entropy(np.asarray([[1.0, 0.0], [0.5, 0.5]]), axis=1)
    np.testing.assert_allclose(values, [0.0, 1.0], atol=1e-10)
