from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scripts.compare_grn_activity_features import (
    compare_feature_tables,
    compare_pij,
    effective_rank,
    feature_specificity,
)


def test_feature_comparison_aligns_units_and_feature_columns() -> None:
    feature_a = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["u1", "u2"],
        columns=["f1", "f2"],
    )
    feature_b = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["u2", "u1"],
        columns=["f2", "f1"],
    )

    per_unit, summary = compare_feature_tables(feature_a, feature_b)

    assert np.allclose(per_unit["feature_cosine"], 1.0)
    assert summary["common_unit_count"] == 2
    assert summary["common_feature_count"] == 2
    assert summary["mean_same_unit_feature_cosine"] == 1.0


def test_specificity_and_effective_rank_are_finite() -> None:
    values = np.eye(3)
    summary = feature_specificity(values)

    assert summary["mean_pairwise_cosine_distance"] == 1.0
    assert summary["feature_variance_across_units"] > 0
    assert effective_rank(values) == pytest.approx(3.0)


def test_pij_comparison_reports_correlation_and_row_entropy(tmp_path) -> None:
    matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    sp.save_npz(path_a, sp.csr_matrix(matrix))
    sp.save_npz(path_b, sp.csr_matrix(matrix))

    summary = compare_pij(path_a, path_b)

    assert summary["comparable"] is True
    assert summary["pij_correlation"] == pytest.approx(1.0)
    assert summary["mean_row_entropy_a"] == pytest.approx(summary["mean_row_entropy_b"])
