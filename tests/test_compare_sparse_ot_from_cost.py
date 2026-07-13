from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.pij.compare.sparse_ot import run_sparse_semi_relaxed_ot_from_cost


def _run(cost: np.ndarray):
    return run_sparse_semi_relaxed_ot_from_cost(
        cost,
        epsilon=0.1,
        gamma=1.0,
        max_iter=50,
        source_k=2,
        target_k=2,
        raw_cost_column="raw_fused_pre_cost",
        cost_source="test_fused_cost",
    )


def test_empty_cost_preserves_custom_column_and_convergence_metadata() -> None:
    result = _run(np.empty((0, 2)))
    assert list(result.candidate_edges.columns) == [
        "source_index",
        "target_index",
        "raw_fused_pre_cost",
        "normalized_cost",
    ]
    assert result.convergence["raw_cost_column"] == "raw_fused_pre_cost"
    assert result.convergence["cost_source"] == "test_fused_cost"


def test_sparse_ot_from_cost_is_nonnegative_finite_and_row_stochastic() -> None:
    result = _run(np.array([[0.0, 0.4, 0.8], [0.7, 0.2, 0.5]]))
    pij = result.pij_row_normalized_sparse.toarray()
    assert np.isfinite(pij).all()
    assert np.all(pij >= 0.0)
    nonempty = np.asarray(result.pij_row_normalized_sparse.sum(axis=1)).ravel() > 0
    np.testing.assert_allclose(pij.sum(axis=1)[nonempty], 1.0)
    assert result.convergence["cost_source"] == "test_fused_cost"
    assert result.convergence["raw_cost_column"] == "raw_fused_pre_cost"


def test_sparse_ot_rejects_negative_and_nonfinite_candidate_costs() -> None:
    with pytest.raises(ValueError, match="negative finite"):
        _run(np.array([[0.0, -0.1]]))
    with pytest.raises(ValueError, match="candidate raw costs"):
        _run(np.array([[np.nan, np.inf]]))
