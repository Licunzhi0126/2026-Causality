from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.visualization.downstream.dynamic_closure.analysis import (
    closure_analysis,
    compose_matrices,
)
from mignet_ce.visualization.downstream.dynamic_closure.config import (
    DynamicClosureConfig,
)


def test_identity_partition_has_zero_closure_floor() -> None:
    transition = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    identity = np.eye(2)
    result = closure_analysis(
        transition,
        identity,
        identity,
        label="identity",
        time_pair="0->1",
        q_direct=transition,
    )
    assert result.summary["best_closure_mean_js"] == pytest.approx(0.0, abs=1e-12)
    assert result.summary["direct_closure_mean_js"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.q_best, transition)


def test_composed_transition_remains_row_stochastic() -> None:
    first = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    second = np.asarray([[0.9, 0.1], [0.4, 0.6]])
    composed = compose_matrices([first, second])
    assert np.allclose(composed.sum(axis=1), 1.0)
    assert np.all(composed >= 0.0)


def test_extended_config_rejects_non_four_time_points(tmp_path) -> None:
    cfg = DynamicClosureConfig(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        time_points=("1", "2", "3"),
    ).normalized()
    with pytest.raises(ValueError, match="four unique"):
        cfg.validate()
