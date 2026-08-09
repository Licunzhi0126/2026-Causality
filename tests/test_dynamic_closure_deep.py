from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.visualization.downstream.dynamic_closure.deep import (
    MappingRecord,
    strong_lumpability_tables,
)


def test_identity_mapping_is_strongly_lumpable() -> None:
    transition = np.asarray([[0.75, 0.25], [0.1, 0.9]], dtype=float)
    identity = np.eye(2)
    record = MappingRecord(
        mapping="identity",
        time_pair="0->1",
        source_time="0",
        target_time="1",
        p_micro=transition,
        source_assignment=identity,
        target_assignment=identity,
        q_best=transition,
        q_direct=transition,
        source_names=["A", "B"],
    )
    summary, states = strong_lumpability_tables([record])
    assert summary.iloc[0]["mean_js"] == pytest.approx(0.0, abs=1e-12)
    assert summary.iloc[0]["mean_q_best_direct_js"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(states["mean_js"], 0.0)
