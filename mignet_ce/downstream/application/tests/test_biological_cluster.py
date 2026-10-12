from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mignet_ce.downstream.application.biological_cluster.analysis import contribution_table, transition_table
from mignet_ce.downstream.application.biological_cluster.cluster_assignment import biological_assignment
from mignet_ce.downstream.application.biological_cluster.reporting import write_outputs


def _map(path: Path) -> None:
    pd.DataFrame({"spot_id": ["a", "b", "c"], "cluster": ["2", "1", "2"], "cell_type": ["Beta", "Alpha", "Beta"]}).to_csv(path, index=False)


def test_biological_assignment_preserves_labels_and_order(tmp_path: Path) -> None:
    path = tmp_path / "clusters.csv"
    _map(path)
    assignment, metadata = biological_assignment(path, ["c", "a", "b"])
    assert metadata["cluster"].tolist() == ["1", "2"]
    assert assignment.tolist() == [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]


def test_transition_reporting_joins_cell_types(tmp_path: Path) -> None:
    path = tmp_path / "clusters.csv"
    _map(path)
    _, metadata = biological_assignment(path, ["a", "b", "c"])
    pij = np.array([[0.8, 0.2], [0.1, 0.9]])
    transitions = transition_table(pij, metadata, metadata)
    assert transitions.iloc[0]["source_cell_type"] == "Alpha"
    assert transitions.iloc[0]["target_cell_type"] == "Alpha"
    contributions = contribution_table(np.array([0.1, 0.2]), metadata)
    write_outputs(tmp_path / "out", pij=pij, source_metadata=metadata, target_metadata=metadata, transitions=transitions, contributions=contributions, causal={"delta_EI": 0.1}, closure={"closure_quality": 0.5}, provenance={"replicate": "B1"})
    assert (tmp_path / "out" / "cluster_transition_long.csv").is_file()
