from __future__ import annotations

import numpy as np

from mignet_ce.downstream.application.optimal_cg.cluster_correspondence import build_macro_correspondence
from mignet_ce.downstream.application.optimal_cg.local_pair_analysis import run_local_pair_analysis
from mignet_ce.downstream.application.optimal_cg.local_pair_builder import build_local_pair


def test_macro_correspondence_is_source_row_argmax_with_target_indegree() -> None:
    matrix, records = build_macro_correspondence(np.array([[1.0, 3.0], [2.0, 6.0], [8.0, 1.0]]))
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert [(item.source_cluster, item.target_cluster) for item in records] == [(0, 1), (1, 1), (2, 0)]
    assert [item.target_in_degree for item in records] == [2, 2, 1]


def test_local_pair_slices_both_axes_and_conditions_rows() -> None:
    micro = np.array([[0.1, 0.6, 0.3], [0.0, 0.0, 1.0], [0.2, 0.2, 0.6]])
    s_t = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    s_tp = np.array([[0, 1], [1, 0], [1, 0]], dtype=float)
    pair = build_local_pair(pij_micro=micro, s_t=s_t, s_tp=s_tp, source_cluster=0, target_cluster=0, min_local_spots=1)
    assert pair.source_indices_original.tolist() == [0, 1]
    assert pair.target_indices.tolist() == [1, 2]
    assert np.allclose(pair.pij_raw, [[0.6, 0.3], [0.0, 1.0]])
    assert np.allclose(pair.pij_conditional, [[2 / 3, 1 / 3], [0.0, 1.0]])
    assert pair.status == "ready"


def test_local_pair_drops_low_mass_source_rows() -> None:
    micro = np.array([[1.0, 0.0], [0.0, 1.0]])
    assignments = np.array([[1, 0], [1, 0]], dtype=float)
    targets = np.array([[1, 0], [0, 1]], dtype=float)
    pair = build_local_pair(pij_micro=micro, s_t=assignments, s_tp=targets, source_cluster=0, target_cluster=1, min_local_spots=1)
    assert pair.n_source_dropped == 1
    assert pair.source_indices_used.tolist() == [1]
    assert np.allclose(pair.pij_conditional, [[1.0]])


def test_orchestrator_continues_after_one_pair_failure(tmp_path, monkeypatch) -> None:
    training = tmp_path / "training"
    training.mkdir()
    np.save(training / "PIJ_micro_train.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.save(training / "PIJ_macro_train.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.save(training / "S_t.npy", np.eye(2))
    np.save(training / "S_tp.npy", np.eye(2))

    def fake_train(**kwargs):
        if kwargs["pair"].source_cluster == 0:
            raise RuntimeError("intentional")
        return {"k_local": 2, "local_micro_ei": 0.1, "local_macro_ei": 0.2, "local_delta_ei": 0.1, "local_i_available": 0.3, "local_i_retained": 0.2, "local_cq": 2 / 3}

    monkeypatch.setattr("mignet_ce.downstream.application.optimal_cg.local_pair_analysis.run_local_optimal_cg", fake_train)
    result = run_local_pair_analysis(training_root=training, inputs={}, maturity_t=tmp_path / "m1.csv", maturity_tp=tmp_path / "m2.csv", k_global=2, device="cpu", min_local_spots=1)
    assert result["completed_count"] == 1
    statuses = [row["status"] for row in result["rows"]]
    assert statuses == ["failed", "completed"]
    assert (tmp_path / "cluster_pair_analysis" / "pair_summary.csv").is_file()
