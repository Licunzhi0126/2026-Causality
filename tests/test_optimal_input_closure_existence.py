from __future__ import annotations

import json

import numpy as np
import pytest

from mignet_ce.downstream.analysis.dynamic_closure.optimal_input_existence import evaluate_optimal_run


def test_saved_optimal_run_has_consistent_ei_and_closure_budget(tmp_path) -> None:
    np.save(tmp_path / "S_t.npy", np.eye(2))
    np.save(tmp_path / "S_tp.npy", np.eye(2))
    np.save(tmp_path / "PIJ_micro_train.npy", np.array([[0.9, 0.1], [0.2, 0.8]]))
    (tmp_path / "summary.json").write_text(json.dumps({
        "method": "complete_combined_coarse_maturity_cci_grn",
        "K": 2, "best_epoch": 10,
        "EI_micro_fixed": 0.5, "EI_macro_best_checkpoint": 0.8,
        "delta_EI_best_checkpoint": 0.3,
    }), encoding="utf-8")
    result = evaluate_optimal_run(tmp_path)
    assert result["delta_EI_best_checkpoint"] == pytest.approx(0.3)
    assert result["closure_quality"] == pytest.approx(1.0)
    assert result["information_identity_error"] < 1e-8
    assert result["signal_status"] == "informative"


def test_saved_optimal_run_rejects_inconsistent_deltaei(tmp_path) -> None:
    np.save(tmp_path / "S_t.npy", np.eye(2))
    np.save(tmp_path / "S_tp.npy", np.eye(2))
    np.save(tmp_path / "PIJ_micro_train.npy", np.eye(2))
    (tmp_path / "summary.json").write_text(json.dumps({
        "method": "complete_combined_coarse_maturity_cci_grn",
        "K": 2, "best_epoch": 10,
        "EI_micro_fixed": 0.5, "EI_macro_best_checkpoint": 0.8,
        "delta_EI_best_checkpoint": 0.4,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="DeltaEI"):
        evaluate_optimal_run(tmp_path)
