from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.downstream.perturbation.propagation_bridge import propagate_delta, propagate_state
from mignet_ce.downstream.perturbation.state_bridge import load_baseline_spot_pij


def test_propagation_uses_transposed_source_to_target_pij() -> None:
    pij = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.allclose(propagate_delta(np.array([1.0, 2.0]), pij), np.array([1.0, 2.0]))


@pytest.mark.parametrize('source,target,features', [(3, 5, 1), (7, 2, 4), (1, 6, 3), (0, 3, 2)])
def test_migrated_propagation_matches_original_matrix_formula(source, target, features):
    rng = np.random.default_rng(20261116)
    pij = rng.random((source, target))
    for delta in (rng.normal(size=(source, features)), rng.normal(size=source)):
        expected = np.asarray(pij).T @ np.asarray(delta)
        assert np.array_equal(propagate_delta(delta, pij), expected)
        assert propagate_delta(delta, pij).shape == expected.shape


def test_propagated_state_retains_baseline_and_rectangular_direction():
    baseline_t = np.array([[1., 2.], [3., 4.]])
    baseline_tp = np.array([[10., 20.], [30., 40.], [50., 60.]])
    delta = np.array([[2., -1.], [-3., 4.]])
    pij = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    state, transferred = propagate_state(baseline_t + delta, baseline_t, baseline_tp, pij)
    assert np.array_equal(transferred, pij.T @ delta)
    assert np.array_equal(state, baseline_tp + pij.T @ delta)


@pytest.mark.parametrize('delta,pij', [
    (np.zeros((3, 2)), np.zeros((2, 4))),
    (np.array(1.), np.zeros((2, 4))),
    (np.zeros(2), np.zeros(2)),
    (np.array([np.nan, 1.]), np.ones((2, 3))),
    (np.zeros(2), np.array([[np.inf], [1.]])),
])
def test_propagation_preserves_input_validation(delta, pij):
    with pytest.raises(ValueError):
        propagate_delta(delta, pij)


@pytest.mark.parametrize("problem", [None, "shape", "order", "nan", "negative", "row_sum"])
def test_rectangular_baseline_pij_and_assignment_validation(tmp_path, problem):
    pij = np.array([[1., 0.], [0.25, 0.75], [0., 1.]])
    source, target = ["s0", "s1", "s2"], ["t0", "t1"]
    if problem == "shape":
        pij = pij.T
    if problem == "nan":
        pij[0, 0] = np.nan
    if problem == "negative":
        pij[0] = [1.2, -0.2]
    if problem == "row_sum":
        pij[0] = [0.2, 0.2]
    path = tmp_path / "PIJ_micro_train.npy"
    np.save(path, pij)
    source_csv = source[::-1] if problem == "order" else source
    (tmp_path / "assignments_t.csv").write_text("unit_id\n" + "\n".join(source_csv))
    (tmp_path / "assignments_tp.csv").write_text("unit_id\n" + "\n".join(target))
    if problem is not None:
        match = {"shape": "shape", "order": "spot order"}.get(problem, "row stochastic")
        with pytest.raises(ValueError, match=match):
            load_baseline_spot_pij(path, source, target, {})
        return
    diagnostics = {}
    actual_pij = load_baseline_spot_pij(path, source, target, diagnostics)
    delta = np.array([[0., 1., 2., 3.], [4., 8., 12., 16.], [2., 3., 4., 5.]])
    propagated = propagate_delta(delta, actual_pij)
    assert propagated.shape == (2, 4)
    np.testing.assert_array_equal(propagated, [[1., 3., 5., 7.], [5., 9., 13., 17.]])
    assert diagnostics["spot_order_verified"] and diagnostics["spot_pij_verified"]
