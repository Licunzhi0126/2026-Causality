from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mignet_ce.visualization.downstream.dynamic_closure.analysis import (
    build_cross_representation_consistency,
    crossfit_macro_q_error,
    information_closure_budget,
)


def _hard(labels: list[int], k: int) -> np.ndarray:
    matrix = np.zeros((len(labels), k), dtype=float)
    matrix[np.arange(len(labels)), labels] = 1.0
    return matrix


def test_information_closure_identity_and_continuous_quality() -> None:
    transition = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.55, 0.30, 0.10, 0.05],
            [0.05, 0.10, 0.30, 0.55],
            [0.05, 0.05, 0.20, 0.70],
        ]
    )
    source = _hard([0, 0, 1, 1], 2)
    target = _hard([0, 0, 1, 1], 2)
    budget = information_closure_budget(
        transition,
        source,
        target,
        low_signal_threshold_bits=10.0,
    )
    assert budget["signal_status"] == "low-signal"
    assert np.isfinite(budget["closure_quality"])
    assert np.isclose(
        budget["I_available_bits"],
        budget["I_macro_retained_bits"] + budget["closure_leakage_bits"],
        atol=1e-10,
    )


def test_crossfit_excludes_singletons_from_coverage() -> None:
    observed = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    hard = _hard([0, 0, 1], 2)
    result = crossfit_macro_q_error(observed, hard, folds=5, seed=7)
    assert 0 < result["crossfit_coverage"] < 1
    assert result["singleton_weight_fraction"] > 0


def test_consistency_contains_all_six_mapping_pairs_at_four_times() -> None:
    mappings = ("a", "b", "c", "d")
    times = ("1", "2", "3", "4")
    pairs = ("1->2", "2->3", "3->4")
    records = {}
    spots = ("s1", "s2", "s3", "s4")
    for mapping_index, mapping in enumerate(mappings):
        for pair_index, pair in enumerate(pairs):
            labels = [(index + mapping_index + pair_index) % 2 for index in range(4)]
            assignment = _hard(labels, 2)
            records[(mapping, pair)] = SimpleNamespace(
                hs=assignment,
                ht=assignment,
                spots_s=spots,
                spots_t=spots,
            )
    cfg = SimpleNamespace(
        times=times,
        adjacent_pairs=pairs,
        mapping_names=mappings,
    )
    table = build_cross_representation_consistency(cfg, records)
    assert len(table) == 24
    assert table[["mapping_a", "mapping_b"]].drop_duplicates().shape[0] == 6
    assert set(table["time"]) == set(times)
