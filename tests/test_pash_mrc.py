from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))

import pash_mrc  # noqa: E402
from pash_mrc import PASHMRCConfig, fit_single_timepoint  # noqa: E402


def _synthetic_sample(n_spots: int = 48, n_genes: int = 24) -> tuple[sp.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(20260723)
    angle = np.linspace(0, 2 * np.pi, n_spots, endpoint=False)
    radius = 2.0 + 0.35 * np.sin(3 * angle)
    coords = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    coords += rng.normal(scale=0.03, size=coords.shape)

    state = ((angle % (2 * np.pi)) / (2 * np.pi / 4)).astype(int)
    rate = np.full((n_spots, n_genes), 0.4)
    for group in range(4):
        rate[state == group, group * 5 : group * 5 + 7] += 4.0
    counts = rng.poisson(rate).astype(np.float64)
    return sp.csr_matrix(counts), coords


def _small_config() -> PASHMRCConfig:
    return PASHMRCConfig(
        n_hvg=20,
        n_pca=8,
        n_states=4,
        ring_ends=(2, 4, 6),
        composition_dim=6,
        k40=6,
        k150=12,
        clustering_knn=5,
        diagnostic_knn=4,
        icm_passes=1,
    )


def test_pash_mrc_is_deterministic_exact_and_strictly_nested() -> None:
    expression, coords = _synthetic_sample()
    first = fit_single_timepoint(expression, coords, config=_small_config())
    second = fit_single_timepoint(expression, coords, config=_small_config())

    np.testing.assert_array_equal(first.labels_k40, second.labels_k40)
    np.testing.assert_array_equal(first.labels_k150, second.labels_k150)
    assert len(np.unique(first.labels_k40)) == 6
    assert len(np.unique(first.labels_k150)) == 12
    for child in np.unique(first.labels_k150):
        assert len(np.unique(first.labels_k40[first.labels_k150 == child])) == 1


def test_pash_mrc_metadata_and_signature_enforce_no_leakage_inputs() -> None:
    expression, coords = _synthetic_sample()
    result = fit_single_timepoint(expression, coords, config=_small_config())

    assert set(inspect.signature(fit_single_timepoint).parameters) == {
        "expression",
        "coords",
        "config",
    }
    assert result.metadata["prospective_single_timepoint"] is True
    assert result.metadata["uses_future_time"] is False
    assert result.metadata["uses_cci"] is False
    assert result.metadata["uses_grn"] is False
    assert result.metadata["uses_pgr"] is False
    assert result.metadata["uses_pij_or_ei"] is False
    assert result.metadata["prototype_neighbor_bug_corrected"] is True

    source = inspect.getsource(pash_mrc)
    assert "mignet_ce.pij" not in source
    assert "mignet_ce.networks" not in source
    assert "effective_information" not in source


def test_neighbor_ranking_keeps_the_nearest_real_neighbor() -> None:
    coords = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
    neighbors = pash_mrc._ranked_neighbors_excluding_self(coords, 3)

    assert neighbors.shape == (8, 3)
    assert neighbors[0].tolist() == [1, 2, 3]
    assert 0 not in neighbors[0]
    assert all(row_index not in row for row_index, row in enumerate(neighbors.tolist()))


def test_sparse_expression_is_supported_without_changing_input() -> None:
    expression, coords = _synthetic_sample()
    before = expression.copy()
    result = fit_single_timepoint(expression, coords, config=_small_config())

    assert result.features.shape[0] == expression.shape[0]
    assert (expression != before).nnz == 0

