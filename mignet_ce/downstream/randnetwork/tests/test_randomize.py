import numpy as np
import pytest
from ..randomize import rewire, sample_without_replacement


def test_dense_bounded_sampling():
    assert np.array_equal(sample_without_replacement(100, 100, np.random.default_rng(4)), np.arange(100))
    with pytest.raises(ValueError):
        sample_without_replacement(3, 4, np.random.default_rng())


def test_uniform_pairs_weights_and_degree_changes():
    nodes = list('abcdefghij')
    rows, cols, values, size = rewire(nodes, nodes, np.arange(1, 31), np.random.SeedSequence(42))
    assert size == 90 and len(set(zip(rows, cols))) == 30
    assert np.all(rows != cols)
    assert np.array_equal(np.sort(values), np.arange(1, 31))
    assert len(set(np.bincount(rows, minlength=10))) > 1


def test_small_universe_sampling_frequency():
    counts = np.zeros(4)
    for i in range(2000):
        counts[sample_without_replacement(4, 2, np.random.default_rng(i))] += 1
    assert np.all(np.abs(counts - 1000) < 100)
