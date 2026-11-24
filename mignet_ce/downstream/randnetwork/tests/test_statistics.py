import pytest
from ..statistics import statistics, METRICS


def test_boundaries_ties_and_independent_metrics():
    repeats = [dict(zip(METRICS, (0., 1.))) for _ in range(100)]
    rows = statistics(dict(zip(METRICS, (1., 1.))), repeats)
    assert rows[0]['p_one_sided'] == 1 / 101
    assert rows[1]['p_one_sided'] == 1
    with pytest.raises(ValueError):
        statistics(repeats[0], repeats[:99])
    with pytest.raises(ValueError):
        statistics(repeats[0], repeats, expected=3)
