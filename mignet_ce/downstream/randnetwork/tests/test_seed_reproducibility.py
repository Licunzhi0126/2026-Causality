import numpy as np
from ..randomize import streams, rewire


def test_seed_and_edge_order_independence():
    def run(idx, weights):
        return {k: rewire(list('abcdef'), list('abcdef'), weights, s)[:3] for k, s in streams(42, idx).items()}
    a, b, c = run(0, [1, 3, 2, 4]), run(0, [4, 2, 3, 1]), run(1, [1, 2, 3, 4])
    for key in a:
        assert all(np.array_equal(x, y) for x, y in zip(a[key], b[key]))
        assert not all(np.array_equal(x, y) for x, y in zip(a[key], c[key]))
