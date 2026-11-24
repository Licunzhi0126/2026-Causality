"""O(M + number of nodes) memory uniform sampling, without dense adjacency."""
import numpy as np

NETWORKS = ('cci_t', 'grn_t', 'cci_tp', 'grn_tp')


def streams(base_seed, repeat_idx):
    children = np.random.SeedSequence([base_seed, repeat_idx]).spawn(4)
    return dict(zip(NETWORKS, children))


def sample_without_replacement(size, count, rng):
    """Floyd sampling: uniform subset, bounded work even for dense graphs."""
    if count > size or count < 0:
        raise ValueError(f'Cannot sample {count} edges from {size} legal pairs')
    chosen = set()
    for end in range(size - count, size):
        draw = int(rng.integers(end + 1))
        chosen.add(end if draw in chosen else draw)
    return np.asarray(sorted(chosen), dtype=np.int64)


def rewire(sources, targets, weights, seed):
    """Cartesian role universe minus self pairs; independent topology/weight streams."""
    topology_seed, weight_seed = seed.spawn(2)
    rng = np.random.default_rng(topology_seed)
    targets = np.asarray(targets)
    target_lookup = {node: idx for idx, node in enumerate(targets)}
    excluded = np.array([target_lookup.get(node, -1) for node in sources])
    counts = len(targets) - (excluded >= 0)
    cumulative = np.cumsum(counts, dtype=np.int64)
    size = int(cumulative[-1]) if len(cumulative) else 0
    ranks = sample_without_replacement(size, len(weights), rng)
    rows = np.searchsorted(cumulative, ranks, side='right')
    starts = np.r_[0, cumulative[:-1]]
    cols = ranks - starts[rows]
    cols += (excluded[rows] >= 0) & (cols >= excluded[rows])
    values = np.sort(np.asarray(weights)).copy()
    np.random.default_rng(weight_seed).shuffle(values)
    return rows, cols, values, size
