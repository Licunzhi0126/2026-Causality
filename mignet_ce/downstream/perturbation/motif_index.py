"""Sparse three- and four-node motif enumeration shared by metrics and perturbation search."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, islice, permutations
from pathlib import Path

import numpy as np

from .grn_io import GRNGraph

Edge = tuple[int, int]

HIGHER_ORDER_TEMPLATES = {
    "P9_four_node_feedback_cycle": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "P10_convergent_motif": ((0, 1), (2, 1)),
    "P11_cascade_motif": ((0, 1), (1, 2)),
    "P12_feedback_with_shortcut": ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)),
    "P13_divergent_motif": ((1, 0), (1, 2)),
}


def _template_matches(
    outgoing: tuple[set[int], ...], incoming: tuple[set[int], ...], template: tuple[Edge, ...],
):
    """Yield one representative per non-induced match without retaining a set."""
    size = 1 + max(max(edge) for edge in template)
    edge_set = set(template)
    automorphisms = tuple(perm for perm in permutations(range(size))
                          if {(perm[a], perm[b]) for a, b in template} == edge_set)
    assigned: dict[int, int] = {}

    def visit():
        if len(assigned) == size:
            nodes = tuple(assigned[i] for i in range(size))
            if all(nodes <= tuple(nodes[i] for i in perm) for perm in automorphisms):
                yield nodes
            return
        remaining = set(range(size)) - assigned.keys()
        # Always expand a role adjacent to the partial match, intersecting
        # sparse neighbor sets instead of scanning a Cartesian node product.
        role = max(remaining, key=lambda r: (sum((a == r and b in assigned) or
                   (b == r and a in assigned) for a, b in template), -r))
        candidates = None
        for a, b in template:
            neighbors = incoming[assigned[b]] if a == role and b in assigned else (
                outgoing[assigned[a]] if b == role and a in assigned else None)
            if neighbors is not None:
                candidates = set(neighbors) if candidates is None else candidates & neighbors
        if candidates is None:
            candidates = range(len(outgoing))
        for node in sorted(set(candidates) - set(assigned.values())):
            assigned[role] = node
            yield from visit()
            del assigned[role]

    yield from visit()


def _enumerate_template(outgoing, incoming, template):
    return tuple(sorted(_template_matches(outgoing, incoming, template)))


@dataclass(frozen=True)
class MotifIndex:
    """Fixed baseline-support motifs; entries contain node IDs, never weights."""
    triangles: tuple[tuple[int, int, int], ...]
    feedforward_loops: tuple[tuple[int, int, int], ...]
    feedback_cycles: tuple[tuple[int, int, int], ...]

    template_occurrences: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)

    def occurrences(self, pair_id: str) -> tuple[tuple[int, ...], ...]:
        return self.template_occurrences[pair_id]

    @property
    def motif_edges(self) -> tuple[Edge, ...]:
        edges: set[Edge] = set()
        for i, j, k in self.triangles:
            edges.update(((i, j), (j, i), (j, k), (k, j), (i, k), (k, i)))
        for a, b, c in self.feedforward_loops:
            edges.update(((a, b), (a, c), (b, c)))
        for a, b, c in self.feedback_cycles:
            edges.update(((a, b), (b, c), (c, a)))
        for pair_id, occurrences in self.template_occurrences.items():
            for nodes in occurrences:
                edges.update((nodes[a], nodes[b]) for a, b in HIGHER_ORDER_TEMPLATES[pair_id])
        return tuple(sorted(edges))


def compact_chunks(values, length=131072):
    for offset in range(0, len(values), length):
        yield values[offset:offset + length]


def _compact_file(directory, name):
    return directory / f"{name.split('_', 1)[0]}.bin"


def read_compact_index(directory: Path) -> MotifIndex:
    def read(name, width):
        path = _compact_file(directory, name)
        count, remainder = divmod(path.stat().st_size, 4 * width)
        if remainder:
            raise ValueError(f"Corrupt motif index: {path}")
        return (np.memmap(path, mode="r", dtype="<i4", shape=(count, width))
                if count else np.empty((0, width), dtype=np.int32))

    return MotifIndex(read("triangles", 3), read("feedforward", 3), read("feedback", 3),
                      {key: read(key, 1 + max(max(edge) for edge in template))
                       for key, template in HIGHER_ORDER_TEMPLATES.items()})


def _motif_streams(outgoing, incoming, undirected):
    for name in ("triangles", "feedforward", "feedback", *HIGHER_ORDER_TEMPLATES):
        yield name, _motif_stream(name, outgoing, incoming, undirected)


def _motif_stream(name, outgoing, incoming, undirected):
    n = len(outgoing)
    if name == "triangles":
        for a in range(n):
            for b, c in combinations(sorted(node for node in undirected[a] if node > a), 2):
                if c in undirected[b]:
                    yield a, b, c
    elif name == "feedforward":
        for a in range(n):
            for b in outgoing[a]:
                for c in outgoing[a] & outgoing[b]:
                    yield a, b, c
    elif name == "feedback":
        for a in range(n):
            for b in outgoing[a]:
                if b > a:
                    for c in outgoing[b] & incoming[a]:
                        if c > a:
                            yield a, b, c
    elif name.startswith("P9_"):
        # Rotations describe the same directed cycle. Start at its smallest ID.
        for a in range(n):
            for b in outgoing[a]:
                if b > a:
                    for c in outgoing[b]:
                        if c > a:
                            for d in outgoing[c] & incoming[a]:
                                if d > a and d != b:
                                    yield a, b, c, d
    elif name.startswith("P12_"):
        # The shortcut fixes roles a and c, so there is no rotational quotient.
        for a in range(n):
            for b in outgoing[a]:
                for c in outgoing[b] & outgoing[a]:
                    for d in outgoing[c] & incoming[a]:
                        if d != b:
                            yield a, b, c, d
    elif name.startswith("P11_"):
        for b in range(n):
            for a in incoming[b]:
                for c in outgoing[b]:
                    if c != a:
                        yield a, b, c
    elif name.startswith(("P10_", "P13_")):
        neighbors = incoming if name.startswith("P10_") else outgoing
        for b in range(n):
            for a, c in combinations(sorted(neighbors[b]), 2):
                yield a, b, c
    else:
        yield from _template_matches(outgoing, incoming, HIGHER_ORDER_TEMPLATES[name])


def _write_stream(directory, name, stream):
    with _compact_file(directory, name).open("wb") as target:
        while chunk := list(islice(stream, 131072)):
            np.asarray(chunk, dtype="<i4").tofile(target)


def _init_index_worker(outgoing, incoming, undirected):
    global _INDEX_NEIGHBORS
    _INDEX_NEIGHBORS = outgoing, incoming, undirected


def _write_index_worker(directory, name):
    _write_stream(directory, name, _motif_stream(name, *_INDEX_NEIGHBORS))


def build_motif_index(graph: GRNGraph, directory: Path | None = None, *, workers: int = 1) -> MotifIndex:
    """Enumerate only motifs present in the fixed directed edge support."""
    adjacency = graph.adjacency
    if np.any(adjacency < 0.0):
        raise ValueError("Weighted motif interventions require non-negative GRN weights.")
    n = len(graph.nodes)
    outgoing = tuple(set(np.flatnonzero(adjacency[i] > 0.0).tolist()) - {i} for i in range(n))
    undirected = tuple(outgoing[i] | set(np.flatnonzero(adjacency[:, i] > 0.0).tolist()) - {i} for i in range(n))
    triangles = ((i, j, k) for i in range(n)
                 for j, k in combinations(sorted(node for node in undirected[i] if node > i), 2)
                 if k in undirected[j])
    ffl = ((a, b, c) for a in range(n) for b in outgoing[a] for c in outgoing[a] & outgoing[b] if c != b)
    cycles = ((a, b, c) for a in range(n) for b in outgoing[a] for c in outgoing[b]
              if c != a and a in outgoing[c] and a < b and a < c)
    incoming = tuple(set(np.flatnonzero(adjacency[:, i] > 0.0).tolist()) - {i} for i in range(n))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        if workers == 1:
            for name, stream in _motif_streams(outgoing, incoming, undirected):
                _write_stream(directory, name, stream)
        else:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context

            names = (*HIGHER_ORDER_TEMPLATES, "triangles", "feedforward", "feedback")
            with ProcessPoolExecutor(max_workers=min(workers, len(names)), mp_context=get_context("spawn"),
                                     initializer=_init_index_worker, initargs=(outgoing, incoming, undirected)) as pool:
                for future in [pool.submit(_write_index_worker, directory, name) for name in names]:
                    future.result()
        return read_compact_index(directory)
    occurrences = {pair_id: _enumerate_template(outgoing, incoming, template)
                   for pair_id, template in HIGHER_ORDER_TEMPLATES.items()}
    return MotifIndex(tuple(triangles), tuple(ffl), tuple(cycles), occurrences)


def motif_edge_participation(index: MotifIndex, pair_id: str, node_count: int | None = None) -> dict[Edge, float]:
    """Unweighted participation counts used only to freeze a candidate edge region."""
    counts: dict[Edge, float] = {}
    if isinstance(index.triangles, np.ndarray):
        patterns = {
            "P6_triangle_intensity": (index.triangles, ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0))),
            "P7_feedforward_loop": (index.feedforward_loops, ((0, 1), (0, 2), (1, 2))),
            "P8_feedback_cycle": (index.feedback_cycles, ((0, 1), (1, 2), (2, 0))),
        }
        values, edges = patterns[pair_id] if pair_id in patterns else (index.occurrences(pair_id), HIGHER_ORDER_TEMPLATES[pair_id])
        if node_count is None:
            node_count = 1 + int(values.max()) if len(values) else 1
        for chunk in compact_chunks(values):
            for a, b in edges:
                keys, frequencies = np.unique(chunk[:, a].astype(np.int64) * node_count + chunk[:, b], return_counts=True)
                for key, count in zip(keys.tolist(), frequencies.tolist(), strict=True):
                    edge = divmod(key, node_count)
                    counts[edge] = counts.get(edge, 0.0) + count
        return counts
    if pair_id == "P6_triangle_intensity":
        for i, j, k in index.triangles:
            for edge in ((i, j), (j, i), (j, k), (k, j), (i, k), (k, i)):
                counts[edge] = counts.get(edge, 0.0) + 1.0
    elif pair_id == "P7_feedforward_loop":
        for a, b, c in index.feedforward_loops:
            for edge in ((a, b), (a, c), (b, c)):
                counts[edge] = counts.get(edge, 0.0) + 1.0
    elif pair_id == "P8_feedback_cycle":
        for a, b, c in index.feedback_cycles:
            for edge in ((a, b), (b, c), (c, a)):
                counts[edge] = counts.get(edge, 0.0) + 1.0
    elif pair_id in HIGHER_ORDER_TEMPLATES:
        for nodes in index.occurrences(pair_id):
            for a, b in HIGHER_ORDER_TEMPLATES[pair_id]:
                edge = (nodes[a], nodes[b])
                counts[edge] = counts.get(edge, 0.0) + 1.0
    else:
        raise ValueError(f"Not a motif pair: {pair_id}")
    return counts
