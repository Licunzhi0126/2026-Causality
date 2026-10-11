"""Sparse three-node motif enumeration shared by metrics and perturbation search."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .grn_io import GRNGraph

Edge = tuple[int, int]


@dataclass(frozen=True)
class MotifIndex:
    """Fixed baseline-support motifs; entries contain node IDs, never weights."""
    triangles: tuple[tuple[int, int, int], ...]
    feedforward_loops: tuple[tuple[int, int, int], ...]
    feedback_cycles: tuple[tuple[int, int, int], ...]

    @property
    def motif_edges(self) -> tuple[Edge, ...]:
        edges: set[Edge] = set()
        for i, j, k in self.triangles:
            edges.update(((i, j), (j, i), (j, k), (k, j), (i, k), (k, i)))
        for a, b, c in self.feedforward_loops:
            edges.update(((a, b), (a, c), (b, c)))
        for a, b, c in self.feedback_cycles:
            edges.update(((a, b), (b, c), (c, a)))
        return tuple(sorted(edges))


def build_motif_index(graph: GRNGraph) -> MotifIndex:
    """Enumerate only motifs present in the fixed directed edge support."""
    adjacency = graph.adjacency
    if np.any(adjacency < 0.0):
        raise ValueError("Weighted motif interventions require non-negative GRN weights.")
    n = len(graph.nodes)
    outgoing = tuple(set(np.flatnonzero(adjacency[i] > 0.0).tolist()) - {i} for i in range(n))
    undirected = tuple(outgoing[i] | set(np.flatnonzero(adjacency[:, i] > 0.0).tolist()) - {i} for i in range(n))
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j, k in combinations(sorted(node for node in undirected[i] if node > i), 2):
            if k in undirected[j]:
                triangles.append((i, j, k))
    ffl: list[tuple[int, int, int]] = []
    for a in range(n):
        for b in outgoing[a]:
            for c in outgoing[a] & outgoing[b]:
                if c != b:
                    ffl.append((a, b, c))
    cycles: list[tuple[int, int, int]] = []
    for a in range(n):
        for b in outgoing[a]:
            for c in outgoing[b]:
                if c != a and a in outgoing[c] and a < b and a < c:
                    cycles.append((a, b, c))
    return MotifIndex(tuple(triangles), tuple(ffl), tuple(cycles))


def motif_edge_participation(index: MotifIndex, pair_id: str) -> dict[Edge, float]:
    """Unweighted participation counts used only to freeze a candidate edge region."""
    counts: dict[Edge, float] = {}
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
    else:
        raise ValueError(f"Not a motif pair: {pair_id}")
    return counts
