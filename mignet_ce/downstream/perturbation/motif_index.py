"""Sparse three- and four-node motif enumeration shared by metrics and perturbation search."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

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


def _enumerate_template(
    outgoing: tuple[set[int], ...], incoming: tuple[set[int], ...], template: tuple[Edge, ...],
) -> tuple[tuple[int, ...], ...]:
    """Non-induced sparse backtracking; quotient only template automorphisms."""
    size = 1 + max(max(edge) for edge in template)
    edge_set = set(template)
    automorphisms = tuple(perm for perm in permutations(range(size))
                          if {(perm[a], perm[b]) for a, b in template} == edge_set)
    matches: set[tuple[int, ...]] = set()
    assigned: dict[int, int] = {}

    def visit() -> None:
        if len(assigned) == size:
            nodes = tuple(assigned[i] for i in range(size))
            matches.add(min(tuple(nodes[i] for i in perm) for perm in automorphisms))
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
            visit()
            del assigned[role]

    visit()
    return tuple(sorted(matches))


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
    incoming = tuple(set(np.flatnonzero(adjacency[:, i] > 0.0).tolist()) - {i} for i in range(n))
    occurrences = {pair_id: _enumerate_template(outgoing, incoming, template)
                   for pair_id, template in HIGHER_ORDER_TEMPLATES.items()}
    return MotifIndex(tuple(triangles), tuple(ffl), tuple(cycles), occurrences)


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
    elif pair_id in HIGHER_ORDER_TEMPLATES:
        for nodes in index.occurrences(pair_id):
            for a, b in HIGHER_ORDER_TEMPLATES[pair_id]:
                edge = (nodes[a], nodes[b])
                counts[edge] = counts.get(edge, 0.0) + 1.0
    else:
        raise ValueError(f"Not a motif pair: {pair_id}")
    return counts
