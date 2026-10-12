"""Fixed topology-region discovery and nested perturbation selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Mapping

import numpy as np

from .condition import PAIR_IDS, derived_seed
from .config import PerturbationConfig
from .grn_io import GRNGraph
from .motif_index import MotifIndex, build_motif_index, motif_edge_participation

Edge = tuple[int, int]


@dataclass(frozen=True)
class TopologyRegion:
    """One reproducible, topology-specific region shared by all strengths."""
    pair_id: str
    ranked_nodes: tuple[int, ...]
    ranked_edges: tuple[Edge, ...]
    node_scores: tuple[float, ...]
    edge_scores: tuple[float, ...]
    region_nodes: tuple[int, ...]
    region_edges: tuple[Edge, ...]
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationSelection:
    pair_id: str
    strength: float
    region_nodes: tuple[int, ...]
    region_edges: tuple[Edge, ...]
    selected_nodes: tuple[int, ...]
    selected_edges: tuple[Edge, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rank(items: list[tuple[object, float]], *, seed: int) -> list[object]:
    return [item for item, _ in sorted(items, key=lambda pair: (-pair[1], derived_seed(seed, pair[0])))]


def _existing_edges(graph: GRNGraph) -> tuple[Edge, ...]:
    return tuple((int(i), int(j)) for i, j in zip(*np.nonzero(graph.adjacency), strict=True) if i != j)


def _node_scores(graph: GRNGraph, pair_id: str, communities: Mapping[str, int], *, motif_index: MotifIndex | None = None) -> tuple[np.ndarray, dict[Edge, float], tuple[Edge, ...]]:
    """Compute method-specific scores once; only discover_region may call this."""
    adjacency = graph.adjacency
    support = (adjacency != 0.0) | (adjacency.T != 0.0)
    total_degree = support.sum(axis=0) + support.sum(axis=1)
    existing = _existing_edges(graph)
    if pair_id == "P1_degree_heterogeneity":
        scores = np.abs(total_degree - np.median(total_degree))
        return scores, {edge: float(scores[edge[0]] + scores[edge[1]]) for edge in existing}, existing
    if pair_id == "P2_modularity":
        labels = np.asarray([communities[node] for node in graph.nodes])
        inter = np.asarray([np.count_nonzero(support[index] & (labels != labels[index])) for index in range(len(graph.nodes))])
        scores = inter / np.maximum(total_degree, 1)
        edge_scores = {edge: float(2 if labels[edge[0]] != labels[edge[1]] else 1) + adjacency[edge] / max(float(adjacency.max()), 1e-12) for edge in existing}
        return scores, edge_scores, existing
    if pair_id == "P3_clustering":
        support_i64 = support.astype(np.int64)
        common = support_i64 @ support_i64
        triangle_counts = (common * support_i64).sum(axis=1) / 2.0
        possible = total_degree * np.maximum(total_degree - 1, 1) / 2.0
        scores = triangle_counts / possible
        return scores, {edge: float(common[edge]) for edge in existing}, existing
    if pair_id == "P4_assortativity":
        out_degree = np.count_nonzero(adjacency, axis=1)
        in_degree = np.count_nonzero(adjacency, axis=0)
        scores = np.abs(total_degree - np.median(total_degree))
        return scores, {edge: float(abs(out_degree[edge[0]] - in_degree[edge[1]])) for edge in existing}, existing
    if pair_id == "P5_reciprocity":
        reciprocal = tuple(edge for edge in existing if adjacency[edge[1], edge[0]] != 0.0)
        scores = np.zeros(len(graph.nodes), dtype=float)
        for i, j in reciprocal:
            scores[i] += 1
            scores[j] += 1
        return scores, {edge: 1.0 for edge in reciprocal}, reciprocal
    if pair_id in {"P6_triangle_intensity", "P7_feedforward_loop", "P8_feedback_cycle"}:
        edge_scores = motif_edge_participation(motif_index or build_motif_index(graph), pair_id)
        scores = np.zeros(len(graph.nodes), dtype=float)
        for (i, j), score in edge_scores.items():
            scores[i] += score
            scores[j] += score
        return scores, edge_scores, tuple(edge for edge in existing if edge in edge_scores)
    raise ValueError(f"Unknown topology pair: {pair_id}")


def _take(ranked: tuple[object, ...], fraction: float) -> tuple[object, ...]:
    if not ranked:
        return ()
    return ranked[:max(1, ceil(fraction * len(ranked)))]


def discover_region(graph: GRNGraph, *, pair_id: str, communities: Mapping[str, int], seed: int, config: PerturbationConfig, motif_index: MotifIndex | None = None) -> TopologyRegion:
    """Discover and freeze one pair-specific region for all perturbation strengths."""
    if pair_id not in PAIR_IDS:
        raise ValueError(f"Unknown topology pair: {pair_id}")
    node_scores, edge_scores, base_edges = _node_scores(graph, pair_id, communities, motif_index=motif_index)
    ranked_nodes = tuple(_rank([(index, float(score)) for index, score in enumerate(node_scores)], seed=seed))
    region_nodes = tuple(_take(ranked_nodes, config.region_node_fraction))
    region_node_set = set(region_nodes)
    candidate_edges = tuple(edge for edge in base_edges if edge[0] in region_node_set or edge[1] in region_node_set)
    ranked_edges = tuple(_rank([(edge, edge_scores[edge]) for edge in candidate_edges], seed=seed))
    region_edges = tuple(_take(ranked_edges, config.region_edge_fraction))
    return TopologyRegion(pair_id, ranked_nodes, ranked_edges, tuple(float(node_scores[node]) for node in ranked_nodes), tuple(float(edge_scores[edge]) for edge in ranked_edges), region_nodes, region_edges, seed)


def selection_from_region(region: TopologyRegion, *, strength: float) -> PerturbationSelection:
    """Take a nested edge prefix from a previously discovered fixed region."""
    if not 0.0 < strength < 1.0:
        raise ValueError("strength must be strictly between zero and one.")
    selected_edges = tuple(_take(region.region_edges, strength))
    selected_nodes = tuple(sorted({node for edge in selected_edges for node in edge}))
    return PerturbationSelection(region.pair_id, strength, region.region_nodes, region.region_edges, selected_nodes, selected_edges)
