"""Descriptive topology metrics for baseline and perturbed graphs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import networkx as nx
import numpy as np

from .grn_io import GRNGraph
from .motif_index import MotifIndex, build_motif_index

EPS = 1e-12


@dataclass(frozen=True)
class TopologyMetrics:
    degree_gini: float
    degree_cv: float
    centralization: float
    modularity: float
    clustering: float
    assortativity: float
    reciprocity: float
    edge_count: int
    node_count: int
    triangle_intensity: float = 0.0
    feedforward_loop: float = 0.0
    feedback_cycle: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


TARGET_METRIC = {
    "P1_degree_heterogeneity": "degree_gini",
    "P2_modularity": "modularity",
    "P3_clustering": "clustering",
    "P4_assortativity": "assortativity",
    "P5_reciprocity": "reciprocity",
    "P6_triangle_intensity": "triangle_intensity",
    "P7_feedforward_loop": "feedforward_loop",
    "P8_feedback_cycle": "feedback_cycle",
}


def _geometric_mean(*weights: float) -> float:
    return float(np.prod(weights) ** (1.0 / len(weights))) if all(weight > 0.0 for weight in weights) else 0.0


def compute_motif_metrics(graph: GRNGraph, *, baseline_scale: float | None = None, index: MotifIndex | None = None) -> dict[str, float]:
    """Compute T6/T7/T8 with one externally fixed normalization scale."""
    if np.any(graph.adjacency < 0.0):
        raise ValueError("Weighted motif metrics require non-negative GRN weights.")
    positive = graph.adjacency[graph.adjacency > 0.0]
    scale = float(baseline_scale) if baseline_scale is not None else (float(np.quantile(positive, 0.95)) if positive.size else 1.0)
    if scale <= EPS:
        scale = 1.0
    weights = graph.adjacency / scale
    index = index or build_motif_index(graph)
    t6 = sum(_geometric_mean(max(weights[i, j], weights[j, i]), max(weights[j, k], weights[k, j]), max(weights[i, k], weights[k, i])) for i, j, k in index.triangles)
    t7 = sum(_geometric_mean(weights[a, b], weights[a, c], weights[b, c]) for a, b, c in index.feedforward_loops)
    t8 = sum(_geometric_mean(weights[a, b], weights[b, c], weights[c, a]) for a, b, c in index.feedback_cycles)
    return {"triangle_intensity": float(t6), "feedforward_loop": float(t7), "feedback_cycle": float(t8), "baseline_weight_scale": scale}


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or float(values.sum()) <= EPS:
        return 0.0
    ranks = np.arange(1, len(values) + 1, dtype=float)
    return float((2.0 * np.dot(ranks, values) / (len(values) * values.sum())) - (len(values) + 1.0) / len(values))


def to_networkx(graph: GRNGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes)
    source, target = np.nonzero(graph.adjacency)
    result.add_weighted_edges_from((graph.nodes[i], graph.nodes[j], float(graph.adjacency[i, j])) for i, j in zip(source, target, strict=True))
    return result


def baseline_communities(graph: GRNGraph) -> dict[str, int]:
    undirected = to_networkx(graph).to_undirected()
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight")) if undirected.number_of_edges() else [{node} for node in graph.nodes]
    ordered = sorted((sorted(group) for group in communities), key=lambda group: group[0])
    return {node: number for number, group in enumerate(ordered) for node in group}


def compute_topology_metrics(graph: GRNGraph, communities: Mapping[str, int] | None = None, *, baseline_scale: float | None = None, motif_index: MotifIndex | None = None) -> TopologyMetrics:
    directed = to_networkx(graph)
    degrees = np.fromiter((directed.out_degree(node) for node in graph.nodes), dtype=float)
    degree_cv = float(np.std(degrees) / max(float(np.mean(degrees)), EPS))
    centralization = float((len(degrees) * float(np.max(degrees)) - float(degrees.sum())) / max((len(degrees) - 1) ** 2, 1))
    if communities is None:
        communities = baseline_communities(graph)
    groups: dict[int, set[str]] = {}
    for node in graph.nodes:
        groups.setdefault(int(communities[node]), set()).add(node)
    modularity = float(nx.algorithms.community.modularity(directed, list(groups.values()), weight="weight")) if directed.number_of_edges() else 0.0
    clustering = float(nx.average_clustering(directed, weight="weight")) if directed.number_of_nodes() else 0.0
    assortativity = nx.degree_pearson_correlation_coefficient(directed, x="out", y="in", weight=None) if directed.number_of_edges() > 1 else 0.0
    assortativity = float(assortativity) if np.isfinite(assortativity) else 0.0
    reciprocity = nx.reciprocity(directed)
    motifs = compute_motif_metrics(graph, baseline_scale=baseline_scale, index=motif_index) if np.all(graph.adjacency >= 0.0) else {"triangle_intensity": float("nan"), "feedforward_loop": float("nan"), "feedback_cycle": float("nan")}
    return TopologyMetrics(_gini(degrees), degree_cv, centralization, modularity, clustering, assortativity, float(reciprocity or 0.0), graph.edge_count, len(graph.nodes), **{key: motifs[key] for key in ("triangle_intensity", "feedforward_loop", "feedback_cycle")})


def metric_value(metrics: TopologyMetrics, pair_id: str) -> float:
    return float(getattr(metrics, TARGET_METRIC[pair_id]))
