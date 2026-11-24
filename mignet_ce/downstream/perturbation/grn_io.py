"""GRN edge-table IO using the repository's established three-column contract."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mignet_ce.io.loaders import standardize_grn_edges


@dataclass(frozen=True)
class GRNGraph:
    nodes: tuple[str, ...]
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.adjacency, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(self.nodes):
            raise ValueError("adjacency must be square and match nodes.")
        if not np.isfinite(matrix).all():
            raise ValueError("GRN weights must be finite.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("GRN node IDs must be unique.")

    def copy(self) -> "GRNGraph":
        return GRNGraph(self.nodes, self.adjacency.copy())

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.adjacency))


def read_grn(path: Path, *, nodes_path: Path | None = None) -> GRNGraph:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = standardize_grn_edges(raw)
    if len(table) != len(raw):
        raise ValueError(f"GRN has invalid weights: {path}")
    if nodes_path is None:
        # Baseline weights retain the same CSV conversion as the formal frontend.
        table["weight"] = standardize_grn_edges(pd.read_csv(path))["weight"]
    else:
        # Parse the original numeric strings to round-trip the serialized floats.
        weight_column = next(col for col in raw if col.strip().lower() in {"weight", "importance", "score"})
        table["weight"] = raw[weight_column].map(float)
    if table.duplicated(["regulator", "target"]).any():
        raise ValueError(f"GRN has duplicate directed edges: {path}")
    endpoints = set(table["regulator"]) | set(table["target"])
    if any(not node.strip() for node in endpoints):
        raise ValueError(f"GRN has blank gene IDs: {path}")
    if nodes_path is None:
        nodes = tuple(sorted(endpoints))
    else:
        saved = json.loads(nodes_path.read_text(encoding="utf-8"))
        if not isinstance(saved, list) or any(not isinstance(node, str) or not node.strip() for node in saved):
            raise ValueError(f"GRN node file must contain an ordered list of gene IDs: {nodes_path}")
        nodes = tuple(saved)
        unknown = endpoints - set(nodes)
        if unknown:
            raise ValueError(f"GRN has unknown genes: {sorted(unknown)[:10]}")
    index = {node: position for position, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    for row in table.itertuples(index=False):
        adjacency[index[str(row.regulator)], index[str(row.target)]] = float(row.weight)
    return GRNGraph(nodes, adjacency)


def edge_table(graph: GRNGraph) -> pd.DataFrame:
    source, target = np.nonzero(graph.adjacency)
    return pd.DataFrame({
        "regulator": [graph.nodes[index] for index in source],
        "target": [graph.nodes[index] for index in target],
        "weight": graph.adjacency[source, target],
    }).sort_values(["regulator", "target"]).reset_index(drop=True)


def write_grn(graph: GRNGraph, path: Path, *, nodes_path: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_table(graph).to_csv(path, index=False)
    if nodes_path is not None:
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        nodes_path.write_text(json.dumps(graph.nodes, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
