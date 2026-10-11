"""GRN edge-table IO using the repository's established three-column contract."""
from __future__ import annotations

from dataclasses import dataclass
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


def read_grn(path: Path) -> GRNGraph:
    table = standardize_grn_edges(pd.read_csv(path))
    if table.duplicated(["regulator", "target"]).any():
        raise ValueError(f"GRN has duplicate directed edges: {path}")
    nodes = tuple(sorted(set(table["regulator"].astype(str)) | set(table["target"].astype(str))))
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


def write_grn(graph: GRNGraph, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_table(graph).to_csv(path, index=False)
    return path
