"""Align normalized biological maps and encode original labels as hard assignments."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def biological_assignment(map_path: Path, spot_order: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    table = pd.read_csv(map_path)
    required = {"spot_id", "cluster", "cell_type"}
    if not required.issubset(table.columns):
        raise ValueError(f"Biological cluster map must contain {sorted(required)}: {map_path}")
    table["spot_id"] = table["spot_id"].astype(str)
    table["cluster"] = table["cluster"].astype(str)
    if table["spot_id"].duplicated().any():
        raise ValueError(f"Biological cluster map has duplicate spot IDs: {map_path}")
    expected, observed = set(map(str, spot_order)), set(table["spot_id"])
    if expected != observed:
        raise ValueError(f"Biological map IDs differ (missing={sorted(expected-observed)[:5]}, extra={sorted(observed-expected)[:5]})")
    aligned = table.set_index("spot_id").loc[list(map(str, spot_order))].reset_index()
    metadata = aligned[["cluster", "cell_type"]].drop_duplicates()
    if metadata["cluster"].duplicated().any():
        raise ValueError("Each dataset-provided biological cluster must map to one cell type.")
    metadata = metadata.sort_values("cluster", kind="stable").reset_index(drop=True)
    positions = {cluster: index for index, cluster in enumerate(metadata["cluster"])}
    assignment = np.zeros((len(aligned), len(metadata)), dtype=np.float32)
    assignment[np.arange(len(aligned)), [positions[value] for value in aligned["cluster"]]] = 1.0
    if not np.allclose(assignment.sum(axis=1), 1.0):
        raise AssertionError("Biological assignments must be hard one-hot rows.")
    return assignment, metadata

