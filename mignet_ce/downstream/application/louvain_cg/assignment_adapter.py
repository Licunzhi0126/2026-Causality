"""Convert existing Louvain spot-domain maps to validated hard assignments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def one_hot_assignment(map_path: Path, spot_order: list[str], domain_order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(map_path)
    required = {"spot_id", "domain_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"{map_path} must contain {sorted(required)}")
    table["spot_id"] = table["spot_id"].astype(str)
    table["domain_id"] = table["domain_id"].astype(str)
    if table["spot_id"].duplicated().any():
        raise ValueError(f"Louvain map contains duplicate spot IDs: {map_path}")
    observed = set(table["spot_id"])
    expected = set(map(str, spot_order))
    if observed != expected:
        raise ValueError(f"Louvain map spot set differs (missing={sorted(expected-observed)[:5]}, extra={sorted(observed-expected)[:5]})")
    aligned = table.set_index("spot_id").loc[list(map(str, spot_order)), "domain_id"]
    domain_ids = list(map(str, domain_order)) if domain_order is not None else sorted(aligned.unique().tolist())
    if set(domain_ids) != set(aligned.unique()):
        raise ValueError("Louvain map domains do not match the macro CCI index order.")
    positions = {domain: index for index, domain in enumerate(domain_ids)}
    assignment = np.zeros((len(aligned), len(domain_ids)), dtype=np.float32)
    assignment[np.arange(len(aligned)), [positions[value] for value in aligned]] = 1.0
    if not np.allclose(assignment.sum(axis=1), 1.0):
        raise AssertionError("Hard Louvain assignment must have exactly one membership per spot.")
    return assignment, domain_ids
