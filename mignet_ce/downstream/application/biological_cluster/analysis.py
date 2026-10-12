"""Dataset-specific transition-table formatting without altering probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def transition_table(pij: np.ndarray, source_metadata: pd.DataFrame, target_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source = source_metadata.reset_index(drop=True)
    target = target_metadata.reset_index(drop=True)
    for source_index, source_row in source.iterrows():
        order = np.argsort(-np.asarray(pij[source_index], dtype=float), kind="stable")
        for rank, target_index in enumerate(order, start=1):
            target_row = target.iloc[int(target_index)]
            rows.append({"source_cluster": source_row["cluster"], "source_cell_type": source_row["cell_type"], "target_cluster": target_row["cluster"], "target_cell_type": target_row["cell_type"], "probability": float(pij[source_index, target_index]), "rank": rank})
    return pd.DataFrame(rows)


def contribution_table(contributions: np.ndarray, source_metadata: pd.DataFrame) -> pd.DataFrame:
    result = source_metadata.copy().reset_index(drop=True)
    result["state_level_ei_contribution"] = np.asarray(contributions, dtype=float)
    return result

