"""Artifact writers for fixed biological cluster evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_outputs(root: Path, *, pij: np.ndarray, source_metadata: pd.DataFrame, target_metadata: pd.DataFrame, transitions: pd.DataFrame, contributions: pd.DataFrame, causal: dict[str, object], closure: dict[str, object], provenance: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "PIJ_cluster.npy", pij)
    pd.DataFrame(pij, index=source_metadata["cluster"], columns=target_metadata["cluster"]).to_csv(root / "PIJ_cluster.csv")
    transitions.to_csv(root / "cluster_transition_long.csv", index=False)
    source_metadata.to_csv(root / "source_clusters.csv", index=False)
    target_metadata.to_csv(root / "target_clusters.csv", index=False)
    contributions.to_csv(root / "cluster_ei_contributions.csv", index=False)
    for name, payload in (("causal_emergence_summary.json", causal), ("dynamical_closure_summary.json", closure), ("provenance.json", provenance)):
        (root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

