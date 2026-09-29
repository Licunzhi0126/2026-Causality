"""Load a cached spot PIJ or delegate its construction to the shared adapter."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..common.spot_reference import load_spot_reference


def load_spot_dynamics(cache_dir: Path) -> dict[str, object]:
    """Load the workflow-wide spot reference without recomputing dynamics."""
    reference = load_spot_reference(cache_dir)
    pij = np.asarray(reference["pij"], dtype=float)
    if pij.shape != (len(reference["source_units"]), len(reference["target_units"])):
        raise ValueError("Spot PIJ shape does not match its stored spot orders.")
    if not np.isfinite(pij).all() or np.any(pij < 0.0) or not np.allclose(pij.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Spot PIJ must be finite, non-negative, and row-stochastic.")
    return reference
