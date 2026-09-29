"""Validation helpers for immutable two-level perturbation batches."""
from __future__ import annotations
import numpy as np

def require_finite(frame) -> None:
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("GRN perturbation table contains non-finite numeric values")

