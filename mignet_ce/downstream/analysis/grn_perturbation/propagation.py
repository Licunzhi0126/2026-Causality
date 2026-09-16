"""Frozen-assignment two-path propagation primitives."""
from __future__ import annotations
import numpy as np

EPS = 1e-12

def horizontal(delta: np.ndarray, spot_pij: np.ndarray) -> np.ndarray:
    return np.asarray(spot_pij).T @ np.asarray(delta)

def path_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    d = np.asarray(left) - np.asarray(right)
    norm = max(float(np.abs(right).sum()), EPS)
    den = max(float(np.linalg.norm(left) * np.linalg.norm(right)), EPS)
    return {"path_delta_l1": float(np.abs(d).sum()), "path_delta_frobenius": float(np.linalg.norm(d)), "path_delta_relative_l1": float(np.abs(d).sum() / norm), "path_delta_cosine": float(np.sum(left * right) / den)}

