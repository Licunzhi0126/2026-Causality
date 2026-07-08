from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.pij.compare.distances import pairwise_cosine_distance


@dataclass
class SparseOTResult:
    candidate_edges: pd.DataFrame
    cost_sparse: sp.csr_matrix
    transport_sparse: sp.csr_matrix
    pij_row_normalized_sparse: sp.csr_matrix
    source_mass_diagnostics: pd.DataFrame
    convergence: Dict[str, object]


def _topk_candidates(cost: np.ndarray, source_k: int, target_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_source, n_target = cost.shape
    if n_source == 0 or n_target == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    source_k = max(1, min(int(source_k), n_target))
    target_k = max(1, min(int(target_k), n_source))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    source_cols = np.argpartition(cost, kth=source_k - 1, axis=1)[:, :source_k]
    source_rows = np.repeat(np.arange(n_source, dtype=int), source_k)
    rows.append(source_rows)
    cols.append(source_cols.reshape(-1).astype(int))

    target_rows = np.argpartition(cost, kth=target_k - 1, axis=0)[:target_k, :]
    rows.append(target_rows.reshape(-1).astype(int))
    cols.append(np.tile(np.arange(n_target, dtype=int), target_k))

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    encoded = row.astype(np.int64) * np.int64(max(1, n_target)) + col.astype(np.int64)
    _, keep = np.unique(encoded, return_index=True)
    return row[keep], col[keep]


def _normalize_cost(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(arr, dtype=float)
    return (arr - vmin) / (vmax - vmin)


def _row_normalize_sparse(matrix: sp.csr_matrix) -> sp.csr_matrix:
    csr = matrix.tocsr(copy=True)
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    if csr.nnz:
        repeats = np.diff(csr.indptr)
        scale = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums > 0)
        csr.data *= np.repeat(scale, repeats)
    return csr


def run_sparse_semi_relaxed_ot(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    epsilon: float,
    gamma: float,
    max_iter: int,
    source_k: int,
    target_k: int,
    tol: float = 1e-6,
) -> SparseOTResult:
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    dense_cost = pairwise_cosine_distance(source, target)
    n_source, n_target = dense_cost.shape
    row, col = _topk_candidates(dense_cost, source_k=source_k, target_k=target_k)
    raw_values = dense_cost[row, col] if row.size else np.array([], dtype=float)
    cost_values = _normalize_cost(raw_values)
    cost_sparse = sp.coo_matrix((cost_values, (row, col)), shape=(n_source, n_target), dtype=float).tocsr()

    if n_source == 0 or n_target == 0:
        empty = sp.csr_matrix((n_source, n_target), dtype=float)
        return SparseOTResult(
            candidate_edges=pd.DataFrame(columns=["source_index", "target_index", "raw_cosine_distance", "normalized_cost"]),
            cost_sparse=empty,
            transport_sparse=empty,
            pij_row_normalized_sparse=empty,
            source_mass_diagnostics=pd.DataFrame(columns=["source_index", "target_mass", "source_prior", "mass_ratio"]),
            convergence={"iterations": 0, "converged": True, "reason": "empty_matrix"},
        )

    if cost_sparse.nnz == 0:
        raise ValueError("Sparse OT candidate set is empty.")
    if epsilon <= 0 or gamma <= 0:
        raise ValueError("epsilon and gamma must be positive.")

    kernel = cost_sparse.copy()
    kernel.data = np.exp(-kernel.data / float(epsilon))
    kernel.data = np.maximum(kernel.data, 1e-300)
    a = np.full(n_source, 1.0 / n_source, dtype=float)
    b = np.full(n_target, 1.0 / n_target, dtype=float)
    u = np.ones(n_source, dtype=float)
    v = np.ones(n_target, dtype=float)
    row_power = float(gamma) / (float(gamma) + float(epsilon))
    converged = False
    col_l1 = np.inf
    row_l1 = np.inf

    for iteration in range(1, int(max_iter) + 1):
        kv = np.asarray(kernel @ v).ravel()
        u = np.power(np.divide(a, kv, out=np.ones_like(a), where=kv > 0), row_power)
        ktu = np.asarray(kernel.T @ u).ravel()
        v = np.divide(b, ktu, out=np.zeros_like(b), where=ktu > 0)

        if iteration == 1 or iteration == int(max_iter) or iteration % 10 == 0:
            transport = kernel.multiply(u[:, None]).multiply(v[None, :]).tocsr()
            col_mass = np.asarray(transport.sum(axis=0)).ravel()
            row_mass = np.asarray(transport.sum(axis=1)).ravel()
            col_l1 = float(np.sum(np.abs(col_mass - b)))
            row_l1 = float(np.sum(np.abs(row_mass - a)))
            if col_l1 <= tol:
                converged = True
                break

    transport = kernel.multiply(u[:, None]).multiply(v[None, :]).tocsr()
    pij = _row_normalize_sparse(transport)
    row_mass = np.asarray(transport.sum(axis=1)).ravel()
    candidate_edges = pd.DataFrame(
        {
            "source_index": row.astype(int),
            "target_index": col.astype(int),
            "raw_cosine_distance": raw_values.astype(float),
            "normalized_cost": cost_values.astype(float),
        }
    )
    source_mass = pd.DataFrame(
        {
            "source_index": np.arange(n_source, dtype=int),
            "target_mass": row_mass,
            "source_prior": a,
            "mass_ratio": np.divide(row_mass, a, out=np.zeros_like(row_mass), where=a > 0),
        }
    )
    convergence = {
        "iterations": int(iteration),
        "converged": bool(converged),
        "column_l1": float(col_l1),
        "row_l1": float(row_l1),
        "epsilon": float(epsilon),
        "gamma": float(gamma),
        "source_k": int(source_k),
        "target_k": int(target_k),
        "candidate_edges": int(cost_sparse.nnz),
        "cost_source": "cosine_distance_on_current_compare_features",
    }
    return SparseOTResult(
        candidate_edges=candidate_edges,
        cost_sparse=cost_sparse,
        transport_sparse=transport,
        pij_row_normalized_sparse=pij,
        source_mass_diagnostics=source_mass,
        convergence=convergence,
    )
