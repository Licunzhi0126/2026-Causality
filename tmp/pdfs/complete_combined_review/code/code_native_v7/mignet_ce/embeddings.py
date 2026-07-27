from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from mignet_ce.graph.builder import LayerGraph


_LAPLACIAN_DENSE_EIGEN_THRESHOLD = 512
_LAPLACIAN_DENSE_FALLBACK_THRESHOLD = 2048
_LAPLACIAN_ARPACK_TOL = 1e-6
_LAPLACIAN_ARPACK_MIN_MAXITER = 5000
_LAPLACIAN_ARPACK_MAXITER_FACTOR = 20


def graph_to_adjacency(
    graph: LayerGraph,
    units: Sequence[str] | None = None,
    weight_col: str = "influence_score",
) -> sp.csr_matrix:
    units = list(map(str, units if units is not None else graph.units))
    unit_index = {unit: idx for idx, unit in enumerate(units)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for edge_table in (graph.intra_edges, graph.inter_edges):
        if edge_table.empty:
            continue
        required = {"src_unit", "dst_unit", weight_col}
        if not required.issubset(edge_table.columns):
            continue
        work = edge_table.loc[:, ["src_unit", "dst_unit", weight_col]].copy()
        work["src_unit"] = work["src_unit"].astype(str)
        work["dst_unit"] = work["dst_unit"].astype(str)
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
        work = work.dropna(subset=[weight_col])
        work = work[work["src_unit"].isin(unit_index) & work["dst_unit"].isin(unit_index)]
        if work.empty:
            continue
        grouped = work.groupby(["src_unit", "dst_unit"], as_index=False)[weight_col].sum()
        for edge in grouped.itertuples(index=False):
            value = float(getattr(edge, weight_col))
            if value <= 0:
                continue
            rows.append(unit_index[str(edge.src_unit)])
            cols.append(unit_index[str(edge.dst_unit)])
            data.append(value)

    shape = (len(units), len(units))
    return sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=float).tocsr()


def _dense_laplacian_eigh(laplacian: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(laplacian.toarray())


def _solve_laplacian_eigenvectors(laplacian: sp.spmatrix, eig_count: int) -> tuple[np.ndarray, np.ndarray]:
    n_units = laplacian.shape[0]
    if eig_count >= n_units or n_units <= _LAPLACIAN_DENSE_EIGEN_THRESHOLD:
        return _dense_laplacian_eigh(laplacian)

    try:
        return eigsh(
            laplacian,
            k=eig_count,
            which="SM",
            tol=_LAPLACIAN_ARPACK_TOL,
            maxiter=max(_LAPLACIAN_ARPACK_MIN_MAXITER, _LAPLACIAN_ARPACK_MAXITER_FACTOR * n_units),
        )
    except ArpackNoConvergence as exc:
        eigvals = getattr(exc, "eigenvalues", None)
        eigvecs = getattr(exc, "eigenvectors", None)
        if eigvals is not None and eigvecs is not None:
            eigvals = np.asarray(eigvals, dtype=float)
            eigvecs = np.asarray(eigvecs, dtype=float)
            if eigvecs.ndim == 1:
                eigvecs = eigvecs[:, None]
            if eigvals.size > 0 and eigvecs.shape[0] == n_units and eigvecs.shape[1] > 0:
                keep = min(eigvals.size, eigvecs.shape[1])
                return eigvals[:keep], eigvecs[:, :keep]
        if n_units <= _LAPLACIAN_DENSE_FALLBACK_THRESHOLD:
            return _dense_laplacian_eigh(laplacian)
        raise


def laplacian_embedding(
    adjacency: sp.spmatrix | np.ndarray,
    n_components: int = 5,
    normalized: bool = True,
) -> np.ndarray:
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if not sp.issparse(adjacency):
        adjacency = sp.csr_matrix(adjacency)
    else:
        adjacency = adjacency.tocsr()

    n_units = adjacency.shape[0]
    if n_units == 0:
        return np.zeros((0, n_components), dtype=float)
    if n_units == 1 or adjacency.nnz == 0:
        return np.zeros((n_units, n_components), dtype=float)

    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if normalized:
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degree + 1e-12))
        laplacian = sp.eye(n_units, format="csr") - d_inv_sqrt @ adjacency @ d_inv_sqrt
    else:
        laplacian = sp.diags(degree) - adjacency

    eig_count = min(n_units, n_components + 1)
    eigvals, eigvecs = _solve_laplacian_eigenvectors(laplacian, eig_count)

    order = np.argsort(eigvals)
    eigvecs = np.asarray(eigvecs[:, order], dtype=float)
    embedding = eigvecs[:, 1 : n_components + 1]
    if embedding.shape[1] < n_components:
        pad = np.zeros((n_units, n_components - embedding.shape[1]), dtype=float)
        embedding = np.hstack([embedding, pad])

    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = np.divide(embedding, norms, out=np.zeros_like(embedding), where=norms > 0)
    return embedding


def layer_graph_laplacian_features(
    graph: LayerGraph,
    n_components: int = 5,
    normalized: bool = True,
) -> np.ndarray:
    adjacency = graph_to_adjacency(graph, units=graph.units)
    return laplacian_embedding(adjacency, n_components=n_components, normalized=normalized)
