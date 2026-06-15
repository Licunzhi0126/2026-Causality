from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence

import mignet_ce.embeddings as embeddings


def test_laplacian_embedding_uses_dense_solver_for_small_graph(monkeypatch) -> None:
    adjacency = sp.csr_matrix(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    def fail_eigsh(*args, **kwargs):
        raise AssertionError("small laplacian graphs should use the dense solver")

    monkeypatch.setattr(embeddings, "eigsh", fail_eigsh)

    result = embeddings.laplacian_embedding(adjacency, n_components=2)

    assert result.shape == (4, 2)
    assert np.isfinite(result).all()


def test_laplacian_embedding_falls_back_to_dense_when_arpack_does_not_converge(monkeypatch) -> None:
    adjacency = sp.diags([1.0], [1], shape=(12, 12), format="csr")
    adjacency = adjacency + adjacency.T

    def fail_eigsh(*args, **kwargs):
        raise ArpackNoConvergence("no convergence", np.array([]), np.empty((12, 0)))

    monkeypatch.setattr(embeddings, "_LAPLACIAN_DENSE_EIGEN_THRESHOLD", 0)
    monkeypatch.setattr(embeddings, "eigsh", fail_eigsh)

    result = embeddings.laplacian_embedding(adjacency, n_components=5)

    assert result.shape == (12, 5)
    assert np.isfinite(result).all()


def test_laplacian_embedding_uses_partial_arpack_result_when_available(monkeypatch) -> None:
    adjacency = sp.diags([1.0], [1], shape=(12, 12), format="csr")
    adjacency = adjacency + adjacency.T
    partial_vectors = np.zeros((12, 2), dtype=float)
    partial_vectors[:, 0] = 1.0
    partial_vectors[::2, 1] = 1.0
    partial_vectors[1::2, 1] = -1.0

    def partial_eigsh(*args, **kwargs):
        raise ArpackNoConvergence("partial convergence", np.array([0.0, 0.5]), partial_vectors)

    monkeypatch.setattr(embeddings, "_LAPLACIAN_DENSE_EIGEN_THRESHOLD", 0)
    monkeypatch.setattr(embeddings, "_LAPLACIAN_DENSE_FALLBACK_THRESHOLD", 0)
    monkeypatch.setattr(embeddings, "eigsh", partial_eigsh)

    result = embeddings.laplacian_embedding(adjacency, n_components=5)

    assert result.shape == (12, 5)
    assert np.isfinite(result).all()
    assert np.count_nonzero(result[:, 0]) == 12
    assert np.count_nonzero(result[:, 1:]) == 0
