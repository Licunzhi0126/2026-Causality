from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def build_knn_adjacency(features: np.ndarray, k: int, local_dims: int = 0) -> sp.csr_matrix:
    from sklearn.neighbors import NearestNeighbors
    values = np.asarray(features, dtype=np.float32); local = values[:, :local_dims] if local_dims > 0 else values
    count = local.shape[0]; effective = min(int(k) + 1, count)
    indices = NearestNeighbors(n_neighbors=effective, metric="euclidean").fit(local).kneighbors(local)[1]
    rows = np.repeat(np.arange(count), max(0, effective - 1)); columns = indices[:, 1:].reshape(-1)
    adjacency = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, columns)), shape=(count, count)).maximum(sp.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, columns)), shape=(count, count)).T).tolil()
    adjacency.setdiag(1.0); adjacency = adjacency.tocsr(); row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(1.0 / np.maximum(row_sum, 1e-12)) @ adjacency).tocsr()
