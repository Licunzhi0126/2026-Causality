"""Fixed source-GRN NMF and Laplacian-HKS bases.

Implements the Round-6 reference algorithms; both factor families are
fit only on the source network. No target-edge information enters the fit.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def fit_nmf_basis(adjacency: sp.spmatrix, *, rank: int = 6, steps: int = 65, seed: int = 42) -> tuple[np.ndarray,np.ndarray]:
    matrix = adjacency.tocsr().astype(np.float64)
    if matrix.nnz and matrix.data.min() < 0:
        raise ValueError("Source GRN NMF requires nonnegative weights.")
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    u = rng.random((n, rank)) + 0.01
    v = rng.random((n, rank)) + 0.01
    eps=1e-10
    for _ in range(steps):
        u *= (matrix @ v + eps) / (u @ (v.T @ v) + eps)
        v *= (matrix.T @ u + eps) / (v @ (u.T @ u) + eps)
    u=np.nan_to_num(u, nan=0.,posinf=0.,neginf=0.)
    v=np.nan_to_num(v, nan=0.,posinf=0.,neginf=0.)
    return u.astype(np.float32), v.astype(np.float32)

def fit_lap_hks_basis(adjacency: sp.spmatrix, *, rank: int = 8) -> np.ndarray:
    matrix=adjacency.tocsr().astype(np.float64)
    n=matrix.shape[0]
    if n < 4:
        raise ValueError("Laplacian HKS requires at least 4 genes.")
    a=(matrix+matrix.T)*0.5
    degree=np.asarray(a.sum(axis=1)).ravel()
    inv=1.0/np.sqrt(np.maximum(degree,1e-12))
    norm=(sp.diags(inv) @ a @ sp.diags(inv)).tocsr()
    k=min(rank+2,n-2)
    eig, vec=eigsh(norm,k=k,which='LA',tol=5e-4,maxiter=1500,v0=np.ones(n))
    order=np.argsort(-eig)
    eig=eig[order];vec=vec[:,order]
    if len(eig)>rank:
        eig=eig[1:rank+1]; vec=vec[:,1:rank+1]
    times=np.array([0.5,1.0,2.0,4.0,8.0])
    heat=(vec*vec)@np.exp(-np.outer(1.0-eig,times))
    return np.nan_to_num(heat).astype(np.float32)
