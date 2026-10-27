from __future__ import annotations

from typing import TextIO
import numpy as np
import scipy.sparse as sp
import torch


def preprocess_route_a(source: np.ndarray, target: np.ndarray, local_dims: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(source, dtype=np.float32).copy(), np.asarray(target, dtype=np.float32).copy()
    combined = np.vstack([left, right])
    if local_dims > 0:
        width = min(int(local_dims), combined.shape[1]); local = combined[:, :width]
        mean, std = local.mean(axis=0, keepdims=True), local.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0; combined[:, :width] = (local - mean) / std
    return combined[:left.shape[0]], combined[left.shape[0]:]


def _torch_sparse(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    csr = matrix.tocsr().astype(np.float32); total = int(csr.shape[0]) * int(csr.shape[1])
    if (float(csr.nnz / total) if total else 0.0) >= .20:
        return torch.tensor(csr.toarray(), dtype=torch.float32, device=device)
    coo = csr.tocoo(); indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, dtype=torch.float32, device=device, check_invariants=True).coalesce()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _log(handle: TextIO, message: str) -> None:
    print(message); handle.write(message + "\n"); handle.flush()
