from __future__ import annotations

import importlib

import numpy as np

from mignet_ce.utils.matrix import safe_row_normalize


def _clean_cost(cost: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D cost matrix, got shape {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    max_finite = float(np.max(arr[finite]))
    min_finite = float(np.min(arr[finite]))
    out = np.where(finite, arr, max_finite)
    out = out - min_finite
    return np.maximum(out, 0.0)


def _kernel_from_cost(cost: np.ndarray, epsilon: float) -> np.ndarray:
    if cost.size == 0:
        return cost.copy()
    return np.exp(-cost / float(epsilon))


def _balanced_sinkhorn(kernel: np.ndarray, max_iter: int, eps: float = 1e-12) -> np.ndarray:
    n_source, n_target = kernel.shape
    if n_source == 0 or n_target == 0:
        return kernel.copy()
    a = np.full(n_source, 1.0 / n_source, dtype=float)
    b = np.full(n_target, 1.0 / n_target, dtype=float)
    u = np.ones(n_source, dtype=float)
    v = np.ones(n_target, dtype=float)
    for _ in range(max(0, int(max_iter))):
        u = a / (kernel @ v + eps)
        v = b / (kernel.T @ u + eps)
    return (u[:, None] * kernel) * v[None, :]


def _unbalanced_sinkhorn(
    kernel: np.ndarray,
    epsilon: float,
    mass_reg: float,
    max_iter: int,
    eps: float = 1e-12,
) -> np.ndarray:
    n_source, n_target = kernel.shape
    if n_source == 0 or n_target == 0:
        return kernel.copy()
    a = np.full(n_source, 1.0 / n_source, dtype=float)
    b = np.full(n_target, 1.0 / n_target, dtype=float)
    u = np.ones(n_source, dtype=float)
    v = np.ones(n_target, dtype=float)
    power = float(mass_reg) / (float(mass_reg) + float(epsilon))
    for _ in range(max(0, int(max_iter))):
        v = np.power(b / (kernel.T @ u + eps), power)
        u = np.power(a / (kernel @ v + eps), power)
    return (u[:, None] * kernel) * v[None, :]


def _pot_sinkhorn(
    cost: np.ndarray,
    epsilon: float,
    unbalanced: bool,
    mass_reg: float,
    max_iter: int,
) -> np.ndarray | None:
    try:
        pot = importlib.import_module("ot")
    except ImportError:
        return None

    n_source, n_target = cost.shape
    a = np.full(n_source, 1.0 / n_source, dtype=float)
    b = np.full(n_target, 1.0 / n_target, dtype=float)
    try:
        if unbalanced:
            unbalanced_mod = getattr(pot, "unbalanced")
            solver = getattr(unbalanced_mod, "sinkhorn_knopp_unbalanced", None)
            if solver is None:
                solver = getattr(unbalanced_mod, "sinkhorn_unbalanced")
            return np.asarray(
                solver(
                    a,
                    b,
                    cost,
                    reg=float(epsilon),
                    reg_m=float(mass_reg),
                    numItermax=int(max_iter),
                ),
                dtype=float,
            )
        return np.asarray(
            pot.sinkhorn(
                a,
                b,
                cost,
                reg=float(epsilon),
                numItermax=int(max_iter),
            ),
            dtype=float,
        )
    except Exception:
        return None


def build_entropic_ot_kernel(
    cost: np.ndarray,
    epsilon: float,
    use_pot: bool = True,
    unbalanced: bool = False,
    mass_reg: float = 1.0,
    max_iter: int = 200,
) -> np.ndarray:
    """Build a row-stochastic transition matrix from an OT cost matrix."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if mass_reg <= 0:
        raise ValueError("mass_reg must be positive.")
    clean_cost = _clean_cost(cost)
    n_source, n_target = clean_cost.shape
    if n_source == 0 or n_target == 0:
        return np.zeros((n_source, n_target), dtype=float)

    transport = None
    if use_pot:
        transport = _pot_sinkhorn(
            clean_cost,
            epsilon=epsilon,
            unbalanced=unbalanced,
            mass_reg=mass_reg,
            max_iter=max_iter,
        )
    if transport is None:
        kernel = _kernel_from_cost(clean_cost, epsilon=epsilon)
        if unbalanced:
            transport = _unbalanced_sinkhorn(
                kernel,
                epsilon=epsilon,
                mass_reg=mass_reg,
                max_iter=max_iter,
            )
        else:
            transport = _balanced_sinkhorn(kernel, max_iter=max_iter)
    return safe_row_normalize(transport)
