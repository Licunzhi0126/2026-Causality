from __future__ import annotations

import numpy as np
import torch

from mignet_ce.pij.compare.compare_NR_fgw_regsim_v9 import regsim_v9_pij_torch
from mignet_ce.pij.compare.compare_NR_kl_sinkhorn_regsim_v7 import (
    regsim_v7_pij_numpy,
    regsim_v7_pij_torch,
)
from mignet_ce.pij.wyt_single_kl import single_kl_pij_numpy, single_kl_pij_torch


def test_single_kl_numpy_and_torch_match() -> None:
    source = np.asarray([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    target = np.asarray([[0.9, 0.1], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    _, numpy_pij = single_kl_pij_numpy(source, target, temperature=0.7)
    torch_pij = single_kl_pij_torch(
        torch.tensor(source),
        torch.tensor(target),
        temperature=0.7,
    )
    np.testing.assert_allclose(numpy_pij, torch_pij.numpy(), atol=1e-6)
    np.testing.assert_allclose(numpy_pij.sum(axis=1), 1.0, atol=1e-7)


def test_regsim_v7_is_balanced_and_torch_is_differentiable() -> None:
    rng = np.random.default_rng(3)
    n_t = rng.normal(size=(4, 3))
    n_tp = rng.normal(size=(5, 3))
    r_t = rng.normal(size=(4, 2))
    r_tp = rng.normal(size=(5, 2))
    joint, conditional, _, _ = regsim_v7_pij_numpy(n_t, n_tp, r_t, r_tp)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(joint.sum(axis=1), 0.25, atol=1e-5)
    np.testing.assert_allclose(joint.sum(axis=0), 0.20, atol=1e-5)
    n_source = torch.tensor(n_t, dtype=torch.float32, requires_grad=True)
    output = regsim_v7_pij_torch(
        n_source,
        torch.tensor(n_tp, dtype=torch.float32),
        torch.tensor(r_t, dtype=torch.float32),
        torch.tensor(r_tp, dtype=torch.float32),
    )
    output.square().sum().backward()
    assert n_source.grad is not None
    assert torch.isfinite(n_source.grad).all()


def test_regsim_v9_torch_is_row_stochastic() -> None:
    rng = np.random.default_rng(9)
    args = [
        torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(5, 3)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(4, 2)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(5, 2)), dtype=torch.float32),
        torch.tensor(rng.random((4, 4)), dtype=torch.float32),
        torch.tensor(rng.random((5, 5)), dtype=torch.float32),
    ]
    output = regsim_v9_pij_torch(*args, outer_iterations=2, sinkhorn_iterations=16)
    assert output.shape == (4, 5)
    torch.testing.assert_close(output.sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)
