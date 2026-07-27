from __future__ import annotations

import numpy as np
import torch

from mignet_ce.coarse_frontends._complete_combined_core import (
    native_v7_pij_numpy,
)
from mignet_ce.pij.compare.native_v7_torch import native_v7_pij_torch


def test_native_v7_torch_matches_numpy_and_has_gradients() -> None:
    rng = np.random.default_rng(20260727)
    n_t_np = rng.normal(size=(5, 4))
    n_tp_np = rng.normal(size=(6, 4))
    g_t_np = rng.normal(size=(5, 7))
    g_tp_np = rng.normal(size=(6, 7))
    _, expected, _ = native_v7_pij_numpy(n_t_np, n_tp_np, g_t_np, g_tp_np)

    n_t = torch.tensor(n_t_np, dtype=torch.float64, requires_grad=True)
    n_tp = torch.tensor(n_tp_np, dtype=torch.float64, requires_grad=True)
    g_t = torch.tensor(g_t_np, dtype=torch.float64, requires_grad=True)
    g_tp = torch.tensor(g_tp_np, dtype=torch.float64, requires_grad=True)
    actual = native_v7_pij_torch(
        n_t,
        n_tp,
        g_t,
        g_tp,
        sinkhorn_iterations=512,
    )

    np.testing.assert_allclose(
        actual.detach().numpy(),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        actual.detach().numpy().sum(axis=1),
        np.ones(5),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual.detach().numpy().mean(axis=0),
        np.full(6, 1.0 / 6.0),
        atol=1e-8,
    )
    loss = (actual * torch.arange(6, dtype=actual.dtype)).sum()
    loss.backward()
    for values in (n_t, n_tp, g_t, g_tp):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
