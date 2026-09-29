from __future__ import annotations

import numpy as np
import torch

from mignet_ce.pij.compare.compare_NR_fgw_regsim_v9 import regsim_v9_pij_torch


def test_regsim_v9_macro_rule_has_finite_gradient() -> None:
    rng = np.random.default_rng(19)
    n_t = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32, requires_grad=True)
    output = regsim_v9_pij_torch(
        n_t,
        torch.tensor(rng.normal(size=(5, 3)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(4, 2)), dtype=torch.float32),
        torch.tensor(rng.normal(size=(5, 2)), dtype=torch.float32),
        torch.tensor(rng.random((4, 4)), dtype=torch.float32),
        torch.tensor(rng.random((5, 5)), dtype=torch.float32),
        outer_iterations=2,
        sinkhorn_iterations=16,
    )
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    assert n_t.grad is not None
    assert torch.isfinite(n_t.grad).all()
