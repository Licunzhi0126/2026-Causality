from __future__ import annotations

import numpy as np
import torch

from mignet_ce.pij.compare.compare_NR_kl_sinkhorn_regsim_v7 import (
    regsim_v7_pij_numpy,
    regsim_v7_pij_torch,
)


def test_regsim_v7_micro_and_macro_rules_are_row_stochastic() -> None:
    rng = np.random.default_rng(17)
    n_t = rng.normal(size=(6, 4))
    n_tp = rng.normal(size=(7, 4))
    r_t = rng.normal(size=(6, 5))
    r_tp = rng.normal(size=(7, 5))
    joint, conditional, _, _ = regsim_v7_pij_numpy(n_t, n_tp, r_t, r_tp)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(joint.sum(axis=1), 1.0 / 6.0, atol=1e-5)
    np.testing.assert_allclose(joint.sum(axis=0), 1.0 / 7.0, atol=1e-5)
    macro = regsim_v7_pij_torch(
        torch.tensor(n_t[:3], dtype=torch.float32),
        torch.tensor(n_tp[:3], dtype=torch.float32),
        torch.tensor(r_t[:3], dtype=torch.float32),
        torch.tensor(r_tp[:3], dtype=torch.float32),
    )
    torch.testing.assert_close(macro.sum(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)
