from __future__ import annotations

import torch

from mignet_ce.downstream.perturbation.direct_state_bridge import _pool, build_direct_state_macro_pij_builder
from mignet_ce.representations.coarse_input import MacroPijInputs


def test_direct_macro_pooling_retains_target_perturbation() -> None:
    assignment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    baseline = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbed = baseline + torch.tensor([[0.0, 0.0], [5.0, -2.0]])
    assert not torch.allclose(_pool(baseline, assignment), _pool(perturbed, assignment))


def test_direct_macro_pij_uses_pooled_grn_state() -> None:
    builder = build_direct_state_macro_pij_builder()
    n = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    baseline = builder(MacroPijInputs(
        z_macro_t=n, z_macro_tp=n, network_macro_t=None, network_macro_tp=None,
        feature_blocks_t={"N": n, "G": n}, feature_blocks_tp={"N": n, "G": n},
    ))
    changed_g = torch.tensor([[10.0, -5.0], [-3.0, 7.0]])
    perturbed = builder(MacroPijInputs(
        z_macro_t=n, z_macro_tp=n, network_macro_t=None, network_macro_tp=None,
        feature_blocks_t={"N": n, "G": n}, feature_blocks_tp={"N": n, "G": changed_g},
    ))
    assert not torch.allclose(baseline, perturbed)
