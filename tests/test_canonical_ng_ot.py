from __future__ import annotations

import numpy as np
import torch

from mignet_ce.pij.compare._shared.ng_kl_ot import (
    CANONICAL_ALPHA_CCI,
    CANONICAL_TEMPERATURE,
    CANONICAL_TRANSITION_PROTOCOL,
    build_canonical_ng_cost_numpy,
    build_canonical_ng_cost_torch,
    build_ng_component_costs_numpy,
    canonical_ng_pij_numpy,
    canonical_ng_pij_torch,
    mix_ng_cost_numpy,
)


def _features():
    rng = np.random.default_rng(20260920)
    return (
        rng.normal(size=(5, 4)),
        rng.normal(size=(6, 4)),
        rng.normal(size=(5, 7)),
        rng.normal(size=(6, 7)),
    )


def test_canonical_numpy_torch_cost_and_pij_parity() -> None:
    n_t, n_tp, g_t, g_tp = _features()
    numpy_cost, metadata = build_canonical_ng_cost_numpy(n_t, n_tp, g_t, g_tp)
    _, numpy_pij, pij_metadata = canonical_ng_pij_numpy(n_t, n_tp, g_t, g_tp)
    tensors = [torch.tensor(value, dtype=torch.float64) for value in (n_t, n_tp, g_t, g_tp)]
    torch_cost = build_canonical_ng_cost_torch(*tensors)
    torch_pij = canonical_ng_pij_torch(*tensors)

    np.testing.assert_allclose(torch_cost.detach().numpy(), numpy_cost, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(torch_pij.detach().numpy(), numpy_pij, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(numpy_pij.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(numpy_pij.mean(axis=0), 1.0 / numpy_pij.shape[1], atol=1e-8)
    assert metadata["alpha_cci"] == CANONICAL_ALPHA_CCI
    assert pij_metadata["tau"] == CANONICAL_TEMPERATURE
    assert CANONICAL_ALPHA_CCI == 0.01
    assert CANONICAL_TEMPERATURE == 0.8
    assert CANONICAL_TRANSITION_PROTOCOL == "canonical_ng_rawkl_v2"
    assert metadata["component_cost"] == "raw_pairwise_feature_KL"
    assert metadata["component_normalization"] == "none"
    assert metadata["combined_scale_control"] == "none"


def test_alpha_endpoints_use_only_the_requested_raw_component() -> None:
    n_t, n_tp, g_t, g_tp = _features()
    d_n, d_g, _ = build_ng_component_costs_numpy(n_t, n_tp, g_t, g_tp)
    pure_g, g_metadata = mix_ng_cost_numpy(d_n, d_g, alpha_cci=0.0)
    pure_n, n_metadata = mix_ng_cost_numpy(d_n, d_g, alpha_cci=1.0)
    np.testing.assert_allclose(pure_g, d_g)
    np.testing.assert_allclose(pure_n, d_n)
    assert g_metadata["nominal_cci_weight"] == 0.0
    assert n_metadata["nominal_grn_weight"] == 0.0
    assert g_metadata["combined_scale_control"] == "none"
    assert n_metadata["combined_scale_control"] == "none"


def test_combined_cost_is_raw_convex_fusion_without_clipping_or_rescaling() -> None:
    d_g = np.linspace(0.0, 5.0, 100).reshape(10, 10)
    d_n = np.square(d_g)
    controlled, metadata = mix_ng_cost_numpy(d_n, d_g, alpha_cci=0.1)
    np.testing.assert_allclose(controlled, 0.9 * d_g + 0.1 * d_n)
    assert controlled.max() > 1.0
    assert metadata["combined_cost_clipped"] is False
    assert metadata["combined_scale_control"] == "none"

    degenerate, degenerate_metadata = mix_ng_cost_numpy(
        np.zeros((3, 4)),
        np.zeros((3, 4)),
        alpha_cci=0.4,
    )
    np.testing.assert_array_equal(degenerate, np.zeros((3, 4)))
    assert degenerate_metadata["combined_scale_control"] == "none"


def test_canonical_torch_path_remains_differentiable() -> None:
    arrays = _features()
    tensors = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in arrays]
    pij = canonical_ng_pij_torch(*tensors)
    loss = (pij * torch.arange(pij.shape[1], dtype=pij.dtype)).sum()
    loss.backward()
    for values in tensors:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
