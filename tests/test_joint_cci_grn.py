from __future__ import annotations

import numpy as np

from mignet_ce.networks.joint_cci_grn import collective_joint_nmf_pair, module_double_end_gate


def test_module_double_end_gate_has_expected_dimensions_and_values() -> None:
    expression = np.array([[2.0, 3.0], [1.0, 4.0]])
    q_reg = np.array([[1.0, 0.0], [0.0, 1.0]])
    q_tar = np.array([[0.5, 0.5], [0.25, 0.75]])
    core = np.array([[2.0, 0.0], [0.0, 1.0]])

    a_reg, a_tar, g_reg, g_tar = module_double_end_gate(expression, q_reg, q_tar, core)

    assert a_reg.shape == a_tar.shape == g_reg.shape == g_tar.shape == (2, 2)
    assert np.allclose(a_reg, expression)
    assert np.allclose(g_reg, a_reg * (a_tar @ core.T))
    assert np.allclose(g_tar, a_tar * (a_reg @ core))


def test_collective_joint_nmf_has_finite_three_term_losses_and_expected_factor_shapes() -> None:
    source_cci = np.array([[1.0, 2.0, 0.5], [0.7, 1.0, 3.0], [2.5, 0.4, 1.0]])
    target_cci = np.array([[1.0, 1.5, 0.8], [1.2, 1.0, 2.5], [2.0, 0.9, 1.0]])
    g_reg_source = np.array([[2.0, 1.0], [1.0, 3.0], [2.5, 0.5]])
    g_tar_source = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 2.5]])
    g_reg_target = g_reg_source + 0.25
    g_tar_target = g_tar_source + 0.5

    source_features, target_features, factors, diagnostics = collective_joint_nmf_pair(
        source_cci,
        target_cci,
        g_reg_source,
        g_tar_source,
        g_reg_target,
        g_tar_target,
        rank=2,
        lambda_cci=1.0,
        lambda_grn=1.0,
        max_iter=12,
        seed=31,
    )

    assert source_features.shape == target_features.shape == (3, 4)
    assert factors["S_C"].shape == (2, 2)
    assert factors["B_reg"].shape == factors["B_tar"].shape == (2, 2)
    assert diagnostics["all_finite"] is True
    assert np.isfinite(diagnostics["weighted_objective"])
    for row in diagnostics["losses"]:
        assert np.isfinite(row["cci_relative_loss"])
        assert np.isfinite(row["grn_reg_relative_loss"])
        assert np.isfinite(row["grn_tar_relative_loss"])
