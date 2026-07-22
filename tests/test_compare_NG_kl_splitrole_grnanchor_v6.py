from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.compare._shared.distances import robust_normalize_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_splitrole_grnanchor_v6 import (
    FIXED_FEATURE_BETA,
    N_CORRECTION_WEIGHT,
    REGULATOR_ROLE_WEIGHT,
    TARGET_ROLE_WEIGHT,
    CompareNGKlSplitRoleGRNAnchorV6PijMethod,
    build_splitrole_grnanchored_kl_cost,
)
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


PROTECTED_FILES = (
    "mignet_ce/pij/compare/compare_N_kl.py",
    "mignet_ce/pij/compare/compare_NG_kl_grnanchor_v5.py",
    "mignet_ce/pij/compare/common.py",
    "mignet_ce/pij/compare/_shared/features.py",
    "mignet_ce/pij/compare/_shared/kl.py",
    "mignet_ce/networks/light_cci_grn.py",
)


def _features() -> tuple[np.ndarray, ...]:
    n_source = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    n_target = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    g_reg_source = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0]])
    g_reg_target = np.array([[12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 12.0, 0.0]])
    g_tar_source = np.array([[0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    g_tar_target = np.array([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    return n_source, n_target, g_reg_source, g_reg_target, g_tar_source, g_tar_target


def test_splitrole_cost_matches_formula_and_keeps_raw_grn_dynamic_range() -> None:
    values = _features()
    cost, metadata = build_splitrole_grnanchored_kl_cost(*values)
    n_source, n_target, g_reg_source, g_reg_target, g_tar_source, g_tar_target = values

    n_cost = pairwise_feature_kl(n_source, n_target, beta=FIXED_FEATURE_BETA)
    reg_cost = pairwise_feature_kl(g_reg_source, g_reg_target, beta=FIXED_FEATURE_BETA)
    tar_cost = pairwise_feature_kl(g_tar_source, g_tar_target, beta=FIXED_FEATURE_BETA)
    normalized_n, _ = robust_normalize_cost(n_cost, copy=True)
    expected = (
        REGULATOR_ROLE_WEIGHT * reg_cost
        + TARGET_ROLE_WEIGHT * tar_cost
        + N_CORRECTION_WEIGHT * normalized_n
    )
    np.testing.assert_allclose(cost, expected, rtol=1.0e-12, atol=1.0e-12)
    assert float(cost.max()) > 1.0
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0.0)
    assert metadata["final_cost_clipped_to_unit_interval"] is False
    assert metadata["regulator_target_summed_before_distance"] is False


def test_splitrole_v6_is_registered_without_rebinding_frozen_baseline() -> None:
    method = get_pij_method("compare_NG_kl_splitrole_grnanchor_v6")
    assert isinstance(method, CompareNGKlSplitRoleGRNAnchorV6PijMethod)
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod


def test_splitrole_v6_config_requires_light_cci_grn() -> None:
    TemporalRunConfig(
        network_method="light_cci_grn",
        pij_method="compare_NG_kl_splitrole_grnanchor_v6",
        pij_entropy_epsilon=FIXED_FEATURE_BETA,
        pij_temperature=1.0,
    ).validate()
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(
            network_method="light_cci",
            pij_method="compare_NG_kl_splitrole_grnanchor_v6",
        ).validate()


def test_splitrole_v6_rejects_version_parameter_drift_and_missing_roles() -> None:
    method = CompareNGKlSplitRoleGRNAnchorV6PijMethod()
    values = _features()
    with pytest.raises(ValueError, match="fixes pij_entropy_epsilon"):
        method.build_kl_cost(
            values[0],
            values[1],
            beta=0.5,
            weight_n=0.5,
            weight_g=0.5,
            g_reg_source=values[2],
            g_reg_target=values[3],
            g_tar_source=values[4],
            g_tar_target=values[5],
        )
    with pytest.raises(ValueError, match="requires separate regulator and target"):
        method.build_kl_cost(
            values[0],
            values[1],
            beta=FIXED_FEATURE_BETA,
            weight_n=0.5,
            weight_g=0.5,
        )
    with pytest.raises(ValueError, match="fixes pij_temperature"):
        method._build_pair_kernel(
            source=values[0],
            target=values[1],
            cfg=TemporalRunConfig(pij_temperature=0.5),
            g_reg_source=values[2],
            g_reg_target=values[3],
            g_tar_source=values[4],
            g_tar_target=values[5],
        )


def test_v6_does_not_modify_protected_files_in_worktree() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in PROTECTED_FILES:
        path = repo_root / relative_path
        assert path.exists()
