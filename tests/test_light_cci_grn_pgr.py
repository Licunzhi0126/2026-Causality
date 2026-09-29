from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.grn_residual import (
    PearsonResidualConfig,
    positive_pearson_residual,
)
from mignet_ce.networks.light_cci_grn import PreparedGRN
from mignet_ce.networks.light_cci_grn_pgr import build_projected_pgr_state
from mignet_ce.networks.registry import get_network_builder


def test_pgr_api_cannot_receive_cross_time_or_evaluation_inputs() -> None:
    assert tuple(inspect.signature(positive_pearson_residual).parameters) == (
        "expression",
        "config",
    )


def test_source_transform_is_independent_of_target_matrix() -> None:
    rng = np.random.default_rng(7)
    source = rng.poisson(3.0, size=(20, 50)).astype(float)
    target_a = rng.poisson(3.0, size=(25, 50)).astype(float)
    target_b = rng.poisson(30.0, size=(25, 50)).astype(float)
    source_a = positive_pearson_residual(source)
    _ = positive_pearson_residual(target_a)
    source_b = positive_pearson_residual(source)
    _ = positive_pearson_residual(target_b)
    np.testing.assert_array_equal(source_a, source_b)


def test_transform_is_deterministic_finite_and_nonnegative() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.poisson(5.0, size=(40, 100)).astype(float)
    config = PearsonResidualConfig(theta=1.0, positive_only=True)
    first = positive_pearson_residual(matrix, config=config)
    second = positive_pearson_residual(matrix, config=config)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0.0)


def test_zero_matrix_is_safe() -> None:
    result = positive_pearson_residual(np.zeros((3, 4)))
    np.testing.assert_array_equal(result, np.zeros((3, 4)))


def test_projected_pgr_state_is_deterministic_and_records_no_leakage() -> None:
    prepared = PreparedGRN(
        units=["u1", "u2", "u3"],
        genes=["a", "b", "c", "d"],
        expression=np.array(
            [
                [8.0, 1.0, 0.0, 4.0],
                [1.0, 7.0, 3.0, 0.0],
                [2.0, 0.0, 6.0, 5.0],
            ]
        ),
        adjacency=sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
        metadata={"fixture": True},
    )
    first = build_projected_pgr_state(prepared, output_dim=6, seed=23)
    second = build_projected_pgr_state(prepared, output_dim=6, seed=23)
    np.testing.assert_array_equal(first.projected, second.projected)
    assert first.projected.shape == (3, 6)
    assert np.all(np.isfinite(first.projected))
    assert first.metadata["grn_expression_transform"] == "positive_pearson_residual"
    assert first.metadata["uses_target_time_for_transform"] is False
    assert first.metadata["uses_cci_for_transform"] is False
    assert first.metadata["uses_pij_or_ei_for_transform"] is False


@pytest.mark.parametrize(
    "pij_method",
    [
        "compare_NG_kl_sinkhorn_grnanchor_v7",
        "compare_NG_fgw_grnanchor_v9",
    ],
)
def test_pgr_network_is_registered_and_allowed_with_frozen_v7_v9(pij_method: str) -> None:
    cfg = TemporalRunConfig(
        network_method="light_cci_grn_pgr",
        pij_method=pij_method,
        pij_entropy_epsilon=0.05,
        pij_temperature=1.0,
    )
    cfg.validate()
    assert get_network_builder(cfg.network_method).network_method == "light_cci_grn_pgr"


def test_pgr_network_rejects_non_grn_compare_method() -> None:
    with pytest.raises(ValueError, match="light_cci_grn/light_cci_grn_pgr requires"):
        TemporalRunConfig(
            network_method="light_cci_grn_pgr",
            pij_method="compare_N_cos",
        ).validate()
