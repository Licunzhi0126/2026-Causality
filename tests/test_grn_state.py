from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mignet_ce.networks.grn_state import (
    build_projected_grn_state,
    deterministic_projection_matrix,
    double_end_grn_state,
    prepare_grn_inputs,
)


def _expression() -> pd.DataFrame:
    return pd.DataFrame(
        [[2.0, 3.0, 4.0], [1.0, 0.0, 5.0]],
        index=["u1", "u2"],
        columns=["a", "b", "c"],
    )


def _grn() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regulator": ["a", "a", "b", "c"],
            "target": ["b", "c", "c", "a"],
            "weight": [-2.0, 1.0, 3.0, 4.0],
        }
    )


def test_double_end_gate_matches_hand_calculation_and_uses_abs_weights() -> None:
    prepared = prepare_grn_inputs(_expression(), ["u1", "u2"], _grn(), top_k_targets=2)

    regulator, target = double_end_grn_state(prepared.expression, prepared.adjacency)

    assert prepared.metadata["grn_weight_mode"] == "abs"
    assert prepared.adjacency.toarray()[0].tolist() == pytest.approx([0.0, 2.0 / 3.0, 1.0 / 3.0])
    assert regulator[0].tolist() == pytest.approx([20.0 / 3.0, 12.0, 8.0])
    assert target[0].tolist() == pytest.approx([8.0, 4.0, 44.0 / 3.0])


def test_topk_is_applied_to_absolute_grn_strength_per_regulator() -> None:
    prepared = prepare_grn_inputs(_expression(), ["u1", "u2"], _grn(), top_k_targets=1)

    assert prepared.adjacency.nnz == 3
    assert prepared.adjacency.toarray()[0].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_gene_role_projection_is_deterministic_and_role_specific() -> None:
    first = deterministic_projection_matrix(["a", "b"], role="reg", output_dim=8, seed=17)
    second = deterministic_projection_matrix(["b", "a"], role="reg", output_dim=8, seed=17)
    target_role = deterministic_projection_matrix(["a"], role="tar", output_dim=8, seed=17)

    assert first[0].tolist() == pytest.approx(second[1].tolist())
    assert first[1].tolist() == pytest.approx(second[0].tolist())
    assert not np.allclose(first[0], target_role[0])


def test_projected_state_has_requested_unit_by_dimension_shape_and_is_repeatable() -> None:
    prepared = prepare_grn_inputs(_expression(), ["u1", "u2"], _grn(), top_k_targets=2)

    first = build_projected_grn_state(prepared, output_dim=4, seed=23)
    second = build_projected_grn_state(prepared, output_dim=4, seed=23)

    assert first.projected.shape == (2, 4)
    assert np.allclose(first.projected, second.projected)
    assert first.metadata["grn_gate_mode"] == "double_end"
