from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mignet_ce.pij.feature_versions.grn_features import build_split_grn_features, transform_expression_new_version
from mignet_ce.pij.feature_versions.recipes import get_feature_recipe
from mignet_ce.pij.feature_versions.sources import RawGRNInputs


def test_new_expression_transform_is_per_gene_and_constant_safe() -> None:
    expression = np.array([[0.0, 5.0, 2.0], [3.0, 5.0, 6.0], [8.0, 5.0, 10.0]])
    transformed = transform_expression_new_version(expression)
    assert np.all(np.isfinite(transformed))
    assert np.all(transformed >= 0.0) and np.all(transformed <= 1.0)
    np.testing.assert_array_equal(transformed[:, 1], np.zeros(3))
    np.testing.assert_allclose(transformed.min(axis=0), np.zeros(3))
    np.testing.assert_allclose(transformed.max(axis=0)[[0, 2]], np.ones(2))


def test_grn_regulator_and_target_roles_remain_separate_and_reproducible() -> None:
    expression = pd.DataFrame(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]],
        index=["u1", "u2", "u3"],
        columns=["a", "b", "c"],
    )
    edges = pd.DataFrame(
        {"regulator": ["a", "a", "b", "c"], "target": ["b", "c", "c", "a"], "weight": [2.0, 1.0, 3.0, 1.5]}
    )
    raw = RawGRNInputs(expression, edges, ("u1", "u2", "u3"), Path("expression.h5ad"), Path("grn_edges.csv"))
    recipe = get_feature_recipe("ncomp_gcos_v2")
    first, metadata, artifacts = build_split_grn_features(raw, recipe)
    second, _, second_artifacts = build_split_grn_features(raw, recipe)

    assert set(first) == {"g_reg", "g_tar"}
    assert metadata["regulator_target_summed"] is False
    assert first["g_reg"].shape == first["g_tar"].shape == (3, recipe.projection_dim)
    np.testing.assert_array_equal(first["g_reg"], second["g_reg"])
    np.testing.assert_array_equal(first["g_tar"], second["g_tar"])
    np.testing.assert_array_equal(artifacts["Q_reg"], second_artifacts["Q_reg"])
    assert not np.array_equal(artifacts["Q_reg"], artifacts["Q_tar"])
    assert not np.array_equal(first["g_reg"], first["g_tar"])
