from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.mapping import OverlapMapping
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.registry import get_pij_method


def assert_row_stochastic(matrix: np.ndarray) -> None:
    assert matrix.ndim == 2
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def _synthetic_context() -> NetworkContext:
    stable_units = ["u1", "u2", "u3"]
    lower_units = ["s1", "s2", "s3"]
    overlaps = [
        OverlapMapping(
            lower_units=lower_units,
            upper_units=stable_units,
            counts=np.eye(3),
            weights=np.eye(3),
        ),
        OverlapMapping(
            lower_units=lower_units,
            upper_units=stable_units,
            counts=np.eye(3),
            weights=np.eye(3),
        ),
    ]
    lower_mats = [
        np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.2, 0.1, 1.0]]),
        np.array([[0.9, 0.1, 0.3], [0.1, 0.9, 0.2], [0.3, 0.2, 0.9]]),
    ]
    upper_mats = [
        np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.2], [0.3, 0.2, 1.0]]),
        np.array([[0.8, 0.2, 0.4], [0.2, 0.8, 0.3], [0.4, 0.3, 0.8]]),
    ]
    return NetworkContext(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        time_points=["11.5", "12.5"],
        network_method="synthetic",
        stable_upper_units=stable_units,
        shared_genes=["g1", "g2", "g3"],
        lower_mats=lower_mats,
        upper_mats=upper_mats,
        overlaps=overlaps,
        lower_units_by_time=[lower_units, lower_units],
        upper_units_by_time=[stable_units, stable_units],
        upper_coords_by_time=[
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.1, 0.0], [1.1, 0.0], [0.0, 1.1]]),
        ],
        feature_names=["f1", "f2", "f3"],
        feature_blocks={"synthetic": ["f1", "f2", "f3"]},
        graph_summaries=[],
        coverage_tables=[pd.DataFrame(), pd.DataFrame()],
    )


def _write_features(root, layer: str, stage: str, units: list[str], velocity_dims: int = 3, include: set[str] | None = None) -> None:
    include = include or {"pseudotime", "sr", "potency_score", "velocity"}
    rows = []
    for idx, unit in enumerate(units):
        row: dict[str, object] = {"unit_id": unit}
        if "pseudotime" in include:
            row["pseudotime"] = 0.1 + 0.2 * idx + (0.05 if stage == "12.5" else 0.0)
        if "sr" in include:
            row["sr"] = 0.9 - 0.1 * idx
        if "potency_score" in include:
            row["potency_score"] = 0.8 - 0.1 * idx
        if "velocity" in include:
            for dim in range(velocity_dims):
                row[f"velocity_{dim}"] = 1.0 if dim == idx % max(1, velocity_dims) else 0.1 * (idx + dim + 1)
        rows.append(row)
    path = root / layer / f"heart_{stage}_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_all_feature_files(root, velocity_dims: int = 3, include: set[str] | None = None) -> None:
    for stage in ("11.5", "12.5"):
        _write_features(root, "spot", stage, ["s1", "s2", "s3"], velocity_dims=velocity_dims, include=include)
        _write_features(root, "louvain_less_than5", stage, ["u1", "u2", "u3"], velocity_dims=velocity_dims, include=include)


@pytest.mark.parametrize(
    "method_name",
    [
        "pseudotime_ot",
        "sr_ot",
        "spatial_ot",
        "sr_spatial_ot",
        "pseudotime_spatial_ot",
        "sr_expression_ot",
        "pseudotime_expression_ot",
        "expr_pseudotime_sr_ot",
        "expr_pseudotime_sr_spatial_ot",
        "expr_pseudotime_sr_energy_ot",
        "velocity_ot",
        "development_ot",
    ],
)
def test_developmental_ot_methods_build_kernels(tmp_path, method_name: str) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root)
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method=method_name,
        pij_feature_components=None,
        ot_max_iter=20,
    )

    result, kernels = get_pij_method(method_name).run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["pij_method"] == method_name
    assert kernels.kernel_metadata["pij_method"] == method_name
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    assert_row_stochastic(kernels.p_upper[(0, 1)])
    assert "main_cost" in kernels.kernel_diagnostics["lower"][(0, 1)]


@pytest.mark.parametrize(
    ("method_name", "expected_components"),
    [
        ("sr_ot", ["sr"]),
        ("pseudotime_ot", ["pseudotime"]),
        ("spatial_ot", ["spatial"]),
        ("sr_spatial_ot", ["sr", "spatial"]),
        ("pseudotime_spatial_ot", ["pseudotime", "spatial"]),
        ("sr_expression_ot", ["sr", "expression"]),
        ("pseudotime_expression_ot", ["pseudotime", "expression"]),
    ],
)
def test_ot_ablation_v2_methods_use_only_declared_cost_components(
    tmp_path,
    method_name: str,
    expected_components: list[str],
) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root)
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method=method_name,
        pij_feature_components=None,
        ot_max_iter=20,
    )

    _, kernels = get_pij_method(method_name).run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    pair_metadata = kernels.kernel_metadata["11.5->12.5"]
    assert pair_metadata["lower"]["cost_components"] == expected_components
    assert pair_metadata["upper"]["cost_components"] == expected_components
    assert set(kernels.kernel_diagnostics["lower"][(0, 1)]) == {
        *(f"{component}_cost" for component in expected_components),
        "main_cost",
    }


@pytest.mark.parametrize(
    ("method_name", "expected_components", "expected_weights"),
    [
        (
            "expr_pseudotime_sr_ot",
            ["expression", "pseudotime", "sr"],
            {"expression": 1.0, "pseudotime": 0.5, "sr": 0.5},
        ),
        (
            "expr_pseudotime_sr_spatial_ot",
            ["expression", "pseudotime", "sr", "spatial"],
            {"expression": 1.0, "pseudotime": 0.5, "sr": 0.5, "spatial": 0.2},
        ),
        (
            "expr_pseudotime_sr_energy_ot",
            ["expression", "pseudotime", "sr", "graph_energy"],
            {"expression": 1.0, "pseudotime": 0.5, "sr": 0.5, "graph_energy": 0.2},
        ),
    ],
)
def test_ot_ablation_v3_methods_use_declared_components_and_weights(
    tmp_path,
    method_name: str,
    expected_components: list[str],
    expected_weights: dict[str, float],
) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root)
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method=method_name,
        pij_feature_components=None,
        ot_max_iter=20,
    )

    _, kernels = get_pij_method(method_name).run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    pair_metadata = kernels.kernel_metadata["11.5->12.5"]
    for space in ("lower", "upper"):
        metadata = pair_metadata[space]
        assert metadata["cost_components"] == expected_components
        for component, expected_weight in expected_weights.items():
            actual_weight = metadata["cost_summary"]["components"][component]["weight"]
            assert actual_weight == pytest.approx(expected_weight)
        assert metadata["cost_summary"]["total_weight"] == pytest.approx(sum(expected_weights.values()))


def test_pseudotime_ot_errors_when_pseudotime_column_is_missing(tmp_path) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root, include={"sr", "potency_score", "velocity"})
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method="pseudotime_ot",
        pij_feature_components=None,
    )

    with pytest.raises(ValueError, match="pseudotime_ot requires developmental feature column"):
        get_pij_method("pseudotime_ot").run(_synthetic_context(), cfg, [(0, 1)])


def test_sr_ot_accepts_potency_score_when_sr_is_missing(tmp_path) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root, include={"pseudotime", "potency_score", "velocity"})
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method="sr_ot",
        pij_feature_components=None,
    )

    _, kernels = get_pij_method("sr_ot").run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    pair_metadata = kernels.kernel_metadata["11.5->12.5"]["lower"]
    assert pair_metadata["cost_components"] == ["potency"]
    assert pair_metadata["feature_columns_used"] == ["potency_score"]


def test_expr_pseudotime_sr_ot_accepts_potency_score_when_sr_is_missing(tmp_path) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root, include={"pseudotime", "potency_score", "velocity"})
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method="expr_pseudotime_sr_ot",
        pij_feature_components=None,
        ot_max_iter=20,
    )

    _, kernels = get_pij_method("expr_pseudotime_sr_ot").run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    pair_metadata = kernels.kernel_metadata["11.5->12.5"]["lower"]
    assert pair_metadata["cost_components"] == ["expression", "pseudotime", "potency"]
    assert pair_metadata["feature_columns_used"] == ["pseudotime", "potency_score"]
    assert pair_metadata["cost_summary"]["components"]["potency"]["weight"] == pytest.approx(0.5)


def test_spatial_ot_does_not_require_developmental_features(tmp_path) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=None,
        pij_method="spatial_ot",
        pij_feature_components=None,
        ot_max_iter=20,
    )
    cfg.validate()

    _, kernels = get_pij_method("spatial_ot").run(_synthetic_context(), cfg, [(0, 1)])

    assert kernels is not None
    assert_row_stochastic(kernels.p_lower[(0, 1)])


def test_spatial_ot_errors_when_coordinates_are_missing(tmp_path) -> None:
    context = _synthetic_context()
    context.upper_coords_by_time = None  # type: ignore[assignment]
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        pij_method="spatial_ot",
        pij_feature_components=None,
    )

    with pytest.raises(ValueError, match="spatial_ot requires coordinates"):
        get_pij_method("spatial_ot").run(context, cfg, [(0, 1)])


def test_expr_pseudotime_sr_spatial_ot_errors_when_coordinates_are_missing(tmp_path) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root)
    context = _synthetic_context()
    context.upper_coords_by_time = None  # type: ignore[assignment]
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method="expr_pseudotime_sr_spatial_ot",
        pij_feature_components=None,
    )

    with pytest.raises(ValueError, match="expr_pseudotime_sr_spatial_ot requires coordinates"):
        get_pij_method("expr_pseudotime_sr_spatial_ot").run(context, cfg, [(0, 1)])


def test_velocity_ot_errors_when_velocity_dimension_does_not_match_features(tmp_path) -> None:
    feature_root = tmp_path / "developmental_features"
    _write_all_feature_files(feature_root, velocity_dims=2)
    cfg = TemporalRunConfig(
        data_root=tmp_path / "data",
        development_feature_root=feature_root,
        pij_method="velocity_ot",
        pij_feature_components=None,
    )

    with pytest.raises(ValueError, match="velocity_\\* dimension to match graph feature dimension"):
        get_pij_method("velocity_ot").run(_synthetic_context(), cfg, [(0, 1)])
