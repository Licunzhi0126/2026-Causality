from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mignet_ce.io.developmental_features import load_developmental_features_for_layer, velocity_columns


def _write_features(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loads_unit_level_developmental_features(tmp_path) -> None:
    root = tmp_path / "developmental_features"
    _write_features(
        root / "spot" / "heart_11.5_features.csv",
        [
            {"unit_id": "s1", "pseudotime": 0.1, "sr": 0.9, "potency_score": 0.8},
            {"unit_id": "s2", "pseudotime": 0.3, "sr": 0.7, "potency_score": 0.6},
        ],
    )

    table = load_developmental_features_for_layer(
        development_feature_root=root,
        data_root=tmp_path,
        layer="spot",
        organ="heart",
        stage="11.5",
        units=["s2", "s1"],
    )

    assert list(table.values.index) == ["s2", "s1"]
    assert table.values.loc["s2", "pseudotime"] == pytest.approx(0.3)
    assert table.metadata["aggregated_from_spot"] is False


def test_aggregates_spot_level_features_to_domain_level(tmp_path) -> None:
    root = tmp_path / "developmental_features"
    _write_features(
        root / "spot" / "heart_11.5_features.csv",
        [
            {"unit_id": "s1", "pseudotime": 0.1, "sr": 0.9, "velocity_0": 1.0, "velocity_1": 0.0},
            {"unit_id": "s2", "pseudotime": 0.3, "sr": 0.7, "velocity_0": 0.0, "velocity_1": 1.0},
            {"unit_id": "s3", "pseudotime": 0.8, "sr": 0.2, "velocity_0": 1.0, "velocity_1": 1.0},
        ],
    )
    spot_map = tmp_path / "map.csv"
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "domain_id": ["d1", "d1", "d2"],
        }
    ).to_csv(spot_map, index=False)

    table = load_developmental_features_for_layer(
        development_feature_root=root,
        data_root=tmp_path,
        layer="louvain_less_than5",
        organ="heart",
        stage="11.5",
        units=["d1", "d2"],
        spot_domain_map=spot_map,
    )

    assert table.metadata["aggregated_from_spot"] is True
    assert table.values.loc["d1", "pseudotime"] == pytest.approx(0.2)
    assert table.values.loc["d1", "velocity_0"] == pytest.approx(0.5)
    assert table.values.loc["d2", "velocity_1"] == pytest.approx(1.0)


def test_missing_unit_errors_by_default(tmp_path) -> None:
    root = tmp_path / "developmental_features"
    _write_features(root / "spot" / "heart_11.5_features.csv", [{"unit_id": "s1", "pseudotime": 0.1}])

    with pytest.raises(ValueError, match="missing_units=1"):
        load_developmental_features_for_layer(
            development_feature_root=root,
            data_root=tmp_path,
            layer="spot",
            organ="heart",
            stage="11.5",
            units=["s1", "s2"],
        )


def test_impute_mean_policy_fills_missing_values(tmp_path) -> None:
    root = tmp_path / "developmental_features"
    _write_features(root / "spot" / "heart_11.5_features.csv", [{"unit_id": "s1", "pseudotime": 0.2}])

    table = load_developmental_features_for_layer(
        development_feature_root=root,
        data_root=tmp_path,
        layer="spot",
        organ="heart",
        stage="11.5",
        units=["s1", "s2"],
        missing_policy="impute_mean",
    )

    assert table.values.loc["s2", "pseudotime"] == pytest.approx(0.2)
    assert "warnings" in table.metadata
    assert np.isfinite(table.values.to_numpy()).all()


def test_velocity_columns_are_loaded_as_matrix_columns(tmp_path) -> None:
    root = tmp_path / "developmental_features"
    _write_features(
        root / "spot" / "heart_11.5_features.csv",
        [
            {"unit_id": "s1", "velocity_1": 0.1, "velocity_0": 0.2},
            {"unit_id": "s2", "velocity_1": 0.3, "velocity_0": 0.4},
        ],
    )

    table = load_developmental_features_for_layer(
        development_feature_root=root,
        data_root=tmp_path,
        layer="spot",
        organ="heart",
        stage="11.5",
        units=["s1", "s2"],
    )

    cols = velocity_columns(table.values)
    assert cols == ["velocity_0", "velocity_1"]
    assert table.values.loc[:, cols].to_numpy().shape == (2, 2)
