from __future__ import annotations

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp
import pytest

from mignet_ce.io.developmental_feature_builder import (
    DevelopmentalFeatureBuildConfig,
    build_developmental_features,
)
from mignet_ce.io.developmental_features import load_developmental_features_for_layer


def _write_spot_h5ad(data_root, stage: str, x: np.ndarray) -> list[str]:
    units = [f"s{idx + 1}" for idx in range(x.shape[0])]
    genes = [f"g{idx + 1}" for idx in range(x.shape[1])]
    obs = pd.DataFrame(
        {
            "Module_1": np.linspace(0.1, 0.8, x.shape[0]),
            "Module_2": np.linspace(0.9, 0.2, x.shape[0]),
        },
        index=units,
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(x),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    adata.layers["count"] = sp.csr_matrix(x)
    adata.obsm["spatial"] = np.column_stack([np.arange(x.shape[0]), np.arange(x.shape[0])])
    path = data_root / "spot" / "heart" / f"spot_heart_{stage}.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return units


def _build_synthetic_features(tmp_path, velocity_components: int = 3):
    data_root = tmp_path / "data"
    output_root = tmp_path / "developmental_features"
    _write_spot_h5ad(
        data_root,
        "11.5",
        np.array(
            [
                [5.0, 1.0, 0.0, 2.0],
                [4.0, 1.5, 0.0, 1.5],
                [3.5, 2.0, 0.5, 1.0],
            ]
        ),
    )
    _write_spot_h5ad(
        data_root,
        "12.5",
        np.array(
            [
                [1.0, 3.0, 4.0, 0.5],
                [0.5, 3.5, 5.0, 1.0],
                [0.0, 4.0, 6.0, 1.5],
            ]
        ),
    )
    result = build_developmental_features(
        DevelopmentalFeatureBuildConfig(
            data_root=data_root,
            output_root=output_root,
            organs=("heart",),
            time_points=("11.5", "12.5"),
            velocity_components=velocity_components,
            overwrite=True,
        )
    )
    return data_root, output_root, result


def test_builds_factory_proxy_spot_csv_from_synthetic_h5ad(tmp_path) -> None:
    _, output_root, result = _build_synthetic_features(tmp_path)

    assert set(result.manifest["status"]) == {"ok"}
    assert set(result.manifest["sr_source"]) == {"module_entropy"}
    for stage in ("11.5", "12.5"):
        path = output_root / "spot" / f"heart_{stage}_features.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df["unit_id"]) == ["s1", "s2", "s3"]
        for column in ["pseudotime", "sr", "potency_score", "velocity_0", "velocity_1", "velocity_2"]:
            assert column in df.columns
        assert np.isfinite(df.drop(columns=["unit_id"]).to_numpy(dtype=float)).all()

    assert (output_root / "manifest" / "developmental_features_manifest.csv").exists()
    assert (output_root / "qc" / "heart_11.5_feature_stats.csv").exists()


def test_builder_pseudotime_is_monotonic_across_stages(tmp_path) -> None:
    _, output_root, _ = _build_synthetic_features(tmp_path)

    early = pd.read_csv(output_root / "spot" / "heart_11.5_features.csv")
    late = pd.read_csv(output_root / "spot" / "heart_12.5_features.csv")

    assert early["pseudotime"].mean() < late["pseudotime"].mean()


def test_builder_output_can_be_loaded_by_existing_reader(tmp_path) -> None:
    data_root, output_root, _ = _build_synthetic_features(tmp_path)

    table = load_developmental_features_for_layer(
        development_feature_root=output_root,
        data_root=data_root,
        layer="spot",
        organ="heart",
        stage="11.5",
        units=["s2", "s1"],
    )

    assert list(table.values.index) == ["s2", "s1"]
    assert {"pseudotime", "sr", "potency_score", "velocity_0"}.issubset(table.values.columns)
    assert np.isfinite(table.values.to_numpy(dtype=float)).all()


def test_builder_spot_csv_auto_aggregates_to_domain_layer(tmp_path) -> None:
    data_root, output_root, _ = _build_synthetic_features(tmp_path)
    map_path = data_root / "louvain_less_than5" / "heart" / "louvainLessThan5_heart_11.5_spot_domain_map.csv"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "domain_id": ["d1", "d1", "d2"],
        }
    ).to_csv(map_path, index=False)

    table = load_developmental_features_for_layer(
        development_feature_root=output_root,
        data_root=data_root,
        layer="louvain_less_than5",
        organ="heart",
        stage="11.5",
        units=["d1", "d2"],
    )

    assert table.metadata["aggregated_from_spot"] is True
    assert list(table.values.index) == ["d1", "d2"]
    assert np.isfinite(table.values.to_numpy(dtype=float)).all()


def _write_domain_inputs(data_root, layer: str, stage: str, units: list[str], spot_units: list[str]) -> None:
    prefix = "seurat150" if layer == "seurat_k150" else "seurat"
    stem = f"{prefix}_heart_{stage}"
    path = data_root / layer / "heart" / f"{stem}.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(
        X=sp.csr_matrix(np.ones((len(units), 2), dtype=float)),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(path)
    pd.DataFrame(
        {
            "spot_id": spot_units,
            "domain_id": [units[idx % len(units)] for idx in range(len(spot_units))],
        }
    ).to_csv(path.with_name(f"{stem}_spot_domain_map.csv"), index=False)


def test_builder_materializes_seurat_layers_in_standard_feature_root(tmp_path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "developmental_features"
    spot_units = _write_spot_h5ad(data_root, "11.5", np.arange(160 * 4, dtype=float).reshape(160, 4) + 1.0)
    units150 = [f"domain_{idx:03d}" for idx in range(1, 151)]
    units40 = [f"domain_{idx:03d}" for idx in range(1, 41)]
    _write_domain_inputs(data_root, "seurat_k150", "11.5", units150, spot_units)
    _write_domain_inputs(data_root, "seurat_k40", "11.5", units40, spot_units)

    result = build_developmental_features(
        DevelopmentalFeatureBuildConfig(
            data_root=data_root,
            output_root=output_root,
            organs=("heart",),
            time_points=("11.5",),
            layers=("spot", "seurat_k150", "seurat_k40"),
            velocity_components=2,
        )
    )

    for layer, expected_units in (("seurat_k150", units150), ("seurat_k40", units40)):
        output = pd.read_csv(output_root / layer / "heart_11.5_features.csv")
        assert output["unit_id"].tolist() == expected_units
        assert "spot_count" in output.columns
        assert np.isfinite(output.drop(columns=["unit_id"]).to_numpy(dtype=float)).all()
        assert (output_root / "qc" / layer / "heart_11.5_feature_stats.csv").exists()
    assert set(result.manifest["layer"]) == {"spot", "seurat_k150", "seurat_k40"}
    domain_rows = result.manifest[result.manifest["layer"] != "spot"]
    assert set(domain_rows["aggregation"]) == {"mean_over_member_spots"}


def test_builder_domain_materialization_rejects_missing_spot_features(tmp_path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "developmental_features"
    spot_units = _write_spot_h5ad(data_root, "11.5", np.arange(9, dtype=float).reshape(3, 3) + 1.0)
    units40 = [f"domain_{idx:03d}" for idx in range(1, 41)]
    _write_domain_inputs(data_root, "seurat_k40", "11.5", units40, [*spot_units, "missing_spot"])

    with pytest.raises(ValueError, match="missing from spot developmental features"):
        build_developmental_features(
            DevelopmentalFeatureBuildConfig(
                data_root=data_root,
                output_root=output_root,
                organs=("heart",),
                time_points=("11.5",),
                layers=("spot", "seurat_k40"),
                velocity_components=2,
            )
        )


def test_domain_only_build_uses_existing_spot_feature_values(tmp_path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "developmental_features"
    spot_units = _write_spot_h5ad(data_root, "11.5", np.ones((40, 3), dtype=float))
    units40 = [f"domain_{idx:03d}" for idx in range(1, 41)]
    _write_domain_inputs(data_root, "seurat_k40", "11.5", units40, spot_units)
    spot_path = output_root / "spot" / "heart_11.5_features.csv"
    spot_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "unit_id": spot_units,
            "pseudotime": np.linspace(0.0, 1.0, 40),
            "sr": np.linspace(1.0, 0.0, 40),
            "potency_score": np.linspace(1.0, 0.0, 40),
            "velocity_0": np.zeros(40),
            "velocity_1": np.ones(40),
        }
    ).to_csv(spot_path, index=False)

    result = build_developmental_features(
        DevelopmentalFeatureBuildConfig(
            data_root=data_root,
            output_root=output_root,
            organs=("heart",),
            time_points=("11.5",),
            layers=("seurat_k40",),
            velocity_components=2,
        )
    )

    output = pd.read_csv(output_root / "seurat_k40" / "heart_11.5_features.csv")
    assert output["pseudotime"].tolist() == pytest.approx(np.linspace(0.0, 1.0, 40))
    assert result.manifest.iloc[0]["sr_source"] == "precomputed_spot_features"
