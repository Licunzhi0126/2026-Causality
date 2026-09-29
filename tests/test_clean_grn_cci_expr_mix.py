from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import (
    _build_intra_edges,
    _expression_activity,
    _transform_expression_activity,
)
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.networks.clean_grn_cci_expr_mix import CleanGRNCCIExpressionMixBuilder
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


ad = pytest.importorskip("anndata")


def _sample_targets(weight: float = 0.5):
    return {"A": [("B", 2.0, weight)]}


def test_expression_activity_transforms_and_modes() -> None:
    expr = pd.DataFrame(
        {
            "A": [0.0, 2.0, 8.0],
            "B": [0.0, 8.0, 8.0],
            "C": [3.0, 3.0, 3.0],
        },
        index=["u0", "u1", "u2"],
    )
    transformed = _transform_expression_activity(expr, "log1p_minmax")

    assert transformed.shape == expr.shape
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[2, 0] == pytest.approx(1.0)
    assert np.allclose(transformed[:, 2], 1.0)
    assert _expression_activity(0.25, 1.0, "geometric_mean", 0.0) == pytest.approx(0.5)
    assert _expression_activity(0.25, 1.0, "product", 0.0) == pytest.approx(0.25)
    assert _expression_activity(0.25, 1.0, "min", 0.0) == pytest.approx(0.25)
    assert _expression_activity(0.0, 0.0, "geometric_mean", 0.1) == pytest.approx(0.1)
    assert _expression_activity(0.0, 0.0, "none", 5.0) == pytest.approx(1.0)


def test_expression_weighted_intra_multiplies_grn_weight_and_keeps_mask() -> None:
    expr = pd.DataFrame(
        {"A": [4.0, 1.0], "B": [9.0, 0.0]},
        index=["u1", "u2"],
    )
    active = expr.to_numpy() > 0

    edges, metadata = _build_intra_edges(
        "spot",
        expr,
        active,
        _sample_targets(weight=0.8),
        use_expression_mask=True,
        expression_weight_mode="geometric_mean",
        expression_transform="none",
        return_metadata=True,
    )

    assert len(edges) == 1
    assert edges.loc[0, "src_unit"] == "u1"
    assert edges.loc[0, "influence_score"] == pytest.approx(0.8 * 6.0)
    assert metadata["expression_weight_mode"] == "geometric_mean"


def test_unit_specific_intra_uses_direct_weight_and_expression_weighted_fallback() -> None:
    expr = pd.DataFrame(
        {"A": [2.0, 4.0], "B": [8.0, 9.0]},
        index=["u1", "u2"],
    )
    unit_targets = {"u1": {"A": [("B", 3.0, 0.9)]}}

    edges, metadata = _build_intra_edges(
        "spot",
        expr,
        expr.to_numpy() > 0,
        _sample_targets(weight=0.5),
        use_expression_mask=True,
        expression_weight_mode="geometric_mean",
        expression_transform="none",
        unit_regulator_to_targets=unit_targets,
        unit_specific_fallback="sample_grn_expression_weighted",
        return_metadata=True,
    )

    by_unit = edges.set_index("src_unit")["influence_score"]
    assert by_unit["u1"] == pytest.approx(0.9)
    assert by_unit["u2"] == pytest.approx(0.5 * 6.0)
    assert metadata["unit_specific_units"] == 1
    assert metadata["unit_specific_fallback_units"] == ["u2"]


def _write_layer_inputs(
    root,
    *,
    layer: str,
    sample: str,
    units: list[str],
    matrix: np.ndarray,
) -> None:
    h5ad_path = root / layer / "heart" / f"{sample}.h5ad"
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.asarray(matrix, dtype=float),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=["A", "B"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(len(units), dtype=float), np.zeros(len(units))]
    )
    adata.write_h5ad(h5ad_path)

    grn_dir = root / "grn" / layer / sample
    grn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"regulator": ["A"], "target": ["B"], "weight": [2.0]}
    ).to_csv(grn_dir / "grn_edges.csv", index=False)

    cci_dir = root / "cci" / layer
    lr_dir = cci_dir / f"{sample}_COMMOT_by_LR"
    lr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit": units}).to_csv(
        cci_dir / f"{sample}_index.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "filename": ["lr0.npz"],
            "ligand": ["A"],
            "receptor": ["B"],
            "lr_key": ["A-B"],
        }
    ).to_csv(cci_dir / f"{sample}_COMMOT_lr_pairs.tsv", sep="\t", index=False)
    matrix_cci = np.ones((len(units), len(units))) - np.eye(len(units))
    sp.save_npz(lr_dir / "lr0.npz", sp.csr_matrix(matrix_cci))


def _write_vertical_inputs(root) -> None:
    for stage in ("11.5", "12.5"):
        _write_layer_inputs(
            root,
            layer="spot",
            sample=f"spot_heart_{stage}",
            units=["s1", "s2", "s3"],
            matrix=np.array([[1.0, 1.0], [4.0, 9.0], [9.0, 4.0]]),
        )
        sample = f"louvainLessThan5_heart_{stage}"
        _write_layer_inputs(
            root,
            layer="louvain_less_than5",
            sample=sample,
            units=["d1", "d2"],
            matrix=np.array([[2.0, 8.0], [8.0, 2.0]]),
        )
        pd.DataFrame(
            {
                "spot_id": ["s1", "s2", "s3"],
                "domain_id": ["d1", "d1", "d2"],
            }
        ).to_csv(
            root
            / "louvain_less_than5"
            / "heart"
            / f"{sample}_spot_domain_map.csv",
            index=False,
        )


def test_expression_weighted_network_builds_native_unit_context(tmp_path) -> None:
    _write_vertical_inputs(tmp_path)
    cfg = TemporalRunConfig(
        data_root=tmp_path,
        output_root=tmp_path / "output",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "louvain_less_than5")],
        network_method="clean_grn_cci_expr_mix",
        grn_expression_transform="none",
        grn_expression_weight_mode="geometric_mean",
        feature_log1p=False,
    )
    context = CleanGRNCCIExpressionMixBuilder().build_pair_context(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        cfg=cfg,
        resolver=LayerDataResolver(tmp_path),
    )

    assert context.network_method == "clean_grn_cci_expr_mix"
    assert context.feature_alignment_space == "native_units"
    assert context.metadata["intra_source"] == "sample_level_grn_expression_weighted"
    lower_intra = context.lower_graphs[0].intra_edges.groupby("src_unit")[
        "influence_score"
    ].sum()
    assert lower_intra["s1"] != lower_intra["s2"]


def test_expression_weighted_pipeline_exports_raw_features_graphs_and_diagnostics(tmp_path) -> None:
    _write_vertical_inputs(tmp_path)
    cfg = TemporalRunConfig(
        data_root=tmp_path,
        output_root=tmp_path / "output",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "louvain_less_than5")],
        network_method="clean_grn_cci_expr_mix",
        pij_method="expr_ot",
        grn_expression_transform="none",
        feature_log1p=False,
        export_graphs=True,
        export_raw_native_features=True,
        export_feature_diagnostics=True,
        export_features=False,
    )

    metrics = VerticalMIGNetPipeline(cfg).run_pair(
        "heart",
        VerticalPairSpec("spot", "louvain_less_than5"),
    )
    pair_dir = (
        cfg.output_root
        / "features"
        / "heart"
        / "spot_to_louvain_less_than5"
    )

    assert not metrics.empty
    assert (pair_dir / "11.5_lower_features_raw_native.csv").exists()
    assert (pair_dir / "11.5_upper_features_raw_native.csv").exists()
    assert (pair_dir / "diagnostics" / "11.5_feature_block_summary.csv").exists()
    assert (pair_dir / "network_exports" / "11.5_lower_intra_edges.csv").exists()
