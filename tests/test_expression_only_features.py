from __future__ import annotations

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.mapping import OverlapMapping
from mignet_ce.networks.expression_only import ExpressionOnlyBuilder
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline
from mignet_ce.representations.expression_only import (
    aggregate_lower_expression_to_upper,
    align_upper_expression_to_stable,
    build_expression_only_feature_result,
)


def test_aggregate_lower_expression_to_stable_upper_units() -> None:
    expression = np.array(
        [
            [5.0, 0.0, 2.0],
            [3.0, 1.0, 4.0],
            [0.0, 6.0, 1.0],
            [2.0, 5.0, 0.0],
            [1.0, 7.0, 2.0],
            [4.0, 0.0, 8.0],
        ]
    )
    counts = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    overlap = OverlapMapping(
        lower_units=["s1", "s2", "s3", "s4", "s5", "s6"],
        upper_units=["A", "B", "C"],
        counts=counts,
        weights=counts,
    )

    aggregated, coverage = aggregate_lower_expression_to_upper(expression, overlap)

    assert np.allclose(
        aggregated,
        np.array(
            [
                [4.0, 0.5, 3.0],
                [1.0, 6.0, 1.0],
                [4.0, 0.0, 8.0],
            ]
        ),
    )
    assert np.allclose(coverage, np.array([2.0, 3.0, 1.0]))


def test_align_upper_expression_zero_fills_missing_stable_units() -> None:
    aligned = align_upper_expression_to_stable(
        np.array([[9.0, 1.0, 6.0], [7.0, 0.0, 9.0]]),
        current_units=["A", "C"],
        stable_upper_units=["A", "B", "C"],
    )

    assert np.allclose(
        aligned,
        np.array(
            [
                [9.0, 1.0, 6.0],
                [0.0, 0.0, 0.0],
                [7.0, 0.0, 9.0],
            ]
        ),
    )


def _write_h5ad(path, values: np.ndarray, units: list[str], genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=sp.csr_matrix(values),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["count"] = sp.csr_matrix(values)
    adata.obsm["spatial"] = np.column_stack([np.arange(values.shape[0]), np.zeros(values.shape[0])])
    adata.write_h5ad(path)


def test_expression_only_builder_and_feature_result_use_h5ad_expression_without_grn_or_cci(tmp_path) -> None:
    data_root = tmp_path / "data"
    genes = ["gene1", "gene2", "gene3"]
    lower_values = np.array(
        [
            [5.0, 0.0, 2.0],
            [3.0, 1.0, 4.0],
            [0.0, 6.0, 1.0],
            [2.0, 5.0, 0.0],
            [1.0, 7.0, 2.0],
            [4.0, 0.0, 8.0],
        ]
    )
    upper_values = np.array(
        [
            [9.0, 1.0, 6.0],
            [2.0, 10.0, 3.0],
            [7.0, 0.0, 9.0],
        ]
    )
    _write_h5ad(data_root / "spot" / "heart" / "spot_heart_11.5.h5ad", lower_values, ["s1", "s2", "s3", "s4", "s5", "s6"], genes)
    _write_h5ad(data_root / "seurat_k40" / "heart" / "seurat_heart_11.5.h5ad", upper_values, ["A", "B", "C"], genes)
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "domain_id": ["A", "A", "B", "B", "B", "C"],
        }
    ).to_csv(data_root / "seurat_k40" / "heart" / "seurat_heart_11.5_spot_domain_map.csv", index=False)
    cfg = TemporalRunConfig(
        data_root=data_root,
        output_root=tmp_path / "out",
        organs=["heart"],
        time_points=["11.5"],
        level_pairs=[VerticalPairSpec("spot", "seurat_k40")],
        network_method="expression_only",
        pij_method="pure_expression_ot",
        pij_feature_components=None,
        pure_expression_normalize=False,
        pure_expression_log1p=False,
        pure_expression_gene_selection="all",
        pure_expression_scaler="none",
    )

    context = ExpressionOnlyBuilder().build_pair_context(
        organ="heart",
        pair=VerticalPairSpec("spot", "seurat_k40"),
        cfg=cfg,
        resolver=LayerDataResolver(data_root),
    )
    result = build_expression_only_feature_result(context, cfg)

    assert context.metadata["uses_grn"] is False
    assert context.metadata["uses_cci"] is False
    assert context.metadata["uses_legacy_graph"] is False
    assert context.shared_genes == genes
    assert np.allclose(result.lower_features[0], np.array([[4.0, 0.5, 3.0], [1.0, 6.0, 1.0], [4.0, 0.0, 8.0]]))
    assert np.allclose(result.upper_features[0], upper_values)
    assert result.method_metadata["feature_source"] == "pure_expression"
    assert result.method_metadata["selected_genes"] == genes


def test_vertical_pipeline_runs_expression_only_pure_expression_ot(tmp_path) -> None:
    data_root = tmp_path / "data"
    genes = ["gene1", "gene2", "gene3"]
    for stage, offset in (("11.5", 0.0), ("12.5", 1.0)):
        _write_h5ad(
            data_root / "spot" / "heart" / f"spot_heart_{stage}.h5ad",
            np.array(
                [
                    [5.0 + offset, 0.0, 2.0],
                    [3.0 + offset, 1.0, 4.0],
                    [0.0, 6.0 + offset, 1.0],
                    [2.0, 5.0 + offset, 0.0],
                    [1.0, 7.0 + offset, 2.0],
                    [4.0, 0.0, 8.0 + offset],
                ]
            ),
            ["s1", "s2", "s3", "s4", "s5", "s6"],
            genes,
        )
        _write_h5ad(
            data_root / "seurat_k40" / "heart" / f"seurat_heart_{stage}.h5ad",
            np.array(
                [
                    [9.0 + offset, 1.0, 6.0],
                    [2.0, 10.0 + offset, 3.0],
                    [7.0, 0.0, 9.0 + offset],
                ]
            ),
            ["A", "B", "C"],
            genes,
        )
        pd.DataFrame(
            {
                "spot_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "domain_id": ["A", "A", "B", "B", "B", "C"],
            }
        ).to_csv(data_root / "seurat_k40" / "heart" / f"seurat_heart_{stage}_spot_domain_map.csv", index=False)

    cfg = TemporalRunConfig(
        data_root=data_root,
        output_root=tmp_path / "out",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "seurat_k40")],
        network_method="expression_only",
        pij_method="pure_expression_ot",
        export_pair_artifacts=True,
        export_pij=True,
        pij_feature_components=None,
        pure_expression_normalize=False,
        pure_expression_log1p=False,
        pure_expression_gene_selection="all",
        pure_expression_scaler="none",
        kraskov_k=1,
        ot_max_iter=20,
    )

    metrics = VerticalMIGNetPipeline(cfg).run()

    assert not metrics.empty
    assert set(metrics["network_method"]) == {"expression_only"}
    pair_dir = cfg.output_root / "features" / "heart" / "spot_to_seurat_k40"
    assert (pair_dir / "pure_expression_genes.csv").exists()
    assert (pair_dir / "pure_expression_feature_schema.json").exists()
    exported = pd.read_csv(pair_dir / "11.5_lower_features_scaled.csv", index_col=0)
    assert list(exported.columns) == ["pure_expression_gene_component_1", "pure_expression_gene_component_2", "pure_expression_gene_component_3"]
