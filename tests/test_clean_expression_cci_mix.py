from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import build_layer_cci_graph
from mignet_ce.io.loaders import LayerDataResolver, read_expression_h5ad
from mignet_ce.networks.clean_expression_cci_mix import (
    CleanExpressionCCIMixBuilder,
    _build_expression_block,
)
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


ad = pytest.importorskip("anndata")


def _write_h5ad(path, values: np.ndarray, units: list[str], genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.asarray(values, dtype=float),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["count"] = np.asarray(values, dtype=float)
    adata.obsm["spatial"] = np.column_stack([np.arange(len(units), dtype=float), np.zeros(len(units))])
    adata.write_h5ad(path)


def _write_cci_inputs(root, *, layer: str, sample: str, units: list[str]) -> None:
    cci_dir = root / "cci" / layer
    lr_dir = cci_dir / f"{sample}_COMMOT_by_LR"
    lr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit": units}).to_csv(cci_dir / f"{sample}_index.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "filename": ["lr0.npz"],
            "ligand": ["A"],
            "receptor": ["B"],
            "lr_key": ["A-B"],
        }
    ).to_csv(cci_dir / f"{sample}_COMMOT_lr_pairs.tsv", sep="\t", index=False)
    matrix = np.ones((len(units), len(units)), dtype=float) - np.eye(len(units), dtype=float)
    sp.save_npz(lr_dir / "lr0.npz", sp.csr_matrix(matrix))


def _write_clean_expression_inputs(root) -> None:
    genes = ["A", "B", "C"]
    for stage, offset in (("11.5", 0.0), ("12.5", 1.0)):
        spot_sample = f"spot_heart_{stage}"
        spot_units = ["s1", "s2", "s3"]
        _write_h5ad(
            root / "spot" / "heart" / f"{spot_sample}.h5ad",
            np.array(
                [
                    [1.0 + offset, 0.0, 2.0],
                    [4.0, 9.0 + offset, 1.0],
                    [9.0, 4.0, 3.0 + offset],
                ]
            ),
            spot_units,
            genes,
        )
        _write_cci_inputs(root, layer="spot", sample=spot_sample, units=spot_units)

        domain_sample = f"louvainLessThan5_heart_{stage}"
        domain_units = ["d1", "d2"]
        _write_h5ad(
            root / "louvain_less_than5" / "heart" / f"{domain_sample}.h5ad",
            np.array(
                [
                    [2.0 + offset, 8.0, 1.0],
                    [8.0, 2.0 + offset, 5.0],
                ]
            ),
            domain_units,
            genes,
        )
        _write_cci_inputs(root, layer="louvain_less_than5", sample=domain_sample, units=domain_units)
        pd.DataFrame(
            {
                "spot_id": spot_units,
                "domain_id": ["d1", "d1", "d2"],
            }
        ).to_csv(
            root / "louvain_less_than5" / "heart" / f"{domain_sample}_spot_domain_map.csv",
            index=False,
        )


def test_clean_expression_cci_mix_registered() -> None:
    builder = get_network_builder("clean_expression_cci_mix")

    assert builder.network_method == "clean_expression_cci_mix"


def test_expression_block_uses_units_genes_order_and_log1p() -> None:
    expr = pd.DataFrame(
        {"A": [3.0, -2.0], "B": [0.0, 8.0]},
        index=["u2", "u1"],
    )

    block = _build_expression_block(expr, ["u1", "u2"], ["B", "A"], feature_log1p=True)

    assert np.allclose(block, np.log1p(np.array([[8.0, 0.0], [0.0, 3.0]])))


def test_build_layer_cci_graph_does_not_require_grn_edges(tmp_path) -> None:
    _write_clean_expression_inputs(tmp_path)
    resolver = LayerDataResolver(tmp_path)
    paths = resolver.paths("spot", "heart", "11.5")
    expression = read_expression_h5ad(paths.h5ad)

    assert not paths.grn_edges.exists()
    graph = build_layer_cci_graph(
        layer_name="spot",
        time_point="11.5",
        expression=expression,
        paths=paths,
        shared_genes=["A", "B", "C"],
        cci_inter_use_expression_mask=False,
    )

    assert graph.intra_edges.empty
    assert not graph.inter_edges.empty
    assert graph.metadata["grn_source"] == "not_used"
    assert graph.metadata["inter_source"] == "cci_only"


def test_clean_expression_cci_mix_builds_native_expression_plus_cci_context(tmp_path) -> None:
    _write_clean_expression_inputs(tmp_path)
    cfg = TemporalRunConfig(
        data_root=tmp_path,
        output_root=tmp_path / "output",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "louvain_less_than5")],
        network_method="clean_expression_cci_mix",
        feature_log1p=False,
    )

    context = CleanExpressionCCIMixBuilder().build_pair_context(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        cfg=cfg,
        resolver=LayerDataResolver(tmp_path),
    )

    assert context.network_method == "clean_expression_cci_mix"
    assert context.feature_alignment_space == "native_units"
    assert context.metadata["uses_grn"] is False
    assert context.metadata["uses_cci"] is True
    assert context.metadata["uses_expression"] is True
    assert any(name.startswith("intra_expr:") for name in context.feature_names)
    assert any(name.startswith("inter_cci:") for name in context.feature_names)
    assert not any(name.startswith("intra_grn:") for name in context.feature_names)
    assert set(context.feature_blocks) == {"intra_expr", "inter_cci"}
    assert context.lower_mats[0].shape[0] == len(context.lower_units_by_time[0])
    assert context.upper_mats[0].shape[0] == len(context.upper_units_by_time[0])
    assert context.lower_mats[0].shape[1] == len(context.feature_names)
    assert context.upper_mats[0].shape[1] == len(context.feature_names)


def test_clean_expression_cci_mix_builder_does_not_call_feature_alignment_helpers() -> None:
    source = inspect.getsource(CleanExpressionCCIMixBuilder)

    assert "aggregate_lower_features_to_upper" not in source
    assert "align_upper_features" not in source


def test_clean_expression_cci_mix_pipeline_exports_expression_and_cci_features(tmp_path) -> None:
    _write_clean_expression_inputs(tmp_path)
    cfg = TemporalRunConfig(
        data_root=tmp_path,
        output_root=tmp_path / "output",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "louvain_less_than5")],
        network_method="clean_expression_cci_mix",
        pij_method="expr_ot",
        feature_log1p=False,
        export_graphs=True,
        export_raw_native_features=True,
        export_feature_diagnostics=True,
        export_features=False,
        pij_feature_components=None,
        kraskov_k=1,
        ot_max_iter=20,
    )

    metrics = VerticalMIGNetPipeline(cfg).run_pair(
        "heart",
        VerticalPairSpec("spot", "louvain_less_than5"),
    )
    pair_dir = cfg.output_root / "features" / "heart" / "spot_to_louvain_less_than5"
    lower_raw = pd.read_csv(pair_dir / "11.5_lower_features_raw_native.csv", index_col=0)
    diagnostics = pd.read_csv(pair_dir / "diagnostics" / "11.5_feature_block_summary.csv")

    assert not metrics.empty
    assert any(column.startswith("intra_expr:") for column in lower_raw.columns)
    assert any(column.startswith("inter_cci:") for column in lower_raw.columns)
    assert not any(column.startswith("intra_grn:") for column in lower_raw.columns)
    assert {"intra_expr_sum", "intra_grn_sum", "inter_sum"}.issubset(diagnostics.columns)
    assert diagnostics["intra_expr_sum"].sum() > 0
    assert (pair_dir / "network_exports" / "11.5_lower_inter_edges.csv").exists()

