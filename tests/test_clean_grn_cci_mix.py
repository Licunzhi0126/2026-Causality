from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features_native import (
    build_native_feature_block_summary,
    build_native_feature_schema,
    build_native_graph_matrix,
)
from mignet_ce.graph.builder import EDGE_COLUMNS, LayerGraph, _build_commot_inter_edges
from mignet_ce.io.pij_exports import export_pij_sparse_archive
from mignet_ce.mapping import OverlapMapping
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import TransitionKernels
from mignet_ce.representations.graph_features import build_graph_feature_result


def _edge(**values: object) -> dict[str, object]:
    row = {column: np.nan for column in EDGE_COLUMNS}
    row.update(values)
    return row


def _graph(units: list[str]) -> LayerGraph:
    intra = pd.DataFrame(
        [
            _edge(
                src_layer="spot",
                src_unit="l1",
                src_gene="A",
                dst_layer="spot",
                dst_unit="l1",
                dst_gene="B",
                edge_type="spot_intra",
                influence_score=0.4,
            ),
            _edge(
                src_layer="spot",
                src_unit="l2",
                src_gene="A",
                dst_layer="spot",
                dst_unit="l2",
                dst_gene="B",
                edge_type="spot_intra",
                influence_score=0.4,
            ),
        ],
        columns=EDGE_COLUMNS,
    )
    inter = pd.DataFrame(
        [
            _edge(
                src_layer="spot",
                src_unit="l1",
                src_gene="LigA",
                dst_layer="spot",
                dst_unit="l2",
                dst_gene="RecA",
                edge_type="spot_inter",
                commot_lr_key="lr1",
                commot_ligand="LigA_LigB",
                commot_receptor="RecA",
                cci_score_norm=0.5,
                influence_score=0.5,
            ),
            _edge(
                src_layer="spot",
                src_unit="l1",
                src_gene="LigB",
                dst_layer="spot",
                dst_unit="l2",
                dst_gene="RecA",
                edge_type="spot_inter",
                commot_lr_key="lr1",
                commot_ligand="LigA_LigB",
                commot_receptor="RecA",
                cci_score_norm=0.5,
                influence_score=0.5,
            ),
            _edge(
                src_layer="spot",
                src_unit="l1",
                src_gene="LigA",
                dst_layer="spot",
                dst_unit="l3",
                dst_gene="RecA",
                edge_type="spot_inter",
                commot_lr_key="lr1",
                commot_ligand="LigA_LigB",
                commot_receptor="RecA",
                cci_score_norm=0.25,
                influence_score=0.25,
            ),
        ],
        columns=EDGE_COLUMNS,
    )
    return LayerGraph(
        layer="spot",
        time_point="11.5",
        units=units,
        genes=["A", "B", "LigA", "LigB", "RecA"],
        intra_edges=intra,
        inter_edges=inter,
        shared_genes=["A", "B", "LigA", "LigB", "RecA"],
    )


def test_native_feature_matrix_uses_gene_pairs_and_outgoing_lr_pairs() -> None:
    graph = _graph(["l1", "l2", "l3"])
    schema = build_native_feature_schema([graph])
    matrix = build_native_graph_matrix(graph, schema, feature_log1p=False)

    assert matrix.shape == (3, 2)
    assert schema.feature_blocks["intra_grn"] == ["intra_grn:A->B"]
    assert len(schema.feature_blocks["inter_cci"]) == 1
    assert matrix[0, 0] == 0.4
    assert matrix[1, 0] == 0.4
    assert matrix[0, 1] == 0.75
    assert matrix[1, 1] == 0.0


def test_native_feature_block_summary_reports_intra_and_inter_activity() -> None:
    summary = build_native_feature_block_summary(
        np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]]),
        ["u1", "u2"],
        ["intra_a", "inter_a", "inter_b"],
        {
            "intra_grn": ["intra_a"],
            "inter_cci": ["inter_a", "inter_b"],
        },
        stage="11.5",
        layer_role="lower",
    )

    assert summary.loc[0, "intra_sum"] == 1.0
    assert summary.loc[0, "intra_grn_sum"] == 1.0
    assert summary.loc[0, "intra_expr_sum"] == 0.0
    assert summary.loc[0, "inter_sum"] == 2.0
    assert summary.loc[1, "intra_grn_nonzero"] == 0
    assert summary.loc[1, "intra_expr_nonzero"] == 0
    assert summary.loc[1, "inter_nonzero"] == 2
    assert summary.loc[1, "feature_norm"] == 5.0


def test_clean_cci_edges_do_not_require_expression_or_coordinates(tmp_path) -> None:
    lr_dir = tmp_path / "lr"
    lr_dir.mkdir()
    sp.save_npz(lr_dir / "lr1.npz", sp.csr_matrix([[0.0, 2.0], [0.0, 0.0]]))
    manifest = pd.DataFrame(
        [
            {
                "filename": "lr1.npz",
                "lr_key": "lr1",
                "ligand": "LigMissing",
                "receptor": "RecMissing",
            }
        ]
    )
    expr = pd.DataFrame([[0.0], [0.0]], index=["u1", "u2"], columns=["A"])

    edges = _build_commot_inter_edges(
        layer_name="spot",
        manifest=manifest,
        lr_dir=lr_dir,
        index_names=["u1", "u2"],
        expr=expr,
        coords=pd.DataFrame(columns=["x", "y"]),
        active_mask=np.zeros((2, 1), dtype=bool),
        unit_index={"u1": 0, "u2": 1},
        gene_to_idx={"A": 0},
        pair_lookup={},
        score_range=(2.0, 2.0, 1),
        cci_min=0.0,
        require_target_expression=True,
        inter_influence_mode="cci_only",
        inter_additive_cci_weight=1.0,
        inter_additive_grn_weight=1.0,
        inter_grn_pair_policy="zero_if_missing",
        use_expression_mask=False,
        require_coords=False,
    )

    assert len(edges) == 1
    assert edges.loc[0, "src_gene"] == "LigMissing"
    assert edges.loc[0, "dst_gene"] == "RecMissing"
    assert edges.loc[0, "influence_score"] == 1.0
    assert np.isnan(edges.loc[0, "distance_raw"])


def test_native_graph_representation_keeps_native_rows_and_coordinates() -> None:
    overlap0 = OverlapMapping(
        lower_units=["l1", "l2"],
        upper_units=["u1"],
        counts=np.ones((2, 1)),
        weights=np.ones((2, 1)),
    )
    overlap1 = OverlapMapping(
        lower_units=["l1", "l2", "l3"],
        upper_units=["u1", "u2"],
        counts=np.ones((3, 2)),
        weights=np.full((3, 2), 0.5),
    )
    lower_coords = [np.zeros((2, 2)), np.zeros((3, 2))]
    upper_coords = [np.zeros((1, 2)), np.zeros((2, 2))]
    context = NetworkContext(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        time_points=["11.5", "12.5"],
        network_method="clean_grn_cci_mix",
        stable_upper_units=["u1", "u2"],
        shared_genes=["A", "B"],
        lower_mats=[np.ones((2, 3)), np.ones((3, 3))],
        upper_mats=[np.ones((1, 3)), np.ones((2, 3))],
        overlaps=[overlap0, overlap1],
        lower_units_by_time=[["l1", "l2"], ["l1", "l2", "l3"]],
        upper_units_by_time=[["u1"], ["u1", "u2"]],
        upper_coords_by_time=upper_coords,
        lower_coords_by_time=lower_coords,
        feature_alignment_space="native_units",
        feature_names=["f1", "f2", "f3"],
        feature_blocks={"intra_grn": ["f1"], "inter_cci": ["f2", "f3"]},
        graph_summaries=[],
    )

    result = build_graph_feature_result(context, n_components=None)

    assert [matrix.shape[0] for matrix in result.lower_features] == [2, 3]
    assert [matrix.shape[0] for matrix in result.upper_features] == [1, 2]
    assert result.lower_coords is lower_coords
    assert result.upper_coords is upper_coords


def test_native_metrics_keep_ei_and_mark_incompatible_te_di_nan() -> None:
    engine = TemporalMetricsEngine()
    lower = [np.ones((2, 2)), np.ones((3, 2))]
    upper = [np.ones((1, 2)), np.ones((2, 2))]
    p_lower = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    p_upper = np.array([[0.6, 0.4]])

    metrics = engine.calculate_metrics_for_pairs(
        lower_feat=lower,
        upper_feat=upper,
        time_points=["11.5", "12.5"],
        pairs=[(0, 1)],
        organ="heart",
        lower_layer="spot",
        upper_layer="louvain_less_than5",
        precomputed_p_lower={(0, 1): p_lower},
        precomputed_p_upper={(0, 1): p_upper},
        feature_alignment_space="native_units",
    )

    assert np.isfinite(metrics.loc[0, "EI_lower"])
    assert np.isfinite(metrics.loc[0, "EI_upper"])
    assert np.isnan(metrics.loc[0, "TE"])
    assert np.isnan(metrics.loc[0, "DI"])
    assert metrics.loc[0, "metric_alignment"] == "native_units_ei_only"


def test_native_sparse_export_writes_rectangular_matrices_and_stage_units(tmp_path) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        network_method="clean_grn_cci_mix",
    )
    pair = VerticalPairSpec("spot", "louvain_less_than5")
    lower_matrix = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    upper_matrix = np.array([[0.4, 0.6]])

    archive = export_pij_sparse_archive(
        cfg=cfg,
        organ="heart",
        pair=pair,
        stable_upper_units=["u1", "u2"],
        kernels=TransitionKernels(
            p_lower={(0, 1): lower_matrix},
            p_upper={(0, 1): upper_matrix},
        ),
        lower_units_by_time=[["l1", "l2"], ["l1", "l2", "l3"]],
        upper_units_by_time=[["u1"], ["u1", "u2"]],
        feature_alignment_space="native_units",
    )

    assert sp.load_npz(archive / "11.5_to_12.5_lower_P.npz").shape == (2, 3)
    assert sp.load_npz(archive / "11.5_to_12.5_upper_P.npz").shape == (1, 2)
    assert (archive / "units" / "lower_11.5_units.csv").exists()
    assert (archive / "units" / "lower_12.5_units.csv").exists()
    assert (archive / "units" / "upper_11.5_units.csv").exists()
    assert (archive / "units" / "upper_12.5_units.csv").exists()
    assert not (archive / "units.csv").exists()
    with (archive / "kernel_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["feature_alignment_space"] == "native_units"
    assert metadata["unit_mapping_file"] is None
    assert metadata["unit_mapping_files"]["11.5_to_12.5_lower_P.npz"] == {
        "source_units": "units/lower_11.5_units.csv",
        "target_units": "units/lower_12.5_units.csv",
    }
