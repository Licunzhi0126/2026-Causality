from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features import coverage_table
from mignet_ce.features_native import build_native_feature_schema, build_native_graph_matrix
from mignet_ce.graph.builder import LayerGraph, build_layer_cci_graph
from mignet_ce.io.loaders import LayerDataResolver, LayerPaths, read_expression_h5ad
from mignet_ce.mapping import (
    build_overlap_edge_table,
    build_overlap_mapping,
    build_spot_correspondence_table,
    load_unit_assignments,
    summarize_overlap_quality,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.clean_grn_cci_mix import CleanGRNCCIMixBuilder
from mignet_ce.utils.coords import align_coords


def _build_expression_block(
    expr: pd.DataFrame,
    units: list[str],
    genes: list[str],
    feature_log1p: bool,
) -> np.ndarray:
    mat = expr.loc[list(units), list(genes)].to_numpy(dtype=float)
    if feature_log1p:
        mat = np.log1p(np.clip(mat, a_min=0.0, a_max=None))
    return mat


class CleanExpressionCCIMixBuilder(CleanGRNCCIMixBuilder):
    network_method = "clean_expression_cci_mix"

    def _required_paths(self, paths: LayerPaths) -> List[Path]:
        required = [paths.h5ad, paths.cci_manifest, paths.cci_index, paths.cci_lr_dir]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        return required

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        all_paths = self._check_pair_paths(cfg, resolver, organ, pair)
        shared_genes = self._compute_shared_expression_genes(cfg, all_paths, pair)
        stable_upper_units = self._stable_upper_units(cfg, all_paths, pair.upper_layer)

        overlaps = []
        lower_units_by_time: List[List[str]] = []
        upper_units_by_time: List[List[str]] = []
        lower_assignments_by_time: List[pd.DataFrame] = []
        upper_assignments_by_time: List[pd.DataFrame] = []
        lower_graphs: List[LayerGraph] = []
        upper_graphs: List[LayerGraph] = []
        lower_expr_blocks: List[np.ndarray] = []
        upper_expr_blocks: List[np.ndarray] = []
        lower_coords_by_time: List[np.ndarray] = []
        upper_coords_by_time: List[np.ndarray] = []
        coverage_tables: List[pd.DataFrame] = []
        spot_correspondence_tables: List[pd.DataFrame] = []
        overlap_edge_tables: List[pd.DataFrame] = []
        overlap_quality_summaries: List[dict[str, object]] = []
        exports: dict[str, pd.DataFrame] = {}

        for stage in map(str, cfg.time_points):
            lower_paths = all_paths[(stage, pair.lower_layer)]
            upper_paths = all_paths[(stage, pair.upper_layer)]
            lower_expr = read_expression_h5ad(lower_paths.h5ad)
            upper_expr = read_expression_h5ad(upper_paths.h5ad)

            lower_assignments = load_unit_assignments(pair.lower_layer, lower_expr, lower_paths.spot_domain_map)
            upper_assignments = load_unit_assignments(pair.upper_layer, upper_expr, upper_paths.spot_domain_map)
            overlap = build_overlap_mapping(
                lower=lower_assignments,
                upper=upper_assignments,
                lower_units=lower_expr.units,
                upper_units=stable_upper_units,
            )
            spot_correspondence = build_spot_correspondence_table(
                lower=lower_assignments,
                upper=upper_assignments,
                stage=stage,
                lower_layer=pair.lower_layer,
                upper_layer=pair.upper_layer,
            )
            overlap_edges = build_overlap_edge_table(
                overlap=overlap,
                stage=stage,
                lower_layer=pair.lower_layer,
                upper_layer=pair.upper_layer,
            )
            overlap_quality = summarize_overlap_quality(overlap_edges)
            overlap_quality["stage"] = stage

            graph_kwargs = {
                "shared_genes": shared_genes,
                "expr_threshold": cfg.expr_threshold,
                "cci_min": cfg.cci_min,
                "require_target_expression_for_inter": cfg.require_target_expression_for_inter,
                "cci_inter_use_expression_mask": cfg.native_cci_inter_use_expression_mask,
                "cci_inter_require_coords": False,
            }
            lower_graph = build_layer_cci_graph(
                layer_name=pair.lower_layer,
                time_point=stage,
                expression=lower_expr,
                paths=lower_paths,
                **graph_kwargs,
            )
            upper_graph = build_layer_cci_graph(
                layer_name=pair.upper_layer,
                time_point=stage,
                expression=upper_expr,
                paths=upper_paths,
                **graph_kwargs,
            )

            lower_expr_block = _build_expression_block(
                lower_expr.expr,
                lower_graph.units,
                shared_genes,
                cfg.feature_log1p,
            )
            upper_expr_block = _build_expression_block(
                upper_expr.expr,
                upper_graph.units,
                shared_genes,
                cfg.feature_log1p,
            )

            overlaps.append(overlap)
            lower_units_by_time.append(lower_graph.units)
            upper_units_by_time.append(upper_graph.units)
            lower_assignments_by_time.append(lower_assignments.rows.copy())
            upper_assignments_by_time.append(upper_assignments.rows.copy())
            lower_graphs.append(lower_graph)
            upper_graphs.append(upper_graph)
            lower_expr_blocks.append(lower_expr_block)
            upper_expr_blocks.append(upper_expr_block)
            lower_coords_by_time.append(align_coords(lower_expr.coords, lower_graph.units))
            upper_coords_by_time.append(align_coords(upper_expr.coords, upper_graph.units))
            coverage_tables.append(coverage_table(stage, stable_upper_units, overlap.coverage_counts(), upper_graph.units))
            spot_correspondence_tables.append(spot_correspondence)
            overlap_edge_tables.append(overlap_edges)
            overlap_quality_summaries.append(overlap_quality)

            if cfg.export_graphs:
                exports[f"network_exports/{stage}_lower_intra_edges.csv"] = lower_graph.intra_edges.copy()
                exports[f"network_exports/{stage}_lower_inter_edges.csv"] = lower_graph.inter_edges.copy()
                exports[f"network_exports/{stage}_upper_intra_edges.csv"] = upper_graph.intra_edges.copy()
                exports[f"network_exports/{stage}_upper_inter_edges.csv"] = upper_graph.inter_edges.copy()

        cci_schema = build_native_feature_schema([*lower_graphs, *upper_graphs])
        lower_cci_mats = [
            build_native_graph_matrix(graph, cci_schema, feature_log1p=cfg.feature_log1p)
            for graph in lower_graphs
        ]
        upper_cci_mats = [
            build_native_graph_matrix(graph, cci_schema, feature_log1p=cfg.feature_log1p)
            for graph in upper_graphs
        ]
        lower_mats = [
            np.hstack([expr_block, cci_mat])
            for expr_block, cci_mat in zip(lower_expr_blocks, lower_cci_mats)
        ]
        upper_mats = [
            np.hstack([expr_block, cci_mat])
            for expr_block, cci_mat in zip(upper_expr_blocks, upper_cci_mats)
        ]

        expr_feature_names = [f"intra_expr:{gene}" for gene in shared_genes]
        cci_feature_names = list(cci_schema.feature_names)
        feature_names = expr_feature_names + cci_feature_names
        feature_blocks = {
            "intra_expr": expr_feature_names,
            "inter_cci": cci_schema.feature_blocks.get("inter_cci", []),
        }
        graph_summaries = [
            self._expression_cci_summary(
                stage,
                lower_graph,
                upper_graph,
                lower_mat,
                upper_mat,
                cfg,
            )
            for stage, lower_graph, upper_graph, lower_mat, upper_mat in zip(
                map(str, cfg.time_points),
                lower_graphs,
                upper_graphs,
                lower_mats,
                upper_mats,
            )
        ]

        return NetworkContext(
            organ=organ,
            pair=pair,
            time_points=list(map(str, cfg.time_points)),
            network_method=self.network_method,
            stable_upper_units=stable_upper_units,
            shared_genes=shared_genes,
            lower_mats=lower_mats,
            upper_mats=upper_mats,
            overlaps=overlaps,
            lower_units_by_time=lower_units_by_time,
            upper_units_by_time=upper_units_by_time,
            upper_coords_by_time=upper_coords_by_time,
            lower_coords_by_time=lower_coords_by_time,
            feature_alignment_space="native_units",
            feature_names=feature_names,
            feature_blocks=feature_blocks,
            graph_summaries=graph_summaries,
            exports=exports,
            metadata={
                "network_method": self.network_method,
                "feature_source": "expression_plus_cci",
                "feature_alignment_space": "native_units",
                "intra_source": "expression",
                "inter_source": "cci_only",
                "uses_grn": False,
                "uses_cci": True,
                "uses_expression": True,
                "expression_feature_names": "intra_expr:<gene>",
                "cci_feature_names": "inter_cci:<ligand>-><receptor>[<lr_key>]",
                "feature_log1p": bool(cfg.feature_log1p),
                "native_cci_inter_use_expression_mask": bool(cfg.native_cci_inter_use_expression_mask),
                "shared_gene_policy": "expression_intersection_across_layers_and_time",
            },
            lower_assignments_by_time=lower_assignments_by_time,
            upper_assignments_by_time=upper_assignments_by_time,
            lower_graphs=lower_graphs,
            upper_graphs=upper_graphs,
            coverage_tables=coverage_tables,
            spot_correspondence_tables=spot_correspondence_tables,
            overlap_edge_tables=overlap_edge_tables,
            overlap_quality_summaries=overlap_quality_summaries,
        )

    def _expression_cci_summary(
        self,
        stage: str,
        lower_graph: LayerGraph,
        upper_graph: LayerGraph,
        lower_mat: np.ndarray,
        upper_mat: np.ndarray,
        cfg: TemporalRunConfig,
    ) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": self.network_method,
            "feature_alignment_space": "native_units",
            "feature_source": "expression_plus_cci",
            "lower_units": len(lower_graph.units),
            "upper_units": len(upper_graph.units),
            "shared_genes": len(lower_graph.shared_genes),
            "lower_intra_edges": int(len(lower_graph.intra_edges)),
            "lower_inter_edges": int(len(lower_graph.inter_edges)),
            "upper_intra_edges": int(len(upper_graph.intra_edges)),
            "upper_inter_edges": int(len(upper_graph.inter_edges)),
            "lower_matrix_shape": list(lower_mat.shape),
            "upper_matrix_shape": list(upper_mat.shape),
            "feature_log1p": bool(cfg.feature_log1p),
            "lower_graph_metadata": lower_graph.metadata,
            "upper_graph_metadata": upper_graph.metadata,
        }

