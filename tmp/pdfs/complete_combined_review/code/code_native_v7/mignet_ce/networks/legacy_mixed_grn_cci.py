from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features import build_lower_graph_matrix, build_upper_graph_matrix, coverage_table
from mignet_ce.graph.builder import LayerGraph, build_layer_graph
from mignet_ce.io.loaders import (
    LayerDataResolver,
    LayerPaths,
    natural_sort,
    peek_h5ad_genes,
    peek_h5ad_units,
    read_expression_h5ad,
    read_grn_edges,
)
from mignet_ce.mapping import (
    build_overlap_edge_table,
    build_overlap_mapping,
    build_spot_correspondence_table,
    load_unit_assignments,
    summarize_overlap_quality,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.utils.coords import align_coords


class LegacyMixedGRNCCIBuilder:
    network_method = "legacy_mixed_grn_cci"
    inter_influence_mode = "product"
    inter_additive_cci_weight = 1.0
    inter_additive_grn_weight = 1.0
    inter_grn_pair_policy = "require_pair"
    include_intra_grn = True

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        all_paths = self._check_pair_paths(cfg, resolver, organ, pair)
        shared_genes = self._compute_shared_genes(cfg, all_paths, pair)
        stable_upper_units = self._stable_upper_units(cfg, all_paths, pair.upper_layer)

        lower_mats: List[np.ndarray] = []
        upper_mats: List[np.ndarray] = []
        overlaps = []
        lower_units_by_time: List[List[str]] = []
        upper_units_by_time: List[List[str]] = []
        lower_assignments_by_time: List[pd.DataFrame] = []
        upper_assignments_by_time: List[pd.DataFrame] = []
        coverage_tables: List[pd.DataFrame] = []
        spot_correspondence_tables: List[pd.DataFrame] = []
        overlap_edge_tables: List[pd.DataFrame] = []
        overlap_quality_summaries: List[dict[str, object]] = []
        graph_summaries: List[dict[str, object]] = []
        lower_graphs: List[LayerGraph] = []
        upper_graphs: List[LayerGraph] = []
        upper_coords_by_time: List[np.ndarray] = []
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

            lower_graph = build_layer_graph(
                layer_name=pair.lower_layer,
                time_point=stage,
                expression=lower_expr,
                paths=lower_paths,
                shared_genes=shared_genes,
                expr_threshold=cfg.expr_threshold,
                cci_min=cfg.cci_min,
                top_k_targets_per_regulator=cfg.top_k_targets_per_regulator,
                require_target_expression_for_inter=cfg.require_target_expression_for_inter,
                inter_influence_mode=self.inter_influence_mode,
                inter_additive_cci_weight=self.inter_additive_cci_weight,
                inter_additive_grn_weight=self.inter_additive_grn_weight,
                inter_grn_pair_policy=self.inter_grn_pair_policy,
                include_intra_grn=self.include_intra_grn,
            )
            upper_graph = build_layer_graph(
                layer_name=pair.upper_layer,
                time_point=stage,
                expression=upper_expr,
                paths=upper_paths,
                shared_genes=shared_genes,
                expr_threshold=cfg.expr_threshold,
                cci_min=cfg.cci_min,
                top_k_targets_per_regulator=cfg.top_k_targets_per_regulator,
                require_target_expression_for_inter=cfg.require_target_expression_for_inter,
                inter_influence_mode=self.inter_influence_mode,
                inter_additive_cci_weight=self.inter_additive_cci_weight,
                inter_additive_grn_weight=self.inter_additive_grn_weight,
                inter_grn_pair_policy=self.inter_grn_pair_policy,
                include_intra_grn=self.include_intra_grn,
            )

            lower_mat = build_lower_graph_matrix(lower_graph, overlap, feature_log1p=cfg.feature_log1p)
            upper_mat = build_upper_graph_matrix(upper_graph, stable_upper_units, feature_log1p=cfg.feature_log1p)
            lower_mats.append(lower_mat)
            upper_mats.append(upper_mat)
            overlaps.append(overlap)
            lower_units_by_time.append(lower_graph.units)
            upper_units_by_time.append(upper_graph.units)
            lower_assignments_by_time.append(lower_assignments.rows.copy())
            upper_assignments_by_time.append(upper_assignments.rows.copy())
            lower_graphs.append(lower_graph)
            upper_graphs.append(upper_graph)
            upper_coords_by_time.append(align_coords(upper_expr.coords, stable_upper_units))
            coverage_tables.append(coverage_table(stage, stable_upper_units, overlap.coverage_counts(), upper_graph.units))
            spot_correspondence_tables.append(spot_correspondence)
            overlap_edge_tables.append(overlap_edges)
            overlap_quality_summaries.append(overlap_quality)
            graph_summaries.append(self._graph_summary(stage, lower_graph, upper_graph, lower_mat, upper_mat))

            if cfg.export_graphs:
                exports[f"network_exports/{stage}_lower_intra_edges.csv"] = lower_graph.intra_edges.copy()
                exports[f"network_exports/{stage}_lower_inter_edges.csv"] = lower_graph.inter_edges.copy()
                exports[f"network_exports/{stage}_upper_intra_edges.csv"] = upper_graph.intra_edges.copy()
                exports[f"network_exports/{stage}_upper_inter_edges.csv"] = upper_graph.inter_edges.copy()

        feature_names = [f"legacy_intra_to_{unit}" for unit in stable_upper_units] + [f"legacy_inter_to_{unit}" for unit in stable_upper_units]
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
            feature_names=feature_names,
            feature_blocks={self.network_method: feature_names},
            graph_summaries=graph_summaries,
            exports=exports,
            metadata={
                "network_method": self.network_method,
                "feature_log1p": bool(cfg.feature_log1p),
                "feature_alignment_space": "stable_upper_units",
                "legacy_inter_influence_mode": self.inter_influence_mode,
                "legacy_inter_grn_pair_policy": self.inter_grn_pair_policy,
                "legacy_include_intra_grn": self.include_intra_grn,
                "legacy_additive_cci_weight": self.inter_additive_cci_weight,
                "legacy_additive_grn_weight": self.inter_additive_grn_weight,
                "legacy_shared_gene_policy": "expression_and_grn_intersection_across_layers_and_time",
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

    def _required_paths(self, paths: LayerPaths) -> List[Path]:
        required = [paths.h5ad, paths.grn_edges, paths.cci_manifest, paths.cci_index, paths.cci_lr_dir]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        return required

    def _check_pair_paths(
        self,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
        organ: str,
        pair: VerticalPairSpec,
    ) -> Dict[Tuple[str, str], LayerPaths]:
        all_paths: Dict[Tuple[str, str], LayerPaths] = {}
        missing: List[str] = []
        for stage in cfg.time_points:
            for layer in (pair.lower_layer, pair.upper_layer):
                paths = resolver.paths(layer, organ, str(stage))
                all_paths[(str(stage), layer)] = paths
                for required in self._required_paths(paths):
                    if not required.exists():
                        missing.append(str(required))
        if missing:
            preview = "\n".join(missing[:20])
            extra = f"\n... {len(missing) - 20} more" if len(missing) > 20 else ""
            raise FileNotFoundError(f"Missing required inputs for {organ} {pair.label()}:\n{preview}{extra}")
        return all_paths

    def _compute_shared_genes(
        self,
        cfg: TemporalRunConfig,
        all_paths: Dict[Tuple[str, str], LayerPaths],
        pair: VerticalPairSpec,
    ) -> List[str]:
        intersections: List[set[str]] = []
        for stage in cfg.time_points:
            lower_paths = all_paths[(str(stage), pair.lower_layer)]
            upper_paths = all_paths[(str(stage), pair.upper_layer)]
            lower_expr_genes = set(peek_h5ad_genes(lower_paths.h5ad))
            upper_expr_genes = set(peek_h5ad_genes(upper_paths.h5ad))
            lower_grn = read_grn_edges(lower_paths.grn_edges, cfg.top_k_targets_per_regulator)
            upper_grn = read_grn_edges(upper_paths.grn_edges, cfg.top_k_targets_per_regulator)
            lower_grn_genes = set(lower_grn["regulator"]).union(lower_grn["target"])
            upper_grn_genes = set(upper_grn["regulator"]).union(upper_grn["target"])
            intersections.append(lower_expr_genes & upper_expr_genes & lower_grn_genes & upper_grn_genes)
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError(f"Shared gene intersection is empty for {pair.label()}.")
        return shared

    def _stable_upper_units(
        self,
        cfg: TemporalRunConfig,
        all_paths: Dict[Tuple[str, str], LayerPaths],
        upper_layer: str,
    ) -> List[str]:
        units = set()
        for stage in cfg.time_points:
            units.update(peek_h5ad_units(all_paths[(str(stage), upper_layer)].h5ad))
        stable = natural_sort(units)
        if not stable:
            raise ValueError(f"No upper units found for {upper_layer}.")
        return stable

    def _graph_summary(
        self,
        stage: str,
        lower_graph: LayerGraph,
        upper_graph: LayerGraph,
        lower_mat: np.ndarray,
        upper_mat: np.ndarray,
    ) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": self.network_method,
            "inter_influence_mode": self.inter_influence_mode,
            "inter_grn_pair_policy": self.inter_grn_pair_policy,
            "include_intra_grn": self.include_intra_grn,
            "inter_additive_cci_weight": self.inter_additive_cci_weight,
            "inter_additive_grn_weight": self.inter_additive_grn_weight,
            "lower_units": len(lower_graph.units),
            "upper_units": len(upper_graph.units),
            "shared_genes": len(lower_graph.shared_genes),
            "lower_intra_edges": int(len(lower_graph.intra_edges)),
            "lower_inter_edges": int(len(lower_graph.inter_edges)),
            "upper_intra_edges": int(len(upper_graph.intra_edges)),
            "upper_inter_edges": int(len(upper_graph.inter_edges)),
            "lower_matrix_shape": list(lower_mat.shape),
            "upper_matrix_shape": list(upper_mat.shape),
        }
