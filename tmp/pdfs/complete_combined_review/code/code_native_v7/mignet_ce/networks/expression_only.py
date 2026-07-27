from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features import coverage_table
from mignet_ce.graph.builder import EDGE_COLUMNS, LayerGraph
from mignet_ce.io.loaders import (
    LayerDataResolver,
    LayerPaths,
    natural_sort,
    peek_h5ad_genes,
    peek_h5ad_units,
    read_expression_h5ad,
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


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


class ExpressionOnlyBuilder:
    network_method = "expression_only"

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

            lower_mat = lower_expr.expr.loc[:, shared_genes].to_numpy(dtype=float)
            upper_mat = upper_expr.expr.loc[:, shared_genes].to_numpy(dtype=float)
            lower_mats.append(lower_mat)
            upper_mats.append(upper_mat)
            overlaps.append(overlap)
            lower_units_by_time.append(lower_expr.units)
            upper_units_by_time.append(upper_expr.units)
            lower_assignments_by_time.append(lower_assignments.rows.copy())
            upper_assignments_by_time.append(upper_assignments.rows.copy())
            upper_coords_by_time.append(align_coords(upper_expr.coords, stable_upper_units))
            coverage_tables.append(coverage_table(stage, stable_upper_units, overlap.coverage_counts(), upper_expr.units))
            spot_correspondence_tables.append(spot_correspondence)
            overlap_edge_tables.append(overlap_edges)
            overlap_quality_summaries.append(overlap_quality)

            lower_graph = LayerGraph(
                layer=pair.lower_layer,
                time_point=stage,
                units=lower_expr.units,
                genes=list(shared_genes),
                intra_edges=_empty_edges(),
                inter_edges=_empty_edges(),
                shared_genes=list(shared_genes),
            )
            upper_graph = LayerGraph(
                layer=pair.upper_layer,
                time_point=stage,
                units=upper_expr.units,
                genes=list(shared_genes),
                intra_edges=_empty_edges(),
                inter_edges=_empty_edges(),
                shared_genes=list(shared_genes),
            )
            lower_graphs.append(lower_graph)
            upper_graphs.append(upper_graph)
            graph_summaries.append(self._expression_summary(stage, lower_expr.units, upper_expr.units, lower_mat, upper_mat, shared_genes))

        feature_names = [f"expression_gene_{gene}" for gene in shared_genes]
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
            feature_blocks={"pure_expression_raw": feature_names},
            graph_summaries=graph_summaries,
            exports={},
            metadata={
                "network_method": self.network_method,
                "feature_source": "pure_expression",
                "feature_alignment_space": "stable_upper_units",
                "uses_grn": False,
                "uses_cci": False,
                "uses_legacy_graph": False,
                "raw_expression_feature_dim": int(len(shared_genes)),
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
        required = [paths.h5ad]
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
            raise FileNotFoundError(f"Missing required expression-only inputs for {organ} {pair.label()}:\n{preview}{extra}")
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
            intersections.append(set(peek_h5ad_genes(lower_paths.h5ad)) & set(peek_h5ad_genes(upper_paths.h5ad)))
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError(f"Shared expression gene intersection is empty for {pair.label()}.")
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

    @staticmethod
    def _expression_summary(
        stage: str,
        lower_units: List[str],
        upper_units: List[str],
        lower_mat: np.ndarray,
        upper_mat: np.ndarray,
        shared_genes: List[str],
    ) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": ExpressionOnlyBuilder.network_method,
            "feature_source": "pure_expression",
            "uses_grn": False,
            "uses_cci": False,
            "uses_legacy_graph": False,
            "lower_units": len(lower_units),
            "upper_units": len(upper_units),
            "shared_genes": len(shared_genes),
            "lower_matrix_shape": list(lower_mat.shape),
            "upper_matrix_shape": list(upper_mat.shape),
        }
