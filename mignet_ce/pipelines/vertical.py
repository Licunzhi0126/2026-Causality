from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features import (
    build_lower_graph_matrix,
    build_upper_graph_matrix,
    coverage_table,
)
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
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.pij.base import MethodResult, TransitionKernels
from mignet_ce.pij.registry import build_method_result_and_kernels
from mignet_ce.pipelines.vertical_context import VerticalPairContext
from mignet_ce.utils.coords import align_coords
from mignet_ce.utils.matrix import save_transition_npz, serialize_metadata, transition_topk_table


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class VerticalMIGNetPipeline:
    def __init__(self, cfg: TemporalRunConfig):
        self.cfg = cfg
        self.cfg.validate()
        self.resolver = LayerDataResolver(cfg.data_root)
        self.metrics_engine = TemporalMetricsEngine()

    def run(self) -> pd.DataFrame:
        _ensure_dir(self.cfg.output_root)
        summary_rows: List[Dict[str, object]] = []
        metrics_tables: List[pd.DataFrame] = []
        for organ in self.cfg.organs:
            for pair in self.cfg.normalized_pairs():
                try:
                    metrics = self.run_pair(organ=str(organ), pair=pair)
                    metrics_tables.append(metrics)
                    summary_rows.append(
                        {
                            "organ": organ,
                            "lower_layer": pair.lower_layer,
                            "upper_layer": pair.upper_layer,
                            "pij_method": self.cfg.effective_pij_method(),
                            "status": "written",
                            "metrics_rows": int(len(metrics)),
                        }
                    )
                except Exception as exc:
                    summary_rows.append(
                        {
                            "organ": organ,
                            "lower_layer": pair.lower_layer,
                            "upper_layer": pair.upper_layer,
                            "pij_method": self.cfg.effective_pij_method(),
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )

        metrics = pd.concat(metrics_tables, ignore_index=True) if metrics_tables else self._empty_metrics()
        metrics.to_csv(self.cfg.output_root / "metrics.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(self.cfg.output_root / "run_summary.csv", index=False)
        with (self.cfg.output_root / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.cfg), handle, ensure_ascii=False, indent=2, default=_json_default)
        return metrics

    @staticmethod
    def _empty_metrics() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "pij_method",
                "organ",
                "lower_layer",
                "upper_layer",
                "time_pair",
                "lag",
                "H_base",
                "H_full",
                "H_macro",
                "EI_lower",
                "EI_upper",
                "EI_gain",
                "TE_raw",
                "TE",
                "DI_raw",
                "DI",
            ]
        )

    def _required_paths(self, paths: LayerPaths) -> List[Path]:
        required = [paths.h5ad, paths.grn_edges, paths.cci_manifest, paths.cci_index, paths.cci_lr_dir]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        return required

    def _check_pair_paths(self, organ: str, pair: VerticalPairSpec) -> Dict[Tuple[str, str], LayerPaths]:
        all_paths: Dict[Tuple[str, str], LayerPaths] = {}
        missing: List[str] = []
        for stage in self.cfg.time_points:
            for layer in (pair.lower_layer, pair.upper_layer):
                paths = self.resolver.paths(layer, organ, str(stage))
                all_paths[(str(stage), layer)] = paths
                for required in self._required_paths(paths):
                    if not required.exists():
                        missing.append(str(required))
        if missing:
            preview = "\n".join(missing[:20])
            extra = f"\n... {len(missing) - 20} more" if len(missing) > 20 else ""
            raise FileNotFoundError(f"Missing required inputs for {organ} {pair.label()}:\n{preview}{extra}")
        return all_paths

    def _compute_shared_genes(self, all_paths: Dict[Tuple[str, str], LayerPaths], pair: VerticalPairSpec) -> List[str]:
        intersections: List[set[str]] = []
        for stage in self.cfg.time_points:
            lower_paths = all_paths[(str(stage), pair.lower_layer)]
            upper_paths = all_paths[(str(stage), pair.upper_layer)]
            lower_expr_genes = set(peek_h5ad_genes(lower_paths.h5ad))
            upper_expr_genes = set(peek_h5ad_genes(upper_paths.h5ad))
            lower_grn = read_grn_edges(lower_paths.grn_edges, self.cfg.top_k_targets_per_regulator)
            upper_grn = read_grn_edges(upper_paths.grn_edges, self.cfg.top_k_targets_per_regulator)
            lower_grn_genes = set(lower_grn["regulator"]).union(lower_grn["target"])
            upper_grn_genes = set(upper_grn["regulator"]).union(upper_grn["target"])
            intersections.append(lower_expr_genes & upper_expr_genes & lower_grn_genes & upper_grn_genes)
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError(f"Shared gene intersection is empty for {pair.label()}.")
        return shared

    def _stable_upper_units(self, all_paths: Dict[Tuple[str, str], LayerPaths], upper_layer: str) -> List[str]:
        units = set()
        for stage in self.cfg.time_points:
            units.update(peek_h5ad_units(all_paths[(str(stage), upper_layer)].h5ad))
        stable = natural_sort(units)
        if not stable:
            raise ValueError(f"No upper units found for {upper_layer}.")
        return stable

    def _build_pair_context(self, organ: str, pair: VerticalPairSpec) -> VerticalPairContext:
        all_paths = self._check_pair_paths(organ, pair)
        shared_genes = self._compute_shared_genes(all_paths, pair)
        stable_upper_units = self._stable_upper_units(all_paths, pair.upper_layer)

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
        overlap_quality_summaries: List[Dict[str, object]] = []
        graph_summaries: List[Dict[str, object]] = []
        lower_graphs: List[LayerGraph] = []
        upper_graphs: List[LayerGraph] = []
        upper_coords_by_time: List[np.ndarray] = []

        for stage in map(str, self.cfg.time_points):
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
                expr_threshold=self.cfg.expr_threshold,
                cci_min=self.cfg.cci_min,
                top_k_targets_per_regulator=self.cfg.top_k_targets_per_regulator,
                require_target_expression_for_inter=self.cfg.require_target_expression_for_inter,
            )
            upper_graph = build_layer_graph(
                layer_name=pair.upper_layer,
                time_point=stage,
                expression=upper_expr,
                paths=upper_paths,
                shared_genes=shared_genes,
                expr_threshold=self.cfg.expr_threshold,
                cci_min=self.cfg.cci_min,
                top_k_targets_per_regulator=self.cfg.top_k_targets_per_regulator,
                require_target_expression_for_inter=self.cfg.require_target_expression_for_inter,
            )

            lower_mat = build_lower_graph_matrix(lower_graph, overlap, feature_log1p=self.cfg.feature_log1p)
            upper_mat = build_upper_graph_matrix(upper_graph, stable_upper_units, feature_log1p=self.cfg.feature_log1p)
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

        return VerticalPairContext(
            organ=organ,
            pair=pair,
            time_points=list(map(str, self.cfg.time_points)),
            stable_upper_units=stable_upper_units,
            shared_genes=shared_genes,
            lower_mats=lower_mats,
            upper_mats=upper_mats,
            overlaps=overlaps,
            lower_units_by_time=lower_units_by_time,
            upper_units_by_time=upper_units_by_time,
            lower_assignments_by_time=lower_assignments_by_time,
            upper_assignments_by_time=upper_assignments_by_time,
            lower_graphs=lower_graphs,
            upper_graphs=upper_graphs,
            upper_coords_by_time=upper_coords_by_time,
            coverage_tables=coverage_tables,
            spot_correspondence_tables=spot_correspondence_tables,
            overlap_edge_tables=overlap_edge_tables,
            overlap_quality_summaries=overlap_quality_summaries,
            graph_summaries=graph_summaries,
        )

    def run_pair(self, organ: str, pair: VerticalPairSpec) -> pd.DataFrame:
        context = self._build_pair_context(organ, pair)
        time_points = list(map(str, self.cfg.time_points))
        pairs = self.metrics_engine.build_time_pairs_all(time_points)
        method_result, kernels = build_method_result_and_kernels(context, self.cfg, pairs)
        metrics = self.metrics_engine.calculate_metrics_for_pairs(
            lower_feat=method_result.lower_features,
            upper_feat=method_result.upper_features,
            time_points=time_points,
            pairs=pairs,
            organ=organ,
            lower_layer=pair.lower_layer,
            upper_layer=pair.upper_layer,
            pij_method=self.cfg.effective_pij_method(),
            pij_temperature=self.cfg.pij_temperature,
            kraskov_k=self.cfg.kraskov_k,
            precomputed_p_lower=kernels.p_lower if kernels is not None else None,
            precomputed_p_upper=kernels.p_upper if kernels is not None else None,
            pairwise_lower_features=method_result.pairwise_lower_features,
            pairwise_upper_features=method_result.pairwise_upper_features,
        )

        self._export_pair_outputs(
            organ=organ,
            pair=pair,
            stable_upper_units=context.stable_upper_units,
            shared_genes=context.shared_genes,
            metrics=metrics,
            coverage_tables=context.coverage_tables,
            graph_summaries=context.graph_summaries,
            method_result=method_result,
            kernels=kernels,
            spot_correspondence_tables=context.spot_correspondence_tables,
            overlap_edge_tables=context.overlap_edge_tables,
            overlap_quality_summaries=context.overlap_quality_summaries,
            upper_units_by_time=context.upper_units_by_time,
        )
        return metrics

    @staticmethod
    def _graph_summary(stage: str, lower_graph: LayerGraph, upper_graph: LayerGraph, lower_mat: np.ndarray, upper_mat: np.ndarray) -> Dict[str, object]:
        return {
            "time_point": stage,
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

    def _export_pair_outputs(
        self,
        organ: str,
        pair: VerticalPairSpec,
        stable_upper_units: Sequence[str],
        shared_genes: Sequence[str],
        metrics: pd.DataFrame,
        coverage_tables: Sequence[pd.DataFrame],
        graph_summaries: Sequence[Dict[str, object]],
        method_result: MethodResult,
        kernels: TransitionKernels | None,
        spot_correspondence_tables: Sequence[pd.DataFrame],
        overlap_edge_tables: Sequence[pd.DataFrame],
        overlap_quality_summaries: Sequence[Dict[str, object]],
        upper_units_by_time: Sequence[Sequence[str]],
    ) -> None:
        pair_dir = self.cfg.output_root / "features" / organ / pair.label()
        _ensure_dir(pair_dir)
        metrics.to_csv(pair_dir / "metrics.csv", index=False)
        pd.DataFrame({"shared_gene": list(shared_genes)}).to_csv(pair_dir / "shared_genes.csv", index=False)
        pd.concat(coverage_tables, ignore_index=True).to_csv(pair_dir / "coverage.csv", index=False)
        with (pair_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "organ": organ,
                    "lower_layer": pair.lower_layer,
                    "upper_layer": pair.upper_layer,
                    "time_points": list(map(str, self.cfg.time_points)),
                    "pij_method": self.cfg.effective_pij_method(),
                    "embedding_method": self.cfg.embedding_method,
                    "method_metadata": method_result.method_metadata,
                    "feature_alignment_space": "stable_upper_units",
                    "lower_feature_meaning": "lower layer features aggregated to upper units by spot overlap",
                    "matching_diagnostics_available": True,
                    "laplacian_components": self.cfg.laplacian_components,
                    "laplacian_normalized": self.cfg.laplacian_normalized,
                    "stable_upper_unit_count": len(stable_upper_units),
                    "graph_summaries": list(graph_summaries),
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        correspondence_dir = pair_dir / "correspondence"
        _ensure_dir(correspondence_dir)
        pd.DataFrame({"upper_unit": list(stable_upper_units)}).to_csv(correspondence_dir / "stable_upper_units.csv", index=False)
        unit_presence_rows = []
        for stage, units in zip(map(str, self.cfg.time_points), upper_units_by_time):
            present = set(map(str, units))
            for unit in stable_upper_units:
                unit_presence_rows.append(
                    {
                        "stage": stage,
                        "upper_unit": unit,
                        "upper_unit_present": unit in present,
                    }
                )
        pd.DataFrame(unit_presence_rows).to_csv(correspondence_dir / "unit_presence.csv", index=False)
        for stage, spot_table, edge_table, quality in zip(
            map(str, self.cfg.time_points),
            spot_correspondence_tables,
            overlap_edge_tables,
            overlap_quality_summaries,
        ):
            spot_table.to_csv(correspondence_dir / f"{stage}_spot_correspondence.csv", index=False)
            edge_table.to_csv(correspondence_dir / f"{stage}_overlap_edges.csv", index=False)
            with (correspondence_dir / f"{stage}_overlap_quality.json").open("w", encoding="utf-8") as handle:
                json.dump(quality, handle, ensure_ascii=False, indent=2, default=_json_default)
        pd.DataFrame(overlap_quality_summaries).to_csv(correspondence_dir / "overlap_quality_summary.csv", index=False)

        if self.cfg.export_pij and kernels is not None:
            pij_dir = pair_dir / "pij"
            _ensure_dir(pij_dir)
            for (t0, t1), matrix in kernels.p_lower.items():
                label = f"{self.cfg.time_points[t0]}_to_{self.cfg.time_points[t1]}"
                save_transition_npz(pij_dir / f"{label}_lower_P.npz", matrix)
                transition_topk_table(
                    matrix,
                    source_units=stable_upper_units,
                    target_units=stable_upper_units,
                    time_pair=f"{self.cfg.time_points[t0]}->{self.cfg.time_points[t1]}",
                    space="lower",
                    top_k=self.cfg.export_pij_topk,
                ).to_csv(pij_dir / f"{label}_lower_P_topk.csv", index=False)
            for (t0, t1), matrix in kernels.p_upper.items():
                label = f"{self.cfg.time_points[t0]}_to_{self.cfg.time_points[t1]}"
                save_transition_npz(pij_dir / f"{label}_upper_P.npz", matrix)
                transition_topk_table(
                    matrix,
                    source_units=stable_upper_units,
                    target_units=stable_upper_units,
                    time_pair=f"{self.cfg.time_points[t0]}->{self.cfg.time_points[t1]}",
                    space="upper",
                    top_k=self.cfg.export_pij_topk,
                ).to_csv(pij_dir / f"{label}_upper_P_topk.csv", index=False)
            with (pij_dir / "kernel_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(serialize_metadata(kernels.kernel_metadata), handle, ensure_ascii=False, indent=2, default=_json_default)

        if self.cfg.export_features:
            for stage, low, up in zip(map(str, self.cfg.time_points), method_result.lower_features, method_result.upper_features):
                pd.DataFrame(low, index=stable_upper_units).to_csv(pair_dir / f"{stage}_lower_features_scaled.csv")
                pd.DataFrame(up, index=stable_upper_units).to_csv(pair_dir / f"{stage}_upper_features_scaled.csv")
