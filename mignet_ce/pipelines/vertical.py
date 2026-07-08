from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.features_native import build_native_feature_block_summary
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.io.pij_exports import export_pij_sparse_archive
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.base import MethodResult, TransitionKernels
from mignet_ce.pij.registry import build_method_result_and_kernels
from mignet_ce.utils.progress import (
    ProgressMonitor,
    QueueProgressReporter,
    emit_progress,
    get_progress_queue_or_none,
    get_progress_reporter,
    set_progress_reporter,
)


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
        pairs = self.cfg.normalized_pairs()
        progress_log = self.cfg.progress_log or (self.cfg.output_root / "progress.jsonl")
        with ProgressMonitor(
            enabled=self.cfg.progress,
            total_pairs=len(self.cfg.organs) * len(pairs),
            log_path=progress_log if self.cfg.progress else None,
            refresh_interval=self.cfg.progress_refresh_interval,
            process_safe=self.cfg.max_workers > 1,
        ):
            for organ in self.cfg.organs:
                for pair in pairs:
                    pair_start = time.perf_counter()
                    emit_progress("pair_start", status="start", phase="pair", organ=str(organ), pair=pair.label())
                    try:
                        metrics = self.run_pair(organ=str(organ), pair=pair)
                        metrics_tables.append(metrics)
                        summary_rows.append(
                            {
                                "organ": organ,
                                "lower_layer": pair.lower_layer,
                                "upper_layer": pair.upper_layer,
                                "network_method": self.cfg.network_method,
                                "pij_method": self.cfg.effective_pij_method(),
                                "status": "written",
                                "metrics_rows": int(len(metrics)),
                            }
                        )
                    except Exception as exc:
                        emit_progress(
                            "pair_error",
                            status="error",
                            phase="pair",
                            organ=str(organ),
                            pair=pair.label(),
                            message=f"{type(exc).__name__}: {exc}",
                            elapsed_seconds=time.perf_counter() - pair_start,
                        )
                        summary_rows.append(
                            {
                                "organ": organ,
                                "lower_layer": pair.lower_layer,
                                "upper_layer": pair.upper_layer,
                                "network_method": self.cfg.network_method,
                                "pij_method": self.cfg.effective_pij_method(),
                                "status": "error",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=8),
                            }
                        )
                    else:
                        emit_progress(
                            "pair_done",
                            status="done",
                            phase="pair",
                            organ=str(organ),
                            pair=pair.label(),
                            advance=1,
                            unit="pair",
                            elapsed_seconds=time.perf_counter() - pair_start,
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
                "network_method",
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
                "metric_alignment",
                "TE_raw",
                "TE",
                "DI_raw",
                "DI",
            ]
        )

    def _build_pair_context(self, organ: str, pair: VerticalPairSpec) -> NetworkContext:
        return get_network_builder(self.cfg.network_method).build_pair_context(
            organ=organ,
            pair=pair,
            cfg=self.cfg,
            resolver=self.resolver,
        )

    def run_pair(self, organ: str, pair: VerticalPairSpec) -> pd.DataFrame:
        scope = {
            "organ": organ,
            "pair": pair.label(),
            "lower_layer": pair.lower_layer,
            "upper_layer": pair.upper_layer,
        }
        previous_reporter = get_progress_reporter()
        progress_queue = get_progress_queue_or_none()
        if progress_queue is not None:
            set_progress_reporter(QueueProgressReporter(progress_queue, default_scope=scope))
        try:
            return self._run_pair_scoped(organ=organ, pair=pair)
        finally:
            set_progress_reporter(previous_reporter)

    def _emit_phase_start(self, stage: str) -> float:
        emit_progress("pair_phase_start", status="start", phase="pair", stage=stage)
        return time.perf_counter()

    def _emit_phase_done(self, stage: str, started: float) -> None:
        emit_progress(
            "pair_phase_done",
            status="done",
            phase="pair",
            stage=stage,
            advance=1,
            unit="phase",
            elapsed_seconds=time.perf_counter() - started,
        )

    def _emit_phase_skip(self, stage: str, message: str) -> None:
        emit_progress(
            "pair_phase_skip",
            status="skip",
            phase="pair",
            stage=stage,
            message=message,
            advance=1,
            unit="phase",
        )

    def _emit_phase_error(self, stage: str, started: float, exc: Exception) -> None:
        emit_progress(
            "pair_phase_error",
            status="error",
            phase="pair",
            stage=stage,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )

    def _run_pair_scoped(self, organ: str, pair: VerticalPairSpec) -> pd.DataFrame:
        phase_started = self._emit_phase_start("network_context")
        try:
            context = self._build_pair_context(organ, pair)
        except Exception as exc:
            self._emit_phase_error("network_context", phase_started, exc)
            raise
        self._emit_phase_done("network_context", phase_started)

        time_points = list(map(str, self.cfg.time_points))
        pairs = self.metrics_engine.build_time_pairs_all(time_points)

        phase_started = self._emit_phase_start("pij")
        try:
            method_result, kernels = build_method_result_and_kernels(context, self.cfg, pairs)
            export_kernels = kernels
            if self.cfg.export_pij and export_kernels is None:
                export_kernels = self._build_feature_transition_kernels(
                    method_result=method_result,
                    pairs=pairs,
                )
        except Exception as exc:
            self._emit_phase_error("pij", phase_started, exc)
            raise
        self._emit_phase_done("pij", phase_started)

        phase_started = self._emit_phase_start("metrics")
        emit_progress("metrics_start", phase="metrics", status="start", total=len(pairs), unit="time_pair")
        try:
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
                precomputed_p_lower=export_kernels.p_lower if export_kernels is not None else None,
                precomputed_p_upper=export_kernels.p_upper if export_kernels is not None else None,
                pairwise_lower_features=method_result.pairwise_lower_features,
                pairwise_upper_features=method_result.pairwise_upper_features,
                feature_alignment_space=context.feature_alignment_space,
            )
        except Exception as exc:
            emit_progress("metrics_error", phase="metrics", status="error", message=f"{type(exc).__name__}: {exc}")
            self._emit_phase_error("metrics", phase_started, exc)
            raise
        emit_progress("metrics_done", phase="metrics", status="done", extra={"rows": int(len(metrics))})
        self._emit_phase_done("metrics", phase_started)
        if "network_method" not in metrics.columns:
            metrics.insert(0, "network_method", context.network_method)

        phase_started = self._emit_phase_start("pij_export")
        if self.cfg.export_pij and export_kernels is not None:
            try:
                export_pij_sparse_archive(
                    cfg=self.cfg,
                    organ=organ,
                    pair=pair,
                    stable_upper_units=context.stable_upper_units,
                    kernels=export_kernels,
                    lower_units_by_time=context.lower_units_by_time,
                    upper_units_by_time=context.upper_units_by_time,
                    feature_alignment_space=context.feature_alignment_space,
                )
            except Exception as exc:
                self._emit_phase_error("pij_export", phase_started, exc)
                raise
            self._emit_phase_done("pij_export", phase_started)
        else:
            self._emit_phase_skip("pij_export", "PIJ export is disabled.")

        phase_started = self._emit_phase_start("pair_artifacts")
        if (
            self.cfg.export_pair_artifacts
            or self.cfg.export_graphs
            or self.cfg.export_raw_native_features
            or self.cfg.export_feature_diagnostics
        ):
            try:
                self._export_pair_outputs(
                    organ=organ,
                    pair=pair,
                    network_context=context,
                    stable_upper_units=context.stable_upper_units,
                    shared_genes=context.shared_genes,
                    metrics=metrics,
                    coverage_tables=context.coverage_tables,
                    graph_summaries=context.graph_summaries,
                    method_result=method_result,
                    spot_correspondence_tables=context.spot_correspondence_tables,
                    overlap_edge_tables=context.overlap_edge_tables,
                    overlap_quality_summaries=context.overlap_quality_summaries,
                    upper_units_by_time=context.upper_units_by_time,
                    lower_units_by_time=context.lower_units_by_time,
                )
            except Exception as exc:
                self._emit_phase_error("pair_artifacts", phase_started, exc)
                raise
            self._emit_phase_done("pair_artifacts", phase_started)
        else:
            self._emit_phase_skip("pair_artifacts", "Pair artifact export is disabled.")
        return metrics

    def _build_feature_transition_kernels(
        self,
        method_result: MethodResult,
        pairs: Sequence[tuple[int, int]],
    ) -> TransitionKernels:
        p_lower = {}
        p_upper = {}

        for t0, t1 in pairs:
            if method_result.pairwise_lower_features is not None and (t0, t1) in method_result.pairwise_lower_features:
                lower_source, lower_target = method_result.pairwise_lower_features[(t0, t1)]
            else:
                lower_source = method_result.lower_features[t0]
                lower_target = method_result.lower_features[t1]

            if method_result.pairwise_upper_features is not None and (t0, t1) in method_result.pairwise_upper_features:
                upper_source, upper_target = method_result.pairwise_upper_features[(t0, t1)]
            else:
                upper_source = method_result.upper_features[t0]
                upper_target = method_result.upper_features[t1]

            p_lower[(t0, t1)] = self.metrics_engine.build_transition_kernel(
                lower_source,
                lower_target,
                temperature=self.cfg.pij_temperature,
            )
            p_upper[(t0, t1)] = self.metrics_engine.build_transition_kernel(
                upper_source,
                upper_target,
                temperature=self.cfg.pij_temperature,
            )

        return TransitionKernels(
            p_lower=p_lower,
            p_upper=p_upper,
            kernel_metadata={
                "kernel_source": "feature_cosine_transition",
                "reason": "pij method returned MethodResult without explicit TransitionKernels",
                "pij_method": self.cfg.effective_pij_method(),
                "temperature": float(self.cfg.pij_temperature),
                "row_stochastic": True,
                "matrix_convention": (
                    "P[i,j] is transition probability from source-stage unit i "
                    "to target-stage unit j."
                ),
            },
        )

    def _export_pair_outputs(
        self,
        organ: str,
        pair: VerticalPairSpec,
        network_context: NetworkContext,
        stable_upper_units: Sequence[str],
        shared_genes: Sequence[str],
        metrics: pd.DataFrame,
        coverage_tables: Sequence[pd.DataFrame],
        graph_summaries: Sequence[Dict[str, object]],
        method_result: MethodResult,
        spot_correspondence_tables: Sequence[pd.DataFrame],
        overlap_edge_tables: Sequence[pd.DataFrame],
        overlap_quality_summaries: Sequence[Dict[str, object]],
        upper_units_by_time: Sequence[Sequence[str]],
        lower_units_by_time: Sequence[Sequence[str]],
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
                    "network_method": network_context.network_method,
                    "pij_method": self.cfg.effective_pij_method(),
                    "embedding_method": self.cfg.embedding_method,
                    "method_metadata": method_result.method_metadata,
                    "network_metadata": network_context.metadata,
                    "feature_blocks": network_context.feature_blocks,
                    "feature_alignment_space": network_context.feature_alignment_space,
                    "lower_feature_meaning": (
                        "lower layer features remain on native lower units"
                        if network_context.feature_alignment_space == "native_units"
                        else "lower layer features aggregated to upper units by spot overlap"
                    ),
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

        with (pair_dir / "feature_schema.json").open("w", encoding="utf-8") as handle:
            exported_feature_names = method_result.method_metadata.get("feature_names", network_context.feature_names)
            json.dump(
                {
                    "network_method": network_context.network_method,
                    "feature_names": exported_feature_names,
                    "network_feature_names": network_context.feature_names,
                    "feature_blocks": network_context.feature_blocks,
                    "metadata": network_context.metadata,
                    "method_metadata": method_result.method_metadata,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        if method_result.method_metadata.get("feature_source") == "pure_expression":
            pd.DataFrame({"gene": method_result.method_metadata.get("selected_genes", [])}).to_csv(
                pair_dir / "pure_expression_genes.csv",
                index=False,
            )
            with (pair_dir / "pure_expression_feature_schema.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "feature_source": "pure_expression",
                        "uses_grn": False,
                        "uses_cci": False,
                        "uses_legacy_graph": False,
                        "feature_names": exported_feature_names,
                        "selected_gene_count": method_result.method_metadata.get("selected_gene_count"),
                        "normalization": method_result.method_metadata.get("normalization"),
                        "gene_selection": method_result.method_metadata.get("gene_selection"),
                        "gene_scaler": method_result.method_metadata.get("gene_scaler"),
                        "feature_reduction": method_result.method_metadata.get("feature_reduction"),
                        "lower_aggregation": method_result.method_metadata.get("lower_aggregation"),
                        "upper_alignment_missing_policy": method_result.method_metadata.get("upper_alignment_missing_policy"),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )

        for relative_path, table in network_context.exports.items():
            export_path = pair_dir / relative_path
            _ensure_dir(export_path.parent)
            table.to_csv(export_path, index=False)

        if self.cfg.export_raw_native_features and network_context.feature_alignment_space == "native_units":
            for time_index, stage in enumerate(map(str, self.cfg.time_points)):
                lower_raw = network_context.lower_mats[time_index]
                upper_raw = network_context.upper_mats[time_index]
                lower_columns = (
                    network_context.feature_names
                    if len(network_context.feature_names) == lower_raw.shape[1]
                    else None
                )
                upper_columns = (
                    network_context.feature_names
                    if len(network_context.feature_names) == upper_raw.shape[1]
                    else None
                )
                pd.DataFrame(
                    lower_raw,
                    index=list(map(str, lower_units_by_time[time_index])),
                    columns=lower_columns,
                ).to_csv(pair_dir / f"{stage}_lower_features_raw_native.csv")
                pd.DataFrame(
                    upper_raw,
                    index=list(map(str, upper_units_by_time[time_index])),
                    columns=upper_columns,
                ).to_csv(pair_dir / f"{stage}_upper_features_raw_native.csv")

        if self.cfg.export_feature_diagnostics and network_context.feature_alignment_space == "native_units":
            diagnostics_dir = pair_dir / "diagnostics"
            _ensure_dir(diagnostics_dir)
            for time_index, stage in enumerate(map(str, self.cfg.time_points)):
                lower_summary = build_native_feature_block_summary(
                    network_context.lower_mats[time_index],
                    lower_units_by_time[time_index],
                    network_context.feature_names,
                    network_context.feature_blocks,
                    stage=stage,
                    layer_role="lower",
                )
                upper_summary = build_native_feature_block_summary(
                    network_context.upper_mats[time_index],
                    upper_units_by_time[time_index],
                    network_context.feature_names,
                    network_context.feature_blocks,
                    stage=stage,
                    layer_role="upper",
                )
                pd.concat([lower_summary, upper_summary], ignore_index=True).to_csv(
                    diagnostics_dir / f"{stage}_feature_block_summary.csv",
                    index=False,
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

        if self.cfg.export_features:
            for time_index, (stage, low, up) in enumerate(
                zip(map(str, self.cfg.time_points), method_result.lower_features, method_result.upper_features)
            ):
                feature_names = method_result.method_metadata.get("feature_names")
                low_columns = feature_names if isinstance(feature_names, list) and len(feature_names) == low.shape[1] else None
                up_columns = feature_names if isinstance(feature_names, list) and len(feature_names) == up.shape[1] else None
                if network_context.feature_alignment_space == "native_units":
                    lower_index = list(map(str, lower_units_by_time[time_index]))
                    upper_index = list(map(str, upper_units_by_time[time_index]))
                else:
                    lower_index = list(map(str, stable_upper_units))
                    upper_index = list(map(str, stable_upper_units))
                pd.DataFrame(low, index=lower_index, columns=low_columns).to_csv(pair_dir / f"{stage}_lower_features_scaled.csv")
                pd.DataFrame(up, index=upper_index, columns=up_columns).to_csv(pair_dir / f"{stage}_upper_features_scaled.csv")
