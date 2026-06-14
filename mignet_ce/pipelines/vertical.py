from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.metrics import TemporalMetricsEngine
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.base import MethodResult, TransitionKernels
from mignet_ce.pij.registry import build_method_result_and_kernels
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
                            "network_method": self.cfg.network_method,
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
                            "network_method": self.cfg.network_method,
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
        if "network_method" not in metrics.columns:
            metrics.insert(0, "network_method", context.network_method)

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
            kernels=kernels,
            spot_correspondence_tables=context.spot_correspondence_tables,
            overlap_edge_tables=context.overlap_edge_tables,
            overlap_quality_summaries=context.overlap_quality_summaries,
            upper_units_by_time=context.upper_units_by_time,
        )
        return metrics

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
                    "network_method": network_context.network_method,
                    "pij_method": self.cfg.effective_pij_method(),
                    "embedding_method": self.cfg.embedding_method,
                    "method_metadata": method_result.method_metadata,
                    "network_metadata": network_context.metadata,
                    "feature_blocks": network_context.feature_blocks,
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

        with (pair_dir / "feature_schema.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "network_method": network_context.network_method,
                    "feature_names": network_context.feature_names,
                    "feature_blocks": network_context.feature_blocks,
                    "metadata": network_context.metadata,
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
