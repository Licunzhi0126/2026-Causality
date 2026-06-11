from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from mignet_ce.config import DEFAULT_DATA_ROOT, DEFAULT_WORK_ROOT
from mignet_ce.features import (
    aggregate_lower_features_to_upper,
    align_upper_features,
    build_lower_graph_matrix,
    build_upper_graph_matrix,
)
from mignet_ce.graph.builder import LayerGraph, build_layer_graph
from mignet_ce.io.cross_organ import CrossOrganDataResolver
from mignet_ce.io.loaders import LayerPaths, natural_sort, peek_h5ad_genes, peek_h5ad_units, read_expression_h5ad, read_grn_edges
from mignet_ce.mapping import build_overlap_mapping, load_unit_assignments
from mignet_ce.metrics import TemporalMetricsEngine


DEFAULT_ORGAN_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "output" / "mignet_organ"


@dataclass
class OrganPipelineConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_ORGAN_OUTPUT_ROOT
    lower_layer: str = "seurat_k40"
    upper_layer: str = "louvain_k150"
    macro_group_column: str = "organ"
    expected_organs: Sequence[str] = ("heart", "brain", "lung")
    time_points: Sequence[str] = ("11.5", "12.5", "13.5")
    strict_complete_organs: bool = True
    cci_scope: str = "all"
    expr_threshold: float = 0.0
    cci_min: float = 0.0
    top_k_targets_per_regulator: int = 20
    require_target_expression_for_inter: bool = True
    nmf_components: int = 5
    nmf_max_iter: int = 300
    nmf_seed: int = 42
    kraskov_k: int = 3
    feature_log1p: bool = True
    export_features: bool = True

    def validate(self) -> None:
        if self.cci_scope not in {"all", "intra", "inter"}:
            raise ValueError("cci_scope must be one of: all, intra, inter.")
        if len(self.expected_organs) < 2:
            raise ValueError("Organ-level metrics need at least two macro organs.")


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


def _filter_cci_scope(graph: LayerGraph, obs: pd.DataFrame, macro_group_column: str, cci_scope: str) -> LayerGraph:
    if cci_scope == "all" or graph.inter_edges.empty:
        return graph
    if macro_group_column not in obs.columns:
        raise KeyError(f"Missing macro group column {macro_group_column!r} in {graph.layer} obs.")
    group_by_unit = obs[macro_group_column].astype(str).to_dict()
    work = graph.inter_edges.copy()
    work["_src_group"] = work["src_unit"].map(group_by_unit)
    work["_dst_group"] = work["dst_unit"].map(group_by_unit)
    work = work.dropna(subset=["_src_group", "_dst_group"])
    if cci_scope == "intra":
        work = work[work["_src_group"] == work["_dst_group"]]
    elif cci_scope == "inter":
        work = work[work["_src_group"] != work["_dst_group"]]
    work = work.drop(columns=["_src_group", "_dst_group"])
    return LayerGraph(
        layer=graph.layer,
        time_point=graph.time_point,
        units=graph.units,
        genes=graph.genes,
        intra_edges=graph.intra_edges,
        inter_edges=work,
        shared_genes=graph.shared_genes,
    )


def _macro_weights(
    stable_upper_units: Sequence[str],
    upper_obs: pd.DataFrame,
    macro_group_column: str,
    expected_organs: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    unit_to_row = {unit: idx for idx, unit in enumerate(map(str, stable_upper_units))}
    organs = list(map(str, expected_organs))
    organ_to_col = {organ: idx for idx, organ in enumerate(organs)}
    weights = np.zeros((len(stable_upper_units), len(organs)), dtype=float)
    rows: List[Dict[str, object]] = []

    obs = upper_obs.copy()
    obs.index = obs.index.astype(str)
    for unit, row in obs.iterrows():
        if unit not in unit_to_row:
            continue
        organ = str(row[macro_group_column])
        if organ not in organ_to_col:
            continue
        raw_weight = row.get("spot_count", 1.0)
        weight = float(raw_weight) if pd.notna(raw_weight) else 1.0
        weights[unit_to_row[unit], organ_to_col[organ]] = max(weight, 0.0)
        rows.append({"upper_unit": unit, "organ": organ, "spot_count_weight": weight})

    return weights, pd.DataFrame(rows)


def _aggregate_to_macro(features: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    weighted = weights.T @ features
    denom = weights.sum(axis=0)
    macro = np.divide(weighted, denom[:, None], out=np.zeros_like(weighted), where=denom[:, None] > 0)
    return macro, denom


class OrganTemporalPipeline:
    def __init__(self, cfg: OrganPipelineConfig):
        self.cfg = cfg
        self.cfg.validate()
        self.resolver = CrossOrganDataResolver(cfg.data_root)
        self.metrics_engine = TemporalMetricsEngine()

    def run(self) -> pd.DataFrame:
        _ensure_dir(self.cfg.output_root)
        all_paths = self._check_paths()
        shared_genes = self._compute_shared_genes(all_paths)
        stable_upper_units = self._stable_upper_units(all_paths)

        lower_mats: List[np.ndarray] = []
        upper_mats: List[np.ndarray] = []
        overlaps = []
        upper_units_by_time: List[List[str]] = []
        macro_weights_by_time: List[np.ndarray] = []
        coverage_rows: List[pd.DataFrame] = []
        graph_summaries: List[Dict[str, object]] = []

        for stage in map(str, self.cfg.time_points):
            lower_paths = all_paths[(stage, self.cfg.lower_layer)]
            upper_paths = all_paths[(stage, self.cfg.upper_layer)]
            lower_expr = read_expression_h5ad(lower_paths.h5ad)
            upper_expr = read_expression_h5ad(upper_paths.h5ad)
            self._validate_organs(stage, lower_expr.obs, upper_expr.obs)

            lower_assignments = load_unit_assignments(self.cfg.lower_layer, lower_expr, lower_paths.spot_domain_map)
            upper_assignments = load_unit_assignments(self.cfg.upper_layer, upper_expr, upper_paths.spot_domain_map)
            overlap = build_overlap_mapping(lower_assignments, upper_assignments, lower_expr.units, stable_upper_units)

            lower_graph = build_layer_graph(
                layer_name=self.cfg.lower_layer,
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
                layer_name=self.cfg.upper_layer,
                time_point=stage,
                expression=upper_expr,
                paths=upper_paths,
                shared_genes=shared_genes,
                expr_threshold=self.cfg.expr_threshold,
                cci_min=self.cfg.cci_min,
                top_k_targets_per_regulator=self.cfg.top_k_targets_per_regulator,
                require_target_expression_for_inter=self.cfg.require_target_expression_for_inter,
            )
            lower_graph = _filter_cci_scope(lower_graph, lower_expr.obs, self.cfg.macro_group_column, self.cfg.cci_scope)
            upper_graph = _filter_cci_scope(upper_graph, upper_expr.obs, self.cfg.macro_group_column, self.cfg.cci_scope)

            lower_mat = build_lower_graph_matrix(lower_graph, overlap, feature_log1p=self.cfg.feature_log1p)
            upper_mat = build_upper_graph_matrix(upper_graph, stable_upper_units, feature_log1p=self.cfg.feature_log1p)
            weights, unit_weight_rows = _macro_weights(
                stable_upper_units=stable_upper_units,
                upper_obs=upper_expr.obs,
                macro_group_column=self.cfg.macro_group_column,
                expected_organs=self.cfg.expected_organs,
            )

            lower_mats.append(lower_mat)
            upper_mats.append(upper_mat)
            overlaps.append(overlap)
            upper_units_by_time.append(upper_graph.units)
            macro_weights_by_time.append(weights)
            coverage_rows.append(self._coverage_table(stage, overlap.coverage_counts(), weights, unit_weight_rows))
            graph_summaries.append(self._graph_summary(stage, lower_graph, upper_graph, lower_mat, upper_mat))

        W_lower_cells, _ = self.metrics_engine.temporal_joint_nmf(
            lower_mats,
            n_components=self.cfg.nmf_components,
            max_iter=self.cfg.nmf_max_iter,
            seed=self.cfg.nmf_seed,
        )
        W_upper_current, _ = self.metrics_engine.temporal_joint_nmf(
            upper_mats,
            n_components=self.cfg.nmf_components,
            max_iter=self.cfg.nmf_max_iter,
            seed=self.cfg.nmf_seed,
        )

        lower_macro_raw: List[np.ndarray] = []
        upper_macro_raw: List[np.ndarray] = []
        for idx in range(len(self.cfg.time_points)):
            lower_domain_feat, _ = aggregate_lower_features_to_upper(W_lower_cells[idx], overlaps[idx])
            upper_domain_feat = align_upper_features(W_upper_current[idx], upper_units_by_time[idx], stable_upper_units)
            lower_macro, _ = _aggregate_to_macro(lower_domain_feat, macro_weights_by_time[idx])
            upper_macro, _ = _aggregate_to_macro(upper_domain_feat, macro_weights_by_time[idx])
            lower_macro_raw.append(lower_macro)
            upper_macro_raw.append(upper_macro)

        lower_scaled, upper_scaled = self.metrics_engine.global_scale_features(lower_macro_raw, upper_macro_raw)
        k_used = self._kraskov_k_used(len(self.cfg.expected_organs), self.cfg.kraskov_k)
        metrics = self.metrics_engine.calculate_metrics_for_pairs(
            lower_feat=lower_scaled,
            upper_feat=upper_scaled,
            time_points=list(map(str, self.cfg.time_points)),
            pairs=self.metrics_engine.build_time_pairs_all(list(map(str, self.cfg.time_points))),
            organ="all_organs",
            lower_layer=self.cfg.lower_layer,
            upper_layer=self.cfg.upper_layer,
            kraskov_k=k_used,
        )
        metrics = metrics.rename(columns={"organ": "organ_scope"})
        metrics.insert(3, "cci_scope", self.cfg.cci_scope)
        metrics["entropy_sample_count"] = len(self.cfg.expected_organs)
        metrics["kraskov_k_used"] = k_used

        self._export_outputs(
            metrics=metrics,
            coverage=pd.concat(coverage_rows, ignore_index=True),
            graph_summaries=graph_summaries,
            stable_upper_units=stable_upper_units,
            shared_genes=shared_genes,
            lower_macro=lower_scaled,
            upper_macro=upper_scaled,
        )
        return metrics

    def _check_paths(self) -> Dict[Tuple[str, str], LayerPaths]:
        all_paths: Dict[Tuple[str, str], LayerPaths] = {}
        missing: List[str] = []
        for stage in map(str, self.cfg.time_points):
            for layer in (self.cfg.lower_layer, self.cfg.upper_layer):
                paths = self.resolver.paths(layer, stage)
                all_paths[(stage, layer)] = paths
                required = [paths.h5ad, paths.grn_edges, paths.cci_manifest, paths.cci_index, paths.cci_lr_dir]
                if paths.spot_domain_map is not None:
                    required.append(paths.spot_domain_map)
                for path in required:
                    if not path.exists():
                        missing.append(str(path))
        if missing:
            preview = "\n".join(missing[:20])
            extra = f"\n... {len(missing) - 20} more" if len(missing) > 20 else ""
            raise FileNotFoundError(f"Missing organ pipeline inputs:\n{preview}{extra}")
        return all_paths

    def _compute_shared_genes(self, all_paths: Dict[Tuple[str, str], LayerPaths]) -> List[str]:
        intersections: List[set[str]] = []
        for stage in map(str, self.cfg.time_points):
            lower_paths = all_paths[(stage, self.cfg.lower_layer)]
            upper_paths = all_paths[(stage, self.cfg.upper_layer)]
            lower_grn = read_grn_edges(lower_paths.grn_edges, self.cfg.top_k_targets_per_regulator)
            upper_grn = read_grn_edges(upper_paths.grn_edges, self.cfg.top_k_targets_per_regulator)
            intersections.append(
                set(peek_h5ad_genes(lower_paths.h5ad))
                & set(peek_h5ad_genes(upper_paths.h5ad))
                & set(lower_grn["regulator"]).union(lower_grn["target"])
                & set(upper_grn["regulator"]).union(upper_grn["target"])
            )
        shared = natural_sort(set.intersection(*intersections)) if intersections else []
        if not shared:
            raise ValueError("Shared gene intersection is empty for organ-level pipeline.")
        return shared

    def _stable_upper_units(self, all_paths: Dict[Tuple[str, str], LayerPaths]) -> List[str]:
        units = set()
        for stage in map(str, self.cfg.time_points):
            units.update(peek_h5ad_units(all_paths[(stage, self.cfg.upper_layer)].h5ad))
        stable = natural_sort(units)
        if not stable:
            raise ValueError(f"No upper units found for {self.cfg.upper_layer}.")
        return stable

    def _validate_organs(self, stage: str, lower_obs: pd.DataFrame, upper_obs: pd.DataFrame) -> None:
        if self.cfg.macro_group_column not in lower_obs.columns or self.cfg.macro_group_column not in upper_obs.columns:
            raise KeyError(f"Both layers must contain obs[{self.cfg.macro_group_column!r}].")
        if not self.cfg.strict_complete_organs:
            return
        expected = set(map(str, self.cfg.expected_organs))
        lower_seen = set(lower_obs[self.cfg.macro_group_column].astype(str))
        upper_seen = set(upper_obs[self.cfg.macro_group_column].astype(str))
        missing_lower = sorted(expected - lower_seen)
        missing_upper = sorted(expected - upper_seen)
        if missing_lower or missing_upper:
            raise ValueError(
                f"Stage {stage} is missing expected organs. "
                f"lower_missing={missing_lower}, upper_missing={missing_upper}. "
                "Use --allow-incomplete-organs only if this is intentional."
            )

    @staticmethod
    def _kraskov_k_used(n_macro_units: int, requested_k: int = 3) -> int:
        if n_macro_units < 2:
            raise ValueError("At least two macro units are required for organ-level metrics.")
        return max(1, min(int(requested_k), n_macro_units - 2 if n_macro_units > 2 else 1))

    def _coverage_table(
        self,
        stage: str,
        lower_overlap_counts: np.ndarray,
        macro_weights: np.ndarray,
        unit_weight_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for idx, organ in enumerate(map(str, self.cfg.expected_organs)):
            unit_mask = macro_weights[:, idx] > 0
            rows.append(
                {
                    "time_point": stage,
                    "organ": organ,
                    "upper_units": int(unit_mask.sum()),
                    "upper_spot_count_weight": float(macro_weights[:, idx].sum()),
                    "lower_overlap_count": float(lower_overlap_counts[unit_mask].sum()),
                    "complete": bool(unit_mask.any()),
                }
            )
        if not unit_weight_rows.empty:
            observed = set(unit_weight_rows["organ"].astype(str))
            for row in rows:
                row["observed"] = row["organ"] in observed
        return pd.DataFrame(rows)

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

    def _load_cci_scope_summary(self) -> pd.DataFrame:
        path = self.cfg.data_root / "manifests" / "cross_organ_cci_intra_inter_summary.csv"
        if not path.exists():
            return pd.DataFrame()
        summary = pd.read_csv(path)
        stages = tuple(map(str, self.cfg.time_points))
        layer_mask = summary["layer"].isin([self.cfg.lower_layer, self.cfg.upper_layer])
        stage_mask = summary["sample"].astype(str).apply(lambda value: any(value.endswith(f"_{stage}") for stage in stages))
        return summary[layer_mask & stage_mask].copy()

    def _export_outputs(
        self,
        metrics: pd.DataFrame,
        coverage: pd.DataFrame,
        graph_summaries: Sequence[Dict[str, object]],
        stable_upper_units: Sequence[str],
        shared_genes: Sequence[str],
        lower_macro: Sequence[np.ndarray],
        upper_macro: Sequence[np.ndarray],
    ) -> None:
        _ensure_dir(self.cfg.output_root)
        metrics.to_csv(self.cfg.output_root / "metrics.csv", index=False)
        coverage.to_csv(self.cfg.output_root / "coverage.csv", index=False)
        pd.DataFrame({"shared_gene": list(shared_genes)}).to_csv(self.cfg.output_root / "shared_genes.csv", index=False)
        cci_summary = self._load_cci_scope_summary()
        if not cci_summary.empty:
            cci_summary.to_csv(self.cfg.output_root / "cci_scope_summary.csv", index=False)

        if self.cfg.export_features:
            feature_dir = self.cfg.output_root / "features"
            _ensure_dir(feature_dir)
            for stage, lower_feat, upper_feat in zip(map(str, self.cfg.time_points), lower_macro, upper_macro):
                index = list(map(str, self.cfg.expected_organs))
                pd.DataFrame(lower_feat, index=index).to_csv(feature_dir / f"{stage}_lower_organ_features_scaled.csv")
                pd.DataFrame(upper_feat, index=index).to_csv(feature_dir / f"{stage}_upper_organ_features_scaled.csv")

        summary = {
            "config": asdict(self.cfg),
            "stable_upper_unit_count": len(stable_upper_units),
            "shared_gene_count": len(shared_genes),
            "graph_summaries": list(graph_summaries),
        }
        with (self.cfg.output_root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
