"""Staged orchestration for direct, symmetric GRN weight perturbations."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .condition import PAIR_IDS, PerturbationCondition, derived_seed
from .config import PerturbationConfig
from .grn_io import read_grn
from .materialize import materialize_source_grn
from .mirror_perturbation import apply_mirrored_weight_perturbation
from .motif_index import build_motif_index
from .optimal_bridge import run_direct_state_baseline, run_direct_state_condition
from .paths import ExperimentPaths
from .reporting import collect_summary, write_json, write_tables
from .selection import discover_region, selection_from_region
from .state_bridge import build_baseline_grn_states, build_propagated_grn_states
from .symmetry import evaluate_mirror_pair
from .topology_metrics import TARGET_METRIC, baseline_communities, compute_topology_metrics, metric_value


def _progress(message: str) -> None:
    """Emit an immediately visible, user-facing stage update."""
    print(message, flush=True)


def _spot_ids(path: Path) -> list[str]:
    table = pd.read_csv(path, sep=None, engine="python")
    if table.empty:
        raise ValueError(f"CCI index is empty: {path}")
    column = "spot_id" if "spot_id" in table.columns else table.columns[0]
    ids = table[column].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"CCI index has duplicate spot IDs: {path}")
    return ids


def _baseline_summary(*, paths: ExperimentPaths, inputs: dict[str, Path], source_sample: str, target_sample: str, config: PerturbationConfig, supplied_maturity_root: Path | None) -> dict[str, object]:
    training = paths.baseline / "training"
    if (training / "summary.json").is_file():
        _progress("[Baseline] 复用已完成的 baseline 二阶段训练结果")
        return collect_summary(training)
    _progress("[Baseline] 开始 baseline 二阶段训练")
    paths.baseline.mkdir(parents=True, exist_ok=False)
    g_raw_t, g_raw_tp = build_baseline_grn_states(baseline_inputs=inputs)
    record = run_direct_state_baseline(baseline_inputs=inputs, g_raw_t=g_raw_t, g_raw_tp=g_raw_tp, source_sample=source_sample, target_sample=target_sample, output_root=paths.baseline, spot_ids_t=_spot_ids(inputs["index_t"]), spot_ids_tp=_spot_ids(inputs["index_tp"]), k=config.k, device=config.device, supplied_maturity_root=supplied_maturity_root)
    write_json(paths.baseline / "result.json", record)
    _progress("[Baseline] baseline 二阶段训练完成")
    return dict(record["summary"])


def _manifest(*, condition: PerturbationCondition, selection, operation, baseline_metric: float, perturbed_metric: float, qc) -> dict[str, object]:
    return {"run_id": condition.run_id, "pair_id": condition.pair_id, "strength": condition.strength, "direction": condition.direction, "region_seed": condition.seed, "region_node_count": len(selection.region_nodes), "selected_node_count": len(selection.selected_nodes), "node_region_coverage": len(selection.selected_nodes) / max(len(selection.region_nodes), 1), "region_edge_count": len(selection.region_edges), "selected_edge_count": len(selection.selected_edges), "edge_region_coverage": len(selection.selected_edges) / max(len(selection.region_edges), 1), "baseline_target_metric": baseline_metric, "perturbed_target_metric": perturbed_metric, "operation": operation.as_dict(), "qc": qc.as_dict()}


def run_experiment(*, time_pair: str, source_sample: str, target_sample: str, baseline_inputs: dict[str, Path], output_root: Path, config: PerturbationConfig = PerturbationConfig(), supplied_maturity_root: Path | None = None, pair_ids: Iterable[str] = PAIR_IDS, strengths: Iterable[float] | None = None, evaluate: bool = True) -> list[dict[str, object]]:
    """Run direct graph generation and, after mirror QC, existing downstream stages."""
    _progress(f"[开始] 读取 baseline GRN 并计算拓扑特征：{time_pair}")
    paths = ExperimentPaths(Path(output_root), time_pair)
    graph = read_grn(Path(baseline_inputs["grn_t"]))
    communities = baseline_communities(graph)
    positive_weights = graph.adjacency[graph.adjacency > 0.0]
    baseline_scale = float(np.quantile(positive_weights, 0.95)) if positive_weights.size else 1.0
    motif_index = build_motif_index(graph) if np.all(graph.adjacency >= 0.0) else None
    baseline_metrics = compute_topology_metrics(graph, communities, baseline_scale=baseline_scale, motif_index=motif_index)
    paths.pair_root.mkdir(parents=True, exist_ok=True)
    write_json(paths.pair_root / "baseline_topology.json", baseline_metrics.as_dict())
    baseline_summary = _baseline_summary(paths=paths, inputs=baseline_inputs, source_sample=source_sample, target_sample=target_sample, config=config, supplied_maturity_root=supplied_maturity_root) if evaluate else {}
    if baseline_summary:
        pd.DataFrame([{"time_pair": time_pair, **baseline_summary}]).to_csv(paths.pair_root / "baseline_summary.csv", index=False)
    records: list[dict[str, object]] = []
    for pair_id in pair_ids:
        if pair_id not in PAIR_IDS:
            raise ValueError(f"Unknown pair ID: {pair_id}")
        region_seed = derived_seed(time_pair, pair_id, "region", config.random_seed)
        _progress(f"[区域] 搜索合适的扰动位置：{pair_id}")
        is_motif = pair_id in {"P6_triangle_intensity", "P7_feedforward_loop", "P8_feedback_cycle"}
        if is_motif and motif_index is None:
            raise ValueError("P6/P7/P8 require a non-negative baseline GRN; signed-weight handling must be audited first.")
        region = discover_region(graph, pair_id=pair_id, communities=communities, seed=region_seed, config=config, motif_index=motif_index)
        write_json(paths.region_file(pair_id), region.as_dict())
        for strength in strengths or config.strengths:
            _progress(f"[扰动] 准备 {pair_id}，strength={strength:g}")
            selection = selection_from_region(region, strength=strength)
            _progress(f"[扰动] {pair_id}：selected_edges={len(selection.selected_edges)}，mirror_fraction={config.mirror_fraction:g}")
            strength_root = paths.strength_root(pair_id, strength)
            write_json(strength_root / "selection.json", selection.as_dict())
            positive_graph, negative_graph, operation = apply_mirrored_weight_perturbation(graph, selected_edges=selection.selected_edges, mirror_fraction=config.mirror_fraction)
            qc = evaluate_mirror_pair(baseline=graph, positive=positive_graph, negative=negative_graph, selection=selection, mirror_tolerance=config.mirror_tolerance)
            outcomes = [("positive", positive_graph, operation, qc, compute_topology_metrics(positive_graph, communities, baseline_scale=baseline_scale, motif_index=motif_index)), ("negative", negative_graph, operation, qc, compute_topology_metrics(negative_graph, communities, baseline_scale=baseline_scale, motif_index=motif_index))]
            for direction, outcome_graph, operation, qc, outcome_metrics in outcomes:
                condition = PerturbationCondition(time_pair, pair_id, strength, direction, region_seed)
                write_json(paths.pair_qc(condition), qc.as_dict())
                _progress(f"[扰动] 开始 {condition.run_id}")
                condition_root = paths.condition_root(condition)
                topology = condition_root / "topology"
                topology.mkdir(parents=True, exist_ok=False)
                baseline_target = metric_value(baseline_metrics, pair_id)
                actual_target = metric_value(outcome_metrics, pair_id)
                write_json(topology / "topology_metrics.json", outcome_metrics.as_dict())
                write_json(topology / "perturbation_manifest.json", _manifest(condition=condition, selection=selection, operation=operation, baseline_metric=baseline_target, perturbed_metric=actual_target, qc=qc))
                result: dict[str, object] = {"time_pair": time_pair, "pair_id": pair_id, "target_metric": TARGET_METRIC[pair_id], "direction": direction, "strength": strength, "run_id": condition.run_id, "baseline_topology": baseline_target, "actual_topology": actual_target, "topology_delta": actual_target - baseline_target, "baseline_weight_scale": baseline_scale, **qc.as_dict()}
                if qc.overall_pass and evaluate:
                    _progress(f"[物化] 写入 T1 扰动 GRN：{condition.run_id}")
                    materialized = materialize_source_grn(baseline=graph, perturbed=outcome_graph, baseline_inputs=baseline_inputs, output_dir=condition_root, pair_id=pair_id)
                    try:
                        _progress(f"[传播] 计算 T1 GRN-state delta 并传播到 T2：{condition.run_id}")
                        states = build_propagated_grn_states(baseline_inputs=baseline_inputs, perturbed_grn_t=materialized.perturbed_grn, baseline_spot_pij=paths.baseline / "training" / "PIJ_micro_train.npy")
                        state_root = condition_root / "state"; state_root.mkdir(parents=True, exist_ok=True)
                        np.save(state_root / "g_raw_t_perturbed.npy", states.g_raw_t_perturbed); np.save(state_root / "g_raw_tp_perturbed.npy", states.g_raw_tp_perturbed)
                        np.save(state_root / "delta_g_raw_t.npy", states.delta_g_raw_t); np.save(state_root / "delta_g_raw_tp.npy", states.delta_g_raw_tp)
                        write_json(state_root / "state_provenance.json", {"intervention_time": "T1", "perturbation_space": "GRN_state", "t1_state_source": "baseline_expression_plus_perturbed_T1_GRN", "t2_state_source": "baseline_G_plus_baseline_PIJ_propagated_delta", "expression_materialization": False, "perturbed_h5ad_created": False})
                        maturity_root = supplied_maturity_root if supplied_maturity_root is not None else paths.baseline / "maturity"
                        _progress(f"[二阶段训练] 使用 direct GRN-state 进入 global two-stage training：{condition.run_id}")
                        training = run_direct_state_condition(baseline_inputs=baseline_inputs, g_raw_t=states.g_raw_t_perturbed, g_raw_tp=states.g_raw_tp_perturbed, output_root=condition_root, k=config.k, device=config.device, maturity_t=maturity_root / "source.csv", maturity_tp=maturity_root / "target.csv", spot_ids_t=_spot_ids(baseline_inputs["index_t"]), spot_ids_tp=_spot_ids(baseline_inputs["index_tp"]))
                        summary = dict(training["summary"])
                        result.update(summary); result.update({"g_raw_t_perturbed_path": str(state_root / "g_raw_t_perturbed.npy"), "g_raw_tp_perturbed_path": str(state_root / "g_raw_tp_perturbed.npy"), "delta_g_raw_t_path": str(state_root / "delta_g_raw_t.npy"), "delta_g_raw_tp_path": str(state_root / "delta_g_raw_tp.npy"), "baseline_delta_EI": float(baseline_summary["delta_EI_best_checkpoint"]), "perturbed_delta_EI": float(summary["delta_EI_best_checkpoint"]), "delta_delta_EI": float(summary["delta_EI_best_checkpoint"]) - float(baseline_summary["delta_EI_best_checkpoint"]), "baseline_CQ": float(baseline_summary["closure_quality"]), "perturbed_CQ": float(summary["closure_quality"]), "delta_CQ": float(summary["closure_quality"]) - float(baseline_summary["closure_quality"]), "training_status": "COMPLETED"})
                        write_json(condition_root / "result.json", {**result, "training_root": training["training_root"]})
                        _progress(f"[完成] {condition.run_id}：ΔEI={result['delta_delta_EI']:.6g}，ΔCQ={result['delta_CQ']:.6g}")
                    except (RuntimeError, ValueError, FileNotFoundError) as error:
                        result.update({"training_status": "DIRECT_STATE_FAILED", "direct_state_error": str(error)}); write_json(condition_root / "result.json", result)
                        _progress(f"[失败] {condition.run_id}：{error}")
                else:
                    result["training_status"] = "SKIPPED_MIRROR_QC" if not qc.overall_pass else "NOT_REQUESTED"; write_json(condition_root / "result.json", result)
                    _progress(f"[跳过] {condition.run_id}：{result['training_status']}")
                records.append(result)
    write_tables(paths.pair_root, records)
    _progress(f"[全部完成] {time_pair}：共生成 {len(records)} 个 perturbation condition 记录")
    return records
