"""Staged orchestration for direct, symmetric GRN weight perturbations."""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from time import time_ns
from typing import Iterable
import json

import numpy as np
import pandas as pd

from ..experiment_scheduler import Job, Training, load_prepared, load_transfer, run_jobs, save_prepared
from ..formal_protocol import maturity_columns
from ..post_perbutation.scheduler import run_lock
from ..randnetwork.audit import digest, file_hash, input_record
from .condition import MOTIF_IDS, PAIR_IDS, PerturbationCondition
from .config import PerturbationConfig
from .grn_io import read_grn
from .materialize import materialize_source_grn
from .mirror_perturbation import apply_mirrored_weight_perturbation
from .motif_index import build_motif_index, read_compact_index
from .optimal_bridge import (
    REQUIRED_OUTPUTS, finish_condition, prepare_condition, run_direct_state_baseline,
    run_direct_state_condition, validate_cached_training,
)
from .paths import ExperimentPaths
from .reporting import collect_summary, plot_results, write_json, write_tables
from .formal_macro_adapter import PERTURBATION_MACRO_SCHEMA
from .selection import discover_region, selection_from_region
from .state_bridge import build_propagated_grn_states, describe_array, describe_genes
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


def _baseline_summary(*, paths: ExperimentPaths, inputs: dict[str, Path], source_sample: str, target_sample: str, config: PerturbationConfig, supplied_maturity_root: Path | None, execution=None) -> dict[str, object]:
    training = paths.baseline / "training"
    if paths.baseline.exists():
        validate_cached_training(paths.baseline, k=config.k, device=config.device, frontend_options=config.frontend_options, training_options=config.training_options or None)
        _progress("[Baseline] 复用已完成的 baseline 二阶段训练结果")
        return collect_summary(training)
    _progress("[Baseline] 开始 baseline 二阶段训练")
    paths.baseline.mkdir(parents=True, exist_ok=False)
    try:
        if execution is None:
            record = run_direct_state_baseline(baseline_inputs=inputs, source_sample=source_sample, target_sample=target_sample, output_root=paths.baseline, spot_ids_t=_spot_ids(inputs["index_t"]), spot_ids_tp=_spot_ids(inputs["index_tp"]), k=config.k, device=config.device, supplied_maturity_root=supplied_maturity_root, frontend_options=config.frontend_options, training_options=config.training_options or None)
        else:
            job = Job("P0", _prepare_baseline, (paths, inputs, config, supplied_maturity_root))
            record, = run_jobs(paths.pair_root, "P0", [job], execution)
            if "error" in record:
                write_json(paths.baseline / "result.json", record)
                raise RuntimeError(f"Baseline training failed: {record['error']}")
    except ImportError as error:
        write_json(paths.baseline / "result.json", {
            "training_status": "BLOCKED_UPSTREAM_IMPORT", "error": str(error),
            "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA})
        raise ImportError(f"BLOCKED_UPSTREAM_IMPORT: {error}") from error
    write_json(paths.baseline / "result.json", record)
    _progress("[Baseline] baseline 二阶段训练完成")
    return dict(record["summary"])


def _manifest(*, condition: PerturbationCondition, selection, operation, baseline_metric: float, perturbed_metric: float, qc) -> dict[str, object]:
    return {"run_id": condition.run_id, "pair_id": condition.pair_id, "strength": condition.strength, "direction": condition.direction, "region_seed": condition.seed, "region_node_count": len(selection.region_nodes), "selected_node_count": len(selection.selected_nodes), "node_region_coverage": len(selection.selected_nodes) / max(len(selection.region_nodes), 1), "region_edge_count": len(selection.region_edges), "selected_edge_count": len(selection.selected_edges), "edge_region_coverage": len(selection.selected_edges) / max(len(selection.region_edges), 1), "baseline_target_metric": baseline_metric, "perturbed_target_metric": perturbed_metric, "operation": operation.as_dict(), "qc": qc.as_dict()}


def _prepare_condition_directory(root: Path, *, retry_failed: bool):
    if not root.exists():
        return None
    result_path = root / "result.json"
    if not result_path.is_file() and retry_failed:
        root.rename(root.with_name(f"{root.name}.failed-{time_ns()}"))
        return None
    if not result_path.is_file():
        raise FileExistsError(f"Cannot establish the status of condition {root}.")
    stamp = result_path.stat().st_mtime_ns
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = result.get("training_status")
    if status in {"COMPLETED", "NOT_REQUESTED", "SKIPPED_MIRROR_QC"}:
        return result
    if not retry_failed or status not in {"DIRECT_STATE_FAILED", "BLOCKED_UPSTREAM_IMPORT"}:
        raise FileExistsError(f"Condition {root} already exists with status {status}.")
    if all((root / "training" / name).is_file() for name in REQUIRED_OUTPUTS):
        raise FileExistsError(f"Condition {root} has all training outputs. Inspect the result before retrying.")
    archive = root.with_name(f"{root.name}.failed-{stamp}")
    if archive.exists():
        raise FileExistsError(f"Failure archive already exists: {archive}")
    root.rename(archive)
    _progress(f"[重试] 已归档失败条件：{archive}")
    return None



def _prepare_baseline(staging, paths, inputs, config, maturity_root):
    from .maturity_resolver import resolve_maturity

    maturity_t, maturity_tp, maturity_record = resolve_maturity(supplied_root=maturity_root)
    prepared, training = prepare_condition(
        baseline_inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, output_root=paths.baseline,
        spot_ids_t=_spot_ids(inputs["index_t"]), spot_ids_tp=_spot_ids(inputs["index_tp"]),
        k=config.k, device=config.device, frontend_options=config.frontend_options,
        training_options=config.training_options)
    cache = staging / "prepared.pkl"
    save_prepared(cache, prepared)
    return Training(_train_baseline, (cache, training, config, maturity_record), (training.out_dir,))


def _train_baseline(device, cache, training, config, maturity_record):
    result = finish_condition(load_prepared(cache), replace(training, device=device),
                              config.frontend_options, config.training_options, baseline=True)
    return {"maturity": maturity_record, **result}


def _checked_baseline(paths, inputs, source_sample, target_sample, config, maturity_root,
                      experiment_id, retry_failed, execution):
    identity_path = paths.baseline / "identity.json"
    if paths.baseline.exists() and not identity_path.is_file():
        if not retry_failed:
            raise ValueError("Incomplete P0. Use --retry-failed to archive its attempt and retry.")
        paths.baseline.rename(paths.baseline.with_name(f"baseline.failed-{time_ns()}"))
    required = [paths.baseline / "training" / name for name in REQUIRED_OUTPUTS]
    required.extend(paths.baseline / name for name in ("formal_assets.pkl", "perturbation_provenance.json"))
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        hashes = {file.name: file_hash(file) for file in required}
        if identity != {"experiment_id": experiment_id, "artifacts": hashes}:
            raise ValueError("P0 artifact or configuration mismatch; use a new --output-root")
        if execution is not None:
            run_jobs(paths.pair_root, "P0", [], execution, cached=1)
    summary = _baseline_summary(paths=paths, inputs=inputs, source_sample=source_sample, target_sample=target_sample,
                                config=config, supplied_maturity_root=maturity_root, execution=execution)
    if not identity_path.is_file():
        hashes = {file.name: file_hash(file) for file in required}
        write_json(identity_path, {"experiment_id": experiment_id, "artifacts": hashes})
    return summary, digest({"experiment_id": experiment_id, "artifacts": hashes})


@lru_cache(maxsize=1)
def _load_assets(path):
    return load_transfer(path)


@lru_cache(maxsize=1)
def _load_index(path):
    return read_compact_index(path)


@lru_cache(maxsize=1)
def _load_graph(path):
    return read_grn(path)


def _compact_index(paths, graph, source_hash, workers):
    directory = paths.pair_root / f"motifs-{source_hash[:12]}"
    marker = directory / "ready.json"
    def counts(index):
        return {"triangles": len(index.triangles), "feedforward": len(index.feedforward_loops),
                "feedback": len(index.feedback_cycles), **{key: len(values) for key, values in index.template_occurrences.items()}}

    if marker.is_file():
        index = _load_index(directory)
        if json.loads(marker.read_text(encoding="utf-8")) != {"source_sha256": source_hash, "counts": counts(index)}:
            raise ValueError(f"Motif cache {directory} does not match its recorded index")
        return directory, index
    if directory.exists():
        directory.rename(directory.with_name(f"{directory.name}.incomplete-{time_ns()}"))
    index = build_motif_index(graph, directory, workers=workers)
    write_json(marker, {"source_sha256": source_hash, "counts": counts(index)})
    return directory, index


def _complete_condition(result, training, baseline, condition_root):
    summary = dict(training["summary"])
    state_root = condition_root / "state"
    metrics = {
        "baseline_delta_EI": float(baseline["delta_EI_best_checkpoint"]),
        "perturbed_delta_EI": float(summary["delta_EI_best_checkpoint"]),
        "baseline_CQ": float(baseline["closure_quality"]),
        "perturbed_CQ": float(summary["closure_quality"]),
    }
    completed = {**result, **summary, **metrics, "training_status": "COMPLETED",
        "delta_delta_EI": metrics["perturbed_delta_EI"] - metrics["baseline_delta_EI"],
        "delta_CQ": metrics["perturbed_CQ"] - metrics["baseline_CQ"],
        "training_root": training["training_root"],
        **{f"{name}_path": str(state_root / f"{name}.npy") for name in STATE_ARRAYS}}
    write_json(condition_root / "result.json", completed)
    return completed


STATE_ARRAYS = ("g_raw_t_perturbed", "g_raw_tp_perturbed", "delta_g_raw_t", "delta_g_raw_tp")


def _train_condition(device, cache, train_config, config, result, baseline, condition_root):
    record = finish_condition(load_prepared(cache), replace(train_config, device=device),
                              config.frontend_options, config.training_options)
    return _complete_condition(result, record, baseline, condition_root)


def _prepare_job(staging, condition, selection, paths, inputs, config, maturity_root, communities,
                 baseline_metrics, baseline_scale, baseline, index_path, experiment_id, baseline_id):
    graph = _load_graph(inputs["grn_t"])
    index = _load_index(index_path) if index_path else None
    return _prepare_condition(staging, condition, selection, paths, inputs, config, maturity_root,
                              communities, baseline_metrics, baseline_scale, baseline, graph, index,
                              experiment_id, baseline_id, evaluate=maturity_root is not None)


def _prepare_condition(staging, condition, selection, paths, inputs, config, maturity_root, communities,
                       baseline_metrics, baseline_scale, baseline, graph, index, experiment_id, baseline_id,
                       *, evaluate):
    pair_id = condition.pair_id
    condition_root = paths.condition_root(condition)
    positive, negative, operation = apply_mirrored_weight_perturbation(
        graph, selected_edges=selection.selected_edges, mirror_fraction=config.mirror_fraction)
    qc = evaluate_mirror_pair(baseline=graph, positive=positive, negative=negative,
                              selection=selection, mirror_tolerance=config.mirror_tolerance)
    if condition.direction == "positive" or not paths.pair_qc(condition).exists():
        write_json(paths.pair_qc(condition), qc.as_dict())
    outcome_graph = positive if condition.direction == "positive" else negative
    outcome_metrics = compute_topology_metrics(outcome_graph, communities, baseline_scale=baseline_scale, motif_index=index)
    topology = condition_root / "topology"
    topology.mkdir(parents=True, exist_ok=False)
    baseline_target = metric_value(baseline_metrics, pair_id)
    actual_target = metric_value(outcome_metrics, pair_id)
    write_json(topology / "topology_metrics.json", outcome_metrics.as_dict())
    write_json(topology / "perturbation_manifest.json", _manifest(
        condition=condition, selection=selection, operation=operation, baseline_metric=baseline_target,
        perturbed_metric=actual_target, qc=qc))
    motif_count = None
    if pair_id in MOTIF_IDS:
        occurrences = {"P6_triangle_intensity": index.triangles, "P7_feedforward_loop": index.feedforward_loops,
                       "P8_feedback_cycle": index.feedback_cycles, **index.template_occurrences}
        motif_count = len(occurrences[pair_id])
    diagnostics = {"baseline_motif_count": motif_count, "selection_status": (
        "NO_BASELINE_MOTIF" if motif_count == 0 else "SELECTED" if selection.selected_edges else "NO_SELECTABLE_EDGES"),
        "selected_edge_count": len(selection.selected_edges)}
    write_json(topology / "motif_diagnostics.json", diagnostics)
    result = {
        "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA, "experiment_id": experiment_id, "baseline_id": baseline_id,
        "time_pair": condition.time_pair, "pair_id": pair_id, "target_metric": TARGET_METRIC[pair_id],
        "direction": condition.direction, "strength": condition.strength, "run_id": condition.run_id,
        "region_seed": condition.seed, "train_seed": config.training_options.get("seed"),
        "baseline_topology": baseline_target, "actual_topology": actual_target,
        "topology_delta": actual_target - baseline_target, "baseline_weight_scale": baseline_scale,
        **diagnostics, **qc.as_dict(),
    }
    if not (qc.overall_pass and evaluate):
        result["training_status"] = "NOT_REQUESTED" if qc.overall_pass else "SKIPPED_MIRROR_QC"
        write_json(condition_root / "result.json", result)
        _progress(f"[跳过] {condition.run_id}：{result['training_status']}")
        return result
    write_json(condition_root / "prepared_result.json", result)
    state_checks = {"stage": "GRN_IO", "graph_t_baseline": describe_genes(graph.nodes, graph.adjacency),
                    "graph_t_perturbed": describe_genes(outcome_graph.nodes, outcome_graph.adjacency)}
    try:
        _progress(f"[物化] 写入 T1 扰动 GRN：{condition.run_id}")
        materialized = materialize_source_grn(baseline=graph, perturbed=outcome_graph, baseline_inputs=inputs,
                                              output_dir=condition_root, pair_id=pair_id)
        state_checks["grn_round_trip_verified"] = True
        baseline_pij = paths.baseline / "training" / "PIJ_micro_train.npy"
        _progress(f"[传播] 计算 T1 GRN-state delta 并传播到 T2：{condition.run_id}")
        states = build_propagated_grn_states(
            baseline_inputs=inputs, perturbed_grn_t=materialized.perturbed_grn, baseline_spot_pij=baseline_pij,
            frontend_options=config.frontend_options, baseline_graph=graph, perturbed_graph=materialized.graph,
            diagnostics=state_checks, baseline_assets=_load_assets(paths.baseline / "formal_assets.pkl"))
        state_root = condition_root / "state"
        state_root.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name in STATE_ARRAYS:
            values = getattr(states, name)
            np.save(state_root / f"{name}.npy", values)
            arrays[name] = {**describe_array(values), "l2_norm": float(np.linalg.norm(values))}
        write_json(state_root / "state_provenance.json", {
            "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA, "baseline_id": baseline_id,
            "intervention_time": "T1", "perturbation_space": "GRN_state",
            "t1_state_source": "baseline_expression_plus_perturbed_T1_GRN",
            "t2_state_source": "baseline_G_plus_baseline_PIJ_propagated_delta",
            "expression_materialization": False, "perturbed_h5ad_created": False,
            "inputs": {key: str(Path(value).resolve()) for key, value in inputs.items()},
            "perturbed_grn_t": str(materialized.perturbed_grn.resolve()),
            "grn_nodes": str(materialized.perturbed_grn.with_name("grn_nodes.json").resolve()),
            "baseline_spot_pij": str(baseline_pij.resolve()),
            "baseline_assignments": {suffix: str((baseline_pij.parent / f"assignments_{suffix}.csv").resolve())
                                     for suffix in ("t", "tp")},
            "checks": state_checks, "arrays": arrays,
        })
        _progress(f"[坐标] Spot GRN：T1={arrays['g_raw_t_perturbed']}，T2={arrays['g_raw_tp_perturbed']}，PIJ={state_checks['spot_pij']}")
        state_checks["stage"] = "TRAIN_PREP"
        kwargs = dict(baseline_inputs=inputs, states=states, frontend_options=config.frontend_options,
                      training_options=config.training_options, output_root=condition_root, k=config.k,
                      device=config.device, maturity_t=maturity_root / "source.csv", maturity_tp=maturity_root / "target.csv",
                      spot_ids_t=list(states.spot_ids_t), spot_ids_tp=list(states.spot_ids_tp))
        if staging is None:
            _progress(f"[二阶段训练] 使用 direct GRN-state 进入 global two-stage training：{condition.run_id}")
            training = run_direct_state_condition(**kwargs, diagnostics=state_checks)
            completed = _complete_condition(result, training, baseline, condition_root)
            _progress(f"[完成] {condition.run_id}：ΔEI={completed['delta_delta_EI']:.6g}，ΔCQ={completed['delta_CQ']:.6g}")
            return completed
        prepared, training = prepare_condition(**kwargs)
        cache = staging / "prepared.pkl"
        save_prepared(cache, prepared)
        return Training(_train_condition, (cache, training, config, result, baseline, condition_root), (training.out_dir,))
    except Exception as error:
        result.update(training_status="BLOCKED_UPSTREAM_IMPORT" if isinstance(error, ImportError) else "DIRECT_STATE_FAILED",
                      failure_stage=state_checks["stage"], error_type=type(error).__name__,
                      direct_state_error=str(error), diagnostics=state_checks)
        write_json(condition_root / "result.json", result)
        _progress(f"[失败] {condition.run_id} [{state_checks['stage']}]：{error}")
        return result


def run_experiment(*, time_pair: str, source_sample: str, target_sample: str, baseline_inputs: dict[str, Path],
                   output_root: Path, config: PerturbationConfig = PerturbationConfig(),
                   supplied_maturity_root: Path | None = None, pair_ids: Iterable[str] = PAIR_IDS,
                   strengths: Iterable[float] | None = None, evaluate: bool = True,
                   retry_failed: bool = False, execution=None) -> list[dict[str, object]]:
    """Train one audited P0 and evaluate all requested mirrored conditions against it."""
    paths = ExperimentPaths(Path(output_root).resolve(), time_pair)
    inputs = {key: Path(value).resolve() for key, value in baseline_inputs.items()}
    maturity_root = Path(supplied_maturity_root).resolve() if supplied_maturity_root is not None else None
    if evaluate and maturity_root is None:
        raise ValueError("Formal training requires --maturity-root")
    if evaluate:
        config = config.formal()
        columns = maturity_columns(tuple(maturity_root / f"{stage}.csv" for stage in ("source", "target")))
        config = replace(config, frontend_options={**columns, **config.frontend_options})
    pair_ids, strengths = tuple(dict.fromkeys(pair_ids)), tuple(dict.fromkeys(strengths or config.strengths))
    if not pair_ids or set(pair_ids) - set(PAIR_IDS):
        raise ValueError(f"Invalid topology group selection: {pair_ids}")
    if not strengths or any(not 0 < strength < 1 for strength in strengths):
        raise ValueError("Every perturbation strength must be in (0, 1)")
    with run_lock(paths.pair_root):
        return _run_locked(paths, inputs, source_sample, target_sample, config, maturity_root, pair_ids, strengths,
                           evaluate, retry_failed, execution)


def _run_locked(paths, inputs, source_sample, target_sample, config, maturity_root, pair_ids, strengths,
                evaluate, retry_failed, execution):
    frozen = dict(inputs)
    if evaluate:
        frozen.update({f"maturity_{stage}": maturity_root / f"{stage}.csv" for stage in ("source", "target")})
    input_identity = input_record(frozen)
    code_root = Path(__file__).resolve().parents[3]
    code = {str(path.relative_to(code_root)): file_hash(path)
            for folder in ("mignet_ce", "wyt_deltaei_coarse_grain") for path in (code_root / folder).rglob("*.py")}
    controls = asdict(config)
    controls.pop("strengths")
    identity = {"time_pair": paths.time_pair, "inputs": input_identity, "evaluate": evaluate,
                "config": controls, "implementation": digest(code)}
    experiment_id = digest(identity)
    manifest = paths.pair_root / "experiment.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")).get("experiment_id") != experiment_id:
            raise ValueError("Perturbation input or configuration mismatch; use a new --output-root")
    else:
        if paths.baseline.exists() or (paths.pair_root / "perturbation_results.csv").exists():
            raise ValueError("Existing results lack current provenance; use a new --output-root")
        write_json(manifest, {**identity, "experiment_id": experiment_id})

    baseline, baseline_id = {}, None
    if evaluate:
        baseline, baseline_id = _checked_baseline(paths, inputs, source_sample, target_sample, config, maturity_root,
                                                  experiment_id, retry_failed, execution)
        pd.DataFrame([{"time_pair": paths.time_pair, "baseline_id": baseline_id, **baseline}]).to_csv(
            paths.pair_root / "baseline_summary.csv", index=False)
    _progress(f"[开始] 读取 baseline GRN 并计算拓扑特征：{paths.time_pair}")
    graph = read_grn(inputs["grn_t"])
    communities = baseline_communities(graph)
    positives = graph.adjacency[graph.adjacency > 0]
    scale = float(np.quantile(positives, 0.95)) if positives.size else 1.0
    index_path, index = None, None
    if np.all(graph.adjacency >= 0):
        index_path, index = _compact_index(paths, graph, input_identity["grn_t"]["sha256"],
                                          execution.cpu_workers if execution else 1)
    if set(pair_ids) & MOTIF_IDS and index is None:
        raise ValueError("Motif conditions require non-negative GRN weights")
    baseline_metrics = compute_topology_metrics(graph, communities, baseline_scale=scale, motif_index=index)
    write_json(paths.pair_root / "baseline_topology.json", baseline_metrics.as_dict())

    records, jobs, scheduled = [], [], []
    for pair_id in pair_ids:
        _progress(f"[区域] 搜索合适的扰动位置：{pair_id}")
        region = discover_region(graph, pair_id=pair_id, communities=communities, seed=config.random_seed,
                                 config=config, motif_index=index)
        write_json(paths.region_file(pair_id), region.as_dict())
        for strength in strengths:
            _progress(f"[扰动] 准备 {pair_id}，strength={strength:g}")
            selection = selection_from_region(region, strength=strength)
            write_json(paths.strength_root(pair_id, strength) / "selection.json", selection.as_dict())
            for direction in ("positive", "negative"):
                condition = PerturbationCondition(paths.time_pair, pair_id, strength, direction, config.random_seed)
                condition_root = paths.condition_root(condition)
                cached = _prepare_condition_directory(condition_root, retry_failed=retry_failed and evaluate)
                if cached is not None:
                    if cached.get("experiment_id") != experiment_id or cached.get("baseline_id") != baseline_id:
                        raise ValueError(f"Condition references a different P0 or configuration: {condition.run_id}")
                    if evaluate and cached["training_status"] == "COMPLETED":
                        validate_cached_training(condition_root, k=config.k, device=config.device,
                                                 frontend_options=config.frontend_options,
                                                 training_options=config.training_options)
                    records.append(cached)
                    _progress(f"[跳过] 复用已完成条件：{condition.run_id}")
                    continue
                if execution is None:
                    records.append(_prepare_condition(None, condition, selection, paths, inputs, config, maturity_root,
                        communities, baseline_metrics, scale, baseline, graph, index, experiment_id, baseline_id,
                        evaluate=evaluate))
                else:
                    scheduled.append(condition)
                    jobs.append(Job(condition.run_id, _prepare_job, (
                        condition, selection, paths, inputs, config, maturity_root if evaluate else None,
                        communities, baseline_metrics, scale, baseline, index_path, experiment_id, baseline_id)))
    if execution is not None:
        def save_result(position, outcome):
            condition = scheduled[position]
            root = paths.condition_root(condition)
            if "error" in outcome:
                prep = root / "prepared_result.json"
                record = json.loads(prep.read_text(encoding="utf-8")) if prep.is_file() else {}
                outcome = {**record, "run_id": condition.run_id, "pair_id": condition.pair_id, "strength": condition.strength,
                           "direction": condition.direction, "time_pair": condition.time_pair,
                           "experiment_id": experiment_id, "baseline_id": baseline_id,
                           "training_status": "DIRECT_STATE_FAILED", "failure_stage": outcome["stage"],
                           "direct_state_error": outcome["error"]}
                write_json(root / "result.json", outcome)
            elif "peak_cuda_reserved_bytes" in outcome:
                write_json(root / "result.json", outcome)
            records.append(outcome)
        run_jobs(paths.pair_root, "GRN perturbations", jobs, execution, cached=len(records), on_result=save_result)
    order = {condition.run_id: i for i, condition in enumerate(
        PerturbationCondition(paths.time_pair, pair, strength, direction, config.random_seed)
        for pair in pair_ids for strength in strengths for direction in ("positive", "negative"))}
    records.sort(key=lambda result: order[result["run_id"]])
    if input_record(frozen) != input_identity:
        raise ValueError("Frozen input changed during the perturbation experiment")
    table_records = {record["run_id"]: record for record in records}
    for pair_id in PAIR_IDS:
        for result_path in sorted(paths.region_root(pair_id).glob("strength_*/*/result.json")):
            if result_path.parent.name in {"positive", "negative"}:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("experiment_id") == experiment_id and previous.get("baseline_id") == baseline_id:
                    table_records.setdefault(previous["run_id"], previous)
    rows = sorted(table_records.values(), key=lambda row: (
        PAIR_IDS.index(row["pair_id"]), row["strength"], row["direction"] != "positive"))
    write_tables(paths.pair_root, rows)
    if evaluate:
        plot_results(paths.pair_root, rows)
    _progress(f"[全部完成] {paths.time_pair}：共生成 {len(records)} 个 perturbation condition 记录")
    return records
