from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from wyt_deltaei_coarse_grain.complete_combined import CompleteCombinedStage, prepare_complete_stage

from .analysis import (
    ClosureResult,
    align_assignments,
    boundary_fraction,
    closure_analysis,
    compose_matrices,
    connected_components,
    correlation_summary,
    effective_information,
    effective_state_number,
    js_rows,
    load_domain_assignment,
    matched_partition_null,
    multistep_closure,
    normalize_assignment,
    overlap_assignment,
    read_h5ad_units_coords,
    relative_frobenius,
    row_normalize,
    state_level_ei,
)
from .config import ClosureFullConfig, DynamicClosureConfig
from .io import layer_paths, layer_stem, read_index, validate_raw_inputs, write_manifest
from .optimal import OptimalRunConfig, ngklot_pij_numpy, prepare_ngklot_pair, run_optimal_pair
from .full_plots import render_all_closure_figures


def _stem(layer: str, organ: str, time: str) -> str:
    return layer_stem(layer, organ, time)


def _layer_paths(config: ClosureFullConfig, layer: str, time: str) -> dict[str, Path]:
    return layer_paths(config.data_root, config.organ, layer, time)


def _load_index(path: Path) -> list[str]:
    return read_index(path)


def _load_stage(config: ClosureFullConfig, layer: str, time: str) -> tuple[CompleteCombinedStage, np.ndarray]:
    paths = _layer_paths(config, layer, time)
    units = _load_index(paths["index"])
    cci = sp.load_npz(paths["cci"]).tocsr().astype(np.float32)
    h5_units, coords = read_h5ad_units_coords(paths["h5ad"])
    lookup = {unit: index for index, unit in enumerate(h5_units)}
    order = np.asarray([lookup[unit] for unit in units], dtype=int)
    stage = prepare_complete_stage(
        h5ad_path=paths["h5ad"],
        grn_path=paths["grn"],
        units=units,
        cci=cci,
        top_k_targets=50,
        state_dim=64,
        projection_seed=20260713,
    )
    return stage, coords[order]


def _natural_pij(
    config: ClosureFullConfig,
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    target_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    iterations = config.large_target_nmf_max_iter if target_count >= 2500 else config.nmf_max_iter
    pair = prepare_ngklot_pair(
        stage_t,
        stage_tp,
        nmf_components=config.nmf_components,
        nmf_max_iter=iterations,
        seed=config.seed,
    )
    return row_normalize(pair.micro_pij), {
        "network_method": "light_cci_grn",
        "pij_method": "NG_KLot",
        "nmf_max_iter_used": iterations,
        "micro_ei": float(pair.micro_ei),
        **pair.v7_metadata,
    }


def _identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)


def _save_closure_result(result: ClosureResult, table_dir: Path, slug: str) -> None:
    result.state_table.to_csv(table_dir / f"state_metrics_{slug}.csv", index=False)
    result.source_residuals.to_csv(table_dir / f"source_metrics_{slug}.csv", index=False)
    np.save(table_dir / f"q_best_{slug}.npy", result.q_best.astype(np.float32))
    if result.q_direct is not None:
        np.save(table_dir / f"q_direct_{slug}.npy", result.q_direct.astype(np.float32))


def _add_spatial_structure(
    result: ClosureResult,
    coords: np.ndarray,
    assignment: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = result.source_residuals.copy()
    table["x"] = coords[:, 0]
    table["y"] = coords[:, 1]
    hard = np.argmax(assignment, axis=1)
    table["boundary_fraction"] = boundary_fraction(coords, hard)
    table["hard_state"] = hard
    components = connected_components(coords, hard)
    table["state_components"] = [components.get(int(value), 0) for value in hard]
    correlations = []
    for x in ("boundary_fraction", "source_ei_micro", "source_ei_to_future_macro", "assignment_confidence"):
        metrics = correlation_summary(table, x, "closure_js")
        correlations.append({"mapping": result.summary["mapping"], "time_pair": result.summary["time_pair"], "x": x, **metrics})
    return table, pd.DataFrame(correlations)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def run_closure_full_analysis(config: ClosureFullConfig) -> dict[str, Any]:
    config = config.normalized()
    config.validate()
    validate_raw_inputs(config.data_root, config.organ, config.time_points)
    out = Path(config.output_root)
    table_dir = out / "tables"
    matrix_dir = out / "matrices"
    figure_dir = out / "figures"
    optimal_dir = out / "optimal_coarse"
    for directory in (table_dir, matrix_dir, figure_dir, optimal_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(out / "analysis_config.json", {**asdict(config), "data_root": str(config.data_root), "output_root": str(config.output_root)})

    adjacent = list(zip(config.time_points[:-1], config.time_points[1:]))
    units_cache: dict[tuple[str, str], list[str]] = {}
    coord_cache: dict[tuple[str, str], np.ndarray] = {}
    for layer in ("spot", "seurat_k150", "seurat_k40"):
        for time in config.time_points:
            paths = _layer_paths(config, layer, time)
            units_cache[(layer, time)] = _load_index(paths["index"])
            h5_units, coords = read_h5ad_units_coords(paths["h5ad"])
            lookup = {unit: index for index, unit in enumerate(h5_units)}
            coord_cache[(layer, time)] = coords[np.asarray([lookup[unit] for unit in units_cache[(layer, time)]], dtype=int)]

    pijs: dict[tuple[str, str, str], np.ndarray] = {}
    pij_meta: list[dict[str, Any]] = []
    for layer in ("spot", "seurat_k150", "seurat_k40"):
        for time_t, time_tp in adjacent:
            cache_path = matrix_dir / f"pij_{layer}_{time_t}_to_{time_tp}.npz"
            if cache_path.exists() and not config.force:
                p = row_normalize(np.load(cache_path)["pij"])
                metadata = {
                    "network_method": "light_cci_grn",
                    "pij_method": "NG_KLot",
                    "nmf_max_iter_used": "cached",
                    "micro_ei": effective_information(p),
                    "cache_reused": True,
                }
            else:
                stage_t, _ = _load_stage(config, layer, time_t)
                stage_tp, _ = _load_stage(config, layer, time_tp)
                p, metadata = _natural_pij(
                    config, stage_t, stage_tp, len(units_cache[(layer, time_tp)])
                )
                np.savez_compressed(cache_path, pij=p.astype(np.float32))
            pijs[(layer, time_t, time_tp)] = p
            pij_meta.append({"layer": layer, "time_pair": f"{time_t}->{time_tp}", **metadata})
    pd.DataFrame(pij_meta).to_csv(table_dir / "pij_method_audit.csv", index=False)

    spot_assignments: dict[tuple[str, str], np.ndarray] = {}
    spot_coords: dict[str, np.ndarray] = {}
    for time in config.time_points:
        spot_units = units_cache[("spot", time)]
        spot_coords[time] = coord_cache[("spot", time)]
        for layer in ("seurat_k150", "seurat_k40"):
            state_order = units_cache[(layer, time)]
            spot_assignments[(layer, time)] = load_domain_assignment(
                _layer_paths(config, layer, time)["map"],
                spot_units,
                state_order,
            )

    closure_results: dict[tuple[str, str], ClosureResult] = {}
    summaries: list[dict[str, Any]] = []
    source_tables: list[pd.DataFrame] = []
    state_tables: list[pd.DataFrame] = []
    correlation_tables: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []

    for time_t, time_tp in adjacent:
        pair = f"{time_t}->{time_tp}"
        p_spot = pijs[("spot", time_t, time_tp)]
        # Spot is the uncompressed reference scale.
        summaries.append({
            "mapping": "Spot (uncompressed)", "time_pair": pair,
            "source_nodes": p_spot.shape[0], "target_nodes": p_spot.shape[1],
            "source_macro_states": p_spot.shape[0], "target_macro_states": p_spot.shape[1],
            "EI_micro_dynamics": effective_information(p_spot),
            "I_micro_to_future_macro": effective_information(p_spot),
            "I_macro_to_future_macro": effective_information(p_spot),
            "micro_residual_information": 0.0, "closure_leakage_bits": 0.0, "macro_sufficiency": 1.0,
            "residual_fraction": 0.0, "signal_status": "informative", "signal_threshold_bits": 0.0,
            "information_identity_error": 0.0, "primary_macro_representation": "hard", "primary_weighting": "state_balanced",
            "I_macro_to_future_micro": effective_information(p_spot),
            "downward_reach_ratio": 1.0, "best_closure_mean_js": 0.0,
            "best_closure_median_js": 0.0, "best_closure_p95_js": 0.0,
            "best_closure_max_js": 0.0, "best_closure_mean_kl": 0.0,
            "best_closure_relative_frobenius": 0.0,
            "induced_closure_mean_js": 0.0, "induced_closure_mean_kl_bits": 0.0, "induced_closure_relative_frobenius": 0.0,
            "crossfit_closure_kl_bits": 0.0, "crossfit_closure_js": 0.0, "crossfit_closure_coverage": 1.0,
            "crossfit_singleton_weight_fraction": 0.0,
            "EI_induced_Q_uniform": effective_information(p_spot),
            "EI_induced_Q_weighted": effective_information(p_spot),
            "Keff_source_macro": float(p_spot.shape[0]),
            "direct_closure_mean_js": 0.0, "direct_closure_relative_frobenius": 0.0,
            "EI_direct_Q": effective_information(p_spot),
            "direct_excess_js_above_best": 0.0,
            "direct_excess_frobenius_above_best": 0.0,
            "independent_q_excess_js": 0.0, "independent_q_gap_js": 0.0,
        })

        for layer, label in (("seurat_k150", "Seurat K150"), ("seurat_k40", "Seurat K40")):
            result = closure_analysis(
                p_spot,
                spot_assignments[(layer, time_t)],
                spot_assignments[(layer, time_tp)],
                label=label,
                time_pair=pair,
                q_direct=pijs[(layer, time_t, time_tp)],
                source_state_names=units_cache[(layer, time_t)],
                weighting="state_balanced",
                crossfit_folds=config.crossfit_folds,
                low_signal_threshold_bits=config.low_signal_threshold_bits,
            )
            closure_results[(label, pair)] = result
            summaries.append(result.summary)
            spatial, correlations = _add_spatial_structure(result, spot_coords[time_t], spot_assignments[(layer, time_t)])
            source_tables.append(spatial)
            state_tables.append(result.state_table)
            correlation_tables.append(correlations)
            slug = f"{layer}_{time_t}_to_{time_tp}"
            _save_closure_result(result, table_dir, slug)
            null = matched_partition_null(result.observed, spot_assignments[(layer, time_t)], repeats=config.null_repeats, seed=config.seed + int(float(time_t) * 10) + (150 if layer.endswith("150") else 40))
            for key, value in result.summary.items():
                if key in {"macro_sufficiency", "residual_fraction", "best_closure_mean_js", "best_closure_relative_frobenius"}:
                    null[f"observed_{key}"] = value
            null["mapping"] = label
            null["time_pair"] = pair
            null_tables.append(null)

        overlap_t = overlap_assignment(spot_assignments[("seurat_k150", time_t)], spot_assignments[("seurat_k40", time_t)])
        overlap_tp = overlap_assignment(spot_assignments[("seurat_k150", time_tp)], spot_assignments[("seurat_k40", time_tp)])
        result = closure_analysis(
            pijs[("seurat_k150", time_t, time_tp)],
            overlap_t,
            overlap_tp,
            label="K150 to K40 overlap",
            time_pair=pair,
            q_direct=pijs[("seurat_k40", time_t, time_tp)],
            source_state_names=units_cache[("seurat_k40", time_t)],
            weighting="state_balanced",
            crossfit_folds=config.crossfit_folds,
            low_signal_threshold_bits=config.low_signal_threshold_bits,
        )
        closure_results[("K150 to K40 overlap", pair)] = result
        summaries.append(result.summary)
        state_tables.append(result.state_table)
        _save_closure_result(result, table_dir, f"k150_to_k40_{time_t}_to_{time_tp}")

    # Optimal complete coarse-graining + current best light_cci_grn / NG_KLot.
    optimal_config = OptimalRunConfig(
        data_root=Path(config.data_root), output_root=optimal_dir, organ=config.organ,
        k=config.k_optimal, epochs=config.optimal_epochs, nmf_components=config.nmf_components,
        nmf_max_iter=config.nmf_max_iter, large_target_nmf_max_iter=config.large_target_nmf_max_iter,
        seed=config.seed, device=config.device, force=config.force,
    )
    optimal_runs: dict[str, dict[str, Any]] = {}
    optimal_spatial_rows: list[pd.DataFrame] = []
    for time_t, time_tp in adjacent:
        pair = f"{time_t}->{time_tp}"
        run = run_optimal_pair(optimal_config, time_t, time_tp)
        optimal_runs[pair] = run
        result = closure_analysis(
            run["micro_pij"],
            run["S_t"], run["S_tp"],
            label="Optimized coarse-graining",
            time_pair=pair,
            q_direct=run["Q_train"],
            source_state_names=[f"M{index:02d}" for index in range(run["S_t"].shape[1])],
            weighting="state_balanced",
            crossfit_folds=config.crossfit_folds,
            low_signal_threshold_bits=config.low_signal_threshold_bits,
        )
        closure_results[("Optimized coarse-graining", pair)] = result
        summary = dict(result.summary)
        source_usage = run["S_t"].sum(axis=0) / run["S_t"].sum()
        target_usage = run["S_tp"].sum(axis=0) / run["S_tp"].sum()
        summary.update({
            "hardK_source": int(np.unique(np.argmax(run["S_t"], axis=1)).size),
            "hardK_target": int(np.unique(np.argmax(run["S_tp"], axis=1)).size),
            "Keff_source_assignment": effective_state_number(source_usage),
            "Keff_target_assignment": effective_state_number(target_usage),
            "EI_training_macro_Q": effective_information(run["Q_train"]),
            "EI_best_fit_macro_Q": effective_information(result.q_best),
        })
        summaries.append(summary)
        spatial, correlations = _add_spatial_structure(result, spot_coords[time_t], run["S_t"])
        source_tables.append(spatial)
        state_tables.append(result.state_table)
        correlation_tables.append(correlations)
        _save_closure_result(result, table_dir, f"optimized_{time_t}_to_{time_tp}")

        macro_ei = state_level_ei(run["Q_train"])
        top_score = run["S_t"] @ macro_ei
        spot_ei = state_level_ei(run["micro_pij"])
        frame = pd.DataFrame({
            "time_pair": pair,
            "source_time": time_t,
            "spot_id": run["unit_ids_t"],
            "x": spot_coords[time_t][:, 0], "y": spot_coords[time_t][:, 1],
            "optimized_macro_ei_contribution": top_score,
            "spot_ei_contribution": spot_ei,
            "optimized_hard_state": np.argmax(run["S_t"], axis=1),
            "assignment_confidence": np.max(run["S_t"], axis=1),
            "closure_js_train_q": js_rows(result.observed, result.predicted_direct),
            "closure_js_best_q": js_rows(result.observed, result.predicted_best),
        })
        optimal_spatial_rows.append(frame)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(table_dir / "closure_summary_all_scales.csv", index=False)
    pd.concat(source_tables, ignore_index=True).to_csv(table_dir / "closure_source_metrics_all.csv", index=False)
    pd.concat(state_tables, ignore_index=True).to_csv(table_dir / "closure_state_metrics_all.csv", index=False)
    pd.concat(correlation_tables, ignore_index=True).to_csv(table_dir / "closure_correlations_all.csv", index=False)
    pd.concat(null_tables, ignore_index=True).to_csv(table_dir / "closure_matched_null_all.csv", index=False)
    optimal_spatial = pd.concat(optimal_spatial_rows, ignore_index=True)
    optimal_spatial.to_csv(table_dir / "ei_contribution_spatial_spot_vs_optimized.csv", index=False)

    # Multi-step natural-scale closure.
    multi_rows: list[dict[str, Any]] = []
    for layer, label in (("seurat_k150", "Seurat K150"), ("seurat_k40", "Seurat K40")):
        for start, end_index in ((0, 2), (0, 3), (1, 3)):
            times = config.time_points[start:end_index + 1]
            p_chain = [pijs[("spot", times[i], times[i + 1])] for i in range(len(times) - 1)]
            q_chain = [pijs[(layer, times[i], times[i + 1])] for i in range(len(times) - 1)]
            assignments = [spot_assignments[(layer, time)] for time in times]
            multi_rows.append(multistep_closure(p_chain, assignments, q_chain, mapping=label, interval=f"{times[0]}->{times[-1]}"))

    alignment_rows: list[dict[str, Any]] = []
    # Pair-specific learned assignments at shared times are aligned before Q composition.
    pair_labels = [f"{a}->{b}" for a, b in adjacent]
    q_aligned = [optimal_runs[pair_labels[0]]["Q_train"]]
    for index in range(len(pair_labels) - 1):
        left = optimal_runs[pair_labels[index]]["S_tp"]
        right = optimal_runs[pair_labels[index + 1]]["S_t"]
        _, permutation, stats = align_assignments(left, right)
        q_aligned.append(optimal_runs[pair_labels[index + 1]]["Q_train"][permutation, :])
        alignment_rows.append({"shared_time": adjacent[index][1], **stats})
    for start, end_index in ((0, 2), (0, 3), (1, 3)):
        times = config.time_points[start:end_index + 1]
        p_chain = [pijs[("spot", times[i], times[i + 1])] for i in range(len(times) - 1)]
        q_chain = q_aligned[start:end_index]
        source_assignment = optimal_runs[pair_labels[start]]["S_t"]
        target_assignment = optimal_runs[pair_labels[end_index - 1]]["S_tp"]
        # Pairwise optimized states are not jointly trained across all times. Keep
        # this as an operational composition diagnostic, but evaluate the endpoint
        # partition with the same hard/state-balanced closure semantics.
        from .analysis import compact_hard_assignment, state_balanced_micro_weights, induced_macro_q, weighted_information, kl_rows
        h_source, active_source = compact_hard_assignment(source_assignment)
        h_target, active_target = compact_hard_assignment(target_assignment)
        observed = row_normalize(compose_matrices(p_chain) @ h_target)
        # The aligned q_chain is still nominal-K. Compact only endpoint rows/cols;
        # intermediate pairwise alignment remains a diagnostic limitation.
        q_composed_full = compose_matrices(q_chain)
        q_composed = row_normalize(q_composed_full[np.ix_(active_source, active_target)])
        predicted = row_normalize(h_source @ q_composed)
        weights = state_balanced_micro_weights(h_source)
        q_induced, macro_mass = induced_macro_q(observed, h_source, weights)
        predicted_induced = row_normalize(h_source @ q_induced)
        composed_js=float(np.sum(weights*js_rows(observed,predicted)))
        induced_js=float(np.sum(weights*js_rows(observed,predicted_induced)))
        multi_rows.append({
            "mapping": "Optimized coarse-graining", "interval": f"{times[0]}->{times[-1]}", "steps": len(p_chain),
            "composed_mean_js": composed_js,
            "composed_relative_frobenius": relative_frobenius(predicted*weights[:,None]**0.5, observed*weights[:,None]**0.5),
            "best_possible_mean_js": induced_js,
            "induced_endpoint_mean_js": induced_js,
            "best_possible_relative_frobenius": relative_frobenius(predicted_induced*weights[:,None]**0.5, observed*weights[:,None]**0.5),
            "semigroup_excess_js": composed_js-induced_js,
            "semigroup_excess_frobenius": relative_frobenius(predicted*weights[:,None]**0.5, observed*weights[:,None]**0.5)-relative_frobenius(predicted_induced*weights[:,None]**0.5, observed*weights[:,None]**0.5),
            "EI_composed_Q": weighted_information(q_composed, macro_mass),
            "EI_best_Q": weighted_information(q_induced, macro_mass),
            "I_endpoint_available_bits": weighted_information(observed,weights),
            "I_endpoint_macro_retained_bits": weighted_information(q_induced,macro_mass),
            "endpoint_closure_leakage_bits": float(np.sum(weights*kl_rows(observed,predicted_induced))),
        })
    multi_frame = pd.DataFrame(multi_rows)
    multi_frame.to_csv(table_dir / "closure_multistep_summary.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(table_dir / "optimized_shared_time_alignment.csv", index=False)

    figure_paths = render_all_closure_figures(
        summary_frame=summary_frame,
        source_frame=pd.concat(source_tables, ignore_index=True),
        null_frame=pd.concat(null_tables, ignore_index=True),
        multistep_frame=multi_frame,
        alignment_frame=pd.DataFrame(alignment_rows),
        optimal_spatial_frame=optimal_spatial,
        closure_results=closure_results,
        output_dir=figure_dir,
    )

    findings = {
        "method": {"network_method": "light_cci_grn", "pij_method": "NG_KLot", "coarse_method": "complete_combined_coarse"},
        "figures": [str(path) for path in figure_paths],
        "summary_rows": int(len(summary_frame)),
        "multistep_rows": int(len(multi_frame)),
    }
    _write_json(out / "findings.json", findings)
    manifest = write_manifest(out, asdict(config))
    return {
        "output_root": out,
        "figures": figure_paths,
        "summary": summary_frame,
        "multistep": multi_frame,
        "findings": findings,
        "manifest": manifest,
    }


def run_dynamic_closure_analysis(
    config: DynamicClosureConfig,
    *,
    stage: str = "all",
) -> dict[str, Any]:
    """Run one or all extended dynamic-closure stages under one output root."""

    cfg = config.normalized()
    cfg.validate()
    validate_raw_inputs(cfg.data_root, cfg.organ, cfg.time_points)
    stage = str(stage).lower()
    allowed = {"full", "deep", "ultradeep", "all"}
    if stage not in allowed:
        raise ValueError(f"stage must be one of {sorted(allowed)}, got {stage!r}")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if stage in {"full", "all"}:
        results["full"] = run_closure_full_analysis(
            replace(cfg, output_root=cfg.full_root)
        )
    elif not cfg.full_root.is_dir():
        raise FileNotFoundError(
            f"The {stage} stage requires full closure results at {cfg.full_root}"
        )

    if stage in {"deep", "all"}:
        from .deep import run_deep_closure_analysis

        results["deep"] = run_deep_closure_analysis(
            data_root=cfg.data_root,
            closure_root=cfg.full_root,
            output_root=cfg.deep_root,
            organ=cfg.organ,
            time_points=cfg.time_points,
            bootstrap_repeats=cfg.bootstrap_repeats,
            seed=cfg.seed,
        )
    elif stage == "ultradeep" and not cfg.deep_root.is_dir():
        raise FileNotFoundError(
            f"The ultradeep stage requires deep closure results at {cfg.deep_root}"
        )

    if stage in {"ultradeep", "all"}:
        from .ultradeep import run_ultradeep_closure_analysis

        results["ultradeep"] = run_ultradeep_closure_analysis(
            data_root=cfg.data_root,
            closure_root=cfg.full_root,
            deep_root=cfg.deep_root,
            output_root=cfg.ultradeep_root,
            organ=cfg.organ,
            time_points=cfg.time_points,
            max_splits=cfg.repair_max_splits,
            random_repeats=cfg.repair_random_repeats,
            seed=cfg.seed,
        )

    if cfg.output_mode in {"paper", "both"} and stage in {"deep", "ultradeep", "all"}:
        from .paper import run_paper_closure_analysis
        results["paper"] = run_paper_closure_analysis(
            closure_root=cfg.full_root,
            deep_root=cfg.deep_root,
            output_root=cfg.paper_root,
        )

    available_stages = [
        name
        for name, path in (
            ("full", cfg.full_root),
            ("deep", cfg.deep_root),
            ("ultradeep", cfg.ultradeep_root),
            ("paper", cfg.paper_root),
        )
        if path.is_dir()
    ]
    summary = {
        "requested_stage": stage,
        "updated_stages": list(results),
        "available_stages": available_stages,
        "figure_count": len(list(cfg.output_root.rglob("*.png"))),
        "output_root": str(cfg.output_root),
    }
    _write_json(cfg.output_root / "findings.json", summary)
    manifest = write_manifest(cfg.output_root, asdict(cfg))
    return {
        "output_root": cfg.output_root,
        "stages": results,
        "findings": summary,
        "manifest": manifest,
    }


__all__ = [
    "ClosureFullConfig",
    "DynamicClosureConfig",
    "run_closure_full_analysis",
    "run_dynamic_closure_analysis",
]
