"""Read-only prerequisite check for the formal publication-asset renderer.

This is not a model trainer and does not create scientific measurements.
"""
from __future__ import annotations

from pathlib import Path
from .tables import PAIRS, SCALES, METHODS, selected_candidate_sensitivity, collect_optimal_runs, build_optimal_grid
from .plots import _coordinates
from .closure_js import natural_js_grid
from .methods import expected_conditions
from .natural_analysis import NaturalAnalysisConfig, inspect_organ_inputs, requested_organs
from mignet_ce.coarse_frontends.method_specs import FORMAL_LOCAL_GRAPH_MODE, graph_configuration


def _inspect_natural_inputs(sensitivity_root: Path, data_root: Path, optimal_root: Path | None,
                   alpha: float, grn_method: str, figure_pair: str, skip_figures: bool,
                   stage: str = "all") -> dict:
    if stage not in {"all", "natural"}:
        raise ValueError(f"Unknown formal core stage: {stage}")
    issues: list[str] = []
    cautions: list[str] = []
    ei, cq, manifest = selected_candidate_sensitivity(sensitivity_root, alpha, grn_method)
    coverage = {(str(row.time_pair), str(row.layer)) for row in ei.itertuples(index=False)}
    expected_ei = {(pair, layer) for pair in PAIRS for layer in (*SCALES, "seurat_k10")}
    missing_ei = sorted(expected_ei - coverage)
    k10_ready = not any(layer == "seurat_k10" for _, layer in missing_ei)
    if missing_ei:
        cautions.append(f"Missing EI measurements for {len(missing_ei)} pair/layer conditions")
    # Existing measurements must contain a real JS for every required hierarchy;
    # K10 absence is permitted only in a visibly incomplete preview.
    missing_natural_js: list[str] = []
    for pair in PAIRS:
        for hierarchy in ("spot->seurat_k150", "spot->seurat_k40", "seurat_k150->seurat_k40", "spot->seurat_k10"):
            hit = cq.loc[cq.time_pair.eq(pair) & cq.hierarchy.eq(hierarchy)]
            if len(hit) != 1 or "closure_JS_direct_bits" not in cq.columns:
                missing_natural_js.append(f"{pair}/{hierarchy}")
    if missing_natural_js:
        cautions.append(f"Missing {len(missing_natural_js)} required natural direct JS cells")
    try:
        natural_js_grid(cq, pairs=PAIRS, allow_missing_k10=True)
    except (ValueError, TypeError) as exc:
        issues.append(f"Natural direct JS invalid: {exc}")
    missing_spatial = []
    if not skip_figures:
        for time_point in figure_pair.split("->"):
            for scale in ("seurat_k150", "seurat_k40") + (("seurat_k10",) if k10_ready else ()):
                try:
                    _coordinates(data_root, scale, time_point)
                except (FileNotFoundError, ValueError) as exc:
                    missing_spatial.append(str(exc))
    if missing_spatial:
        issues.append("Spatial coordinates missing/invalid for Figure 1")
    available_runs = 0
    missing_optimal = []
    rejected = []
    summary_json_found = 0
    if stage == "natural":
        cautions.append("Natural-only mode: optimal runs are not requested or assessed")
    elif optimal_root is None:
        cautions.append("No optimal runs supplied; cannot construct optimal tables")
        missing_optimal = [f"{method}/{scale}/{pair}" for method in METHODS for scale in SCALES for pair in PAIRS]
    elif not Path(optimal_root).exists():
        issues.append(f"Optimal run root does not exist: {optimal_root}")
        missing_optimal = [f"{method}/{scale}/{pair}" for method in METHODS for scale in SCALES for pair in PAIRS]
    else:
        root = Path(optimal_root)
        summary_json_found = 1 if (root / "summary.json").is_file() else sum(1 for _ in root.rglob("summary.json"))
        runs, rejected = collect_optimal_runs(optimal_root)
        available_runs = len(runs)
        valid_set = {(str(r.method), str(r.input_scale), str(r.time_pair)) for r in runs.itertuples(index=False)} if len(runs) else set()
        missing_optimal = [f"{method}/{scale}/{pair}" for method in METHODS for scale in SCALES for pair in PAIRS if (method, scale, pair) not in valid_set]
        if not missing_optimal:
            try:
                build_optimal_grid(runs, "trained_delta_EI", require_complete=True)
                build_optimal_grid(runs, "CQ_raw", require_complete=True)
                build_optimal_grid(runs, "closure_JS_direct_bits", require_complete=True)
            except ValueError as exc:
                issues.append(str(exc))
        if rejected:
            cautions.append(f"{len(rejected)} run directories rejected by trained PIJ audit")
    if missing_optimal:
        cautions.append(f"Missing {len(missing_optimal)} of {expected_conditions(SCALES, PAIRS)} trained optimal conditions")
    return {
        "pipeline": "formal_core_analysis", "stage": stage,
        "scientific_input_status": manifest.get("status"),
        "optimal_run_root": str(optimal_root) if stage == "all" and optimal_root else None,
        "trained_summary_json_found": summary_json_found,
        "alpha": alpha, "grn_method": grn_method, "has_k10": k10_ready,
        "missing_ei_conditions": [f"{a}/{b}" for a, b in missing_ei],
        "missing_natural_direct_js": missing_natural_js,
        "expected_optimal_runs": expected_conditions(SCALES, PAIRS),
        "trained_optimal_runs": available_runs, "missing_optimal_conditions": missing_optimal,
        "rejected_runs": rejected, "missing_spatial_maps": missing_spatial,
        "ready_for_publication_candidate": not issues and not missing_ei and not missing_natural_js and (stage == "natural" or not missing_optimal),
        "ready_for_incomplete_preview": not issues,
        "issues": issues, "cautions": cautions,
        "does_not_train_optimal_models": True,
    }


def _inspect_fixed_natural_inputs(data_root, natural_root, grn_root, cci_root,
                                  alpha, figure_pair, skip_figures, organs):
    cfg = NaturalAnalysisConfig(data_root=Path(data_root),
                                output_root=Path(natural_root) if natural_root else Path(data_root) / "unused_natural_preflight",
                                grn_root=grn_root, cci_root=cci_root, alpha=alpha)
    natural_organs = requested_organs(organs, include_heart_figure=not skip_figures)
    source = inspect_organ_inputs(cfg, natural_organs, figure_pair=None if skip_figures else figure_pair)
    issues = list(source["issues"]) + list(source["missing_inputs"])
    measured = source["measurements_ready"]
    return {"pipeline": "formal_core_analysis", "issues": issues, "cautions": [],
            "natural_source_type": "fixed_alpha_from_original_inputs", "sensitivity_required": False,
            "natural_inputs": source, "has_k10": False, "k10_required": False,
            "natural_measurements_ready": measured,
            "natural_computation_pending": not measured,
            "missing_ei_conditions": [] if measured else [f"{organ}/{pair}/{layer}" for organ in natural_organs
                                                          for pair in PAIRS for layer in SCALES],
            "missing_natural_direct_js": [] if measured else [f"{organ}/{pair}/{hierarchy}" for organ in natural_organs
                                                               for pair in PAIRS for hierarchy in (
                "spot->seurat_k150", "spot->seurat_k40", "seurat_k150->seurat_k40")],
            "missing_spatial_maps": [], "ready_for_execution": not issues,
            "ready_for_publication_candidate": measured and not issues,
            "ready_for_incomplete_preview": not issues,
            "does_not_compute_natural_measurements": True,
            "does_not_train_optimal_models": True}


def inspect_inputs(sensitivity_root, data_root, optimal_root, alpha, grn_method,
                   figure_pair, skip_figures, stage="all", *, organs=None,
                   figure_organ="heart", figure_input_scale="spot",
                   local_graph_mode=FORMAL_LOCAL_GRAPH_MODE, knn_k=30, local_dims=2,
                   natural_root=None, grn_root=None, cci_root=None):
    """Audit natural assets independently from the selected formal organ matrix."""
    from .methods import ORGANS
    from .tables import OPTIMAL_TABLES, build_optimal_report_grid
    graph = graph_configuration("formal_two_stage", local_graph_mode=local_graph_mode,
                                 knn_k=knn_k, local_dims=local_dims)
    organs = tuple(ORGANS if organs is None else organs)
    if not organs or len(set(organs)) != len(organs) or not set(organs) <= set(ORGANS):
        raise ValueError("Select unique supported organs")
    if stage not in {"all", "natural", "optimal"}:
        raise ValueError(f"Unknown formal core stage: {stage}")
    if stage != "optimal":
        if sensitivity_root is not None:
            report = _inspect_natural_inputs(sensitivity_root, data_root, None, alpha,
                                            grn_method, figure_pair, skip_figures, stage="natural")
            report["ready_for_execution"] = report["ready_for_publication_candidate"]
        else:
            report = _inspect_fixed_natural_inputs(data_root, natural_root, grn_root, cci_root,
                                                   alpha, figure_pair, skip_figures, organs)
    else:
        report = {"pipeline":"formal_core_analysis", "issues":[], "cautions":[],
                  "missing_ei_conditions":[], "missing_natural_direct_js":[],
                  "missing_spatial_maps":[], "ready_for_publication_candidate":True,
                  "ready_for_incomplete_preview":True, "ready_for_execution":True}
    report.update(stage=stage, organs=list(organs), methods=list(METHODS),
                  expected_optimal_runs=0 if stage == "natural" else expected_conditions(SCALES, PAIRS, organs),
                  does_not_train_optimal_models=True)
    report.update(graph)
    if stage == "natural":
        return report
    runs, rejected = collect_optimal_runs(optimal_root, **graph) if optimal_root else (None, ["No optimal run root supplied"])
    selected = runs.loc[runs.organ.isin(organs)] if runs is not None and not runs.empty else None
    valid = set() if selected is None else set(zip(selected.organ, selected.input_scale, selected.time_pair))
    missing = [f"{organ}/{scale}/{pair}" for organ in organs for scale in SCALES for pair in PAIRS
               if (organ, scale, pair) not in valid]
    if selected is not None and not missing:
        try:
            for metrics in OPTIMAL_TABLES.values():
                build_optimal_report_grid(selected, metrics, organs=organs, **graph)
        except ValueError as exc:
            report["issues"].append(str(exc))
    if not skip_figures:
        if (figure_organ, figure_input_scale, figure_pair) not in valid:
            report["issues"].append("Selected Figure 2 run is unavailable")
        else:
            for time in figure_pair.split("->"):
                for scale in {"seurat_k40", figure_input_scale} - {"spot"}:
                    try:
                        _coordinates(data_root, scale, time, figure_organ)
                    except (FileNotFoundError, ValueError) as exc:
                        report["missing_spatial_maps"].append(str(exc))
    report.update(optimal_run_root=str(optimal_root) if optimal_root else None,
                  trained_optimal_runs=0 if selected is None else len(selected),
                  missing_optimal_conditions=missing, rejected_runs=rejected)
    report["ready_for_publication_candidate"] = (report["ready_for_publication_candidate"] and not missing
                                                and not report["issues"] and not report["missing_spatial_maps"])
    report["ready_for_incomplete_preview"] = not report["issues"] and not report["missing_spatial_maps"]
    report["ready_for_execution"] = (report["ready_for_execution"] and not missing
                                      and not report["issues"] and not report["missing_spatial_maps"])
    return report
