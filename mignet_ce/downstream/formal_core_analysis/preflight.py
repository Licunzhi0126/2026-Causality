"""Read-only prerequisite check for the formal publication-asset renderer.

This is not a model trainer and does not create scientific measurements.
"""
from __future__ import annotations

from pathlib import Path
from .tables import PAIRS, SCALES, METHODS, selected_candidate_sensitivity, collect_optimal_runs, build_optimal_grid
from .plots import _coordinates


def inspect_inputs(sensitivity_root: Path, data_root: Path, optimal_root: Path | None,
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
    missing_spatial = []
    if not skip_figures:
        for stage in figure_pair.split("->"):
            for scale in ("seurat_k150", "seurat_k40") + (("seurat_k10",) if k10_ready else ()):
                try:
                    _coordinates(data_root, scale, stage)
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
                build_optimal_grid(runs, "delta_EI_strict_raw", require_complete=True)
                build_optimal_grid(runs, "CQ_display", require_complete=True)
            except ValueError as exc:
                issues.append(str(exc))
        if rejected:
            cautions.append(f"{len(rejected)} run directories rejected by strict audit")
    if missing_optimal:
        cautions.append(f"Missing {len(missing_optimal)} of 18 strict optimal conditions")
    return {
        "pipeline": "formal_core_analysis", "stage": stage,
        "scientific_input_status": manifest.get("status"),
        "optimal_run_root": str(optimal_root) if stage == "all" and optimal_root else None,
        "strict_summary_json_found": summary_json_found,
        "alpha": alpha, "grn_method": grn_method, "has_k10": k10_ready,
        "missing_ei_conditions": [f"{a}/{b}" for a, b in missing_ei],
        "strict_optimal_runs": available_runs, "missing_optimal_conditions": missing_optimal,
        "rejected_runs": rejected, "missing_spatial_maps": missing_spatial,
        "ready_for_publication_candidate": not issues and not missing_ei and (stage == "natural" or not missing_optimal),
        "ready_for_incomplete_preview": not issues,
        "issues": issues, "cautions": cautions,
        "does_not_train_optimal_models": True,
    }
