"""Plan 6 orchestrator for recursive Optimal CG over macro transition pairs."""

from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np

from .cluster_correspondence import build_macro_correspondence, correspondence_rows
from .local_pair_builder import MIN_LOCAL_SPOTS, build_local_pair, validate_global_outputs
from .local_pair_reporting import write_csv, write_json, write_pair_inputs
from .local_pair_trainer import GLOBAL_SEED, choose_local_k, run_local_optimal_cg


SUMMARY_FIELDS = [
    "source_cluster", "target_cluster", "global_macro_transition_probability", "target_in_degree",
    "n_source_spots_original", "n_source_spots_used", "n_source_spots_dropped", "n_target_spots",
    "mean_pair_retained_mass", "median_pair_retained_mass", "min_pair_retained_mass", "max_pair_retained_mass",
    "k_global", "k_local", "local_micro_ei", "local_macro_ei", "local_delta_ei",
    "local_i_available", "local_i_retained", "local_cq", "status", "warning",
]


def run_local_pair_analysis(*, training_root: Path, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, k_global: int, device: str, global_seed: int = GLOBAL_SEED, min_local_spots: int = MIN_LOCAL_SPOTS) -> dict[str, object]:
    """Run all source-row argmax pairs; failures are contained per pair."""
    output_root = training_root.parent / "cluster_pair_analysis"
    pij_micro = np.load(training_root / "PIJ_micro_train.npy", allow_pickle=False)
    pij_macro = np.load(training_root / "PIJ_macro_train.npy", allow_pickle=False)
    s_t = np.load(training_root / "S_t.npy", allow_pickle=False)
    s_tp = np.load(training_root / "S_tp.npy", allow_pickle=False)
    validate_global_outputs(pij_micro, pij_macro, s_t, s_tp)
    normalized_macro, correspondences = build_macro_correspondence(pij_macro)
    write_csv(
        output_root / "macro_transition_matrix.csv",
        [dict(source_cluster=index, **{f"T2_C{column}": float(value) for column, value in enumerate(row)}) for index, row in enumerate(normalized_macro)],
        ["source_cluster", *[f"T2_C{item}" for item in range(normalized_macro.shape[1])]],
    )
    write_csv(output_root / "macro_cluster_correspondence.csv", correspondence_rows(correspondences), ["source_cluster", "target_cluster", "transition_probability", "source_rank", "target_in_degree"])
    rows: list[dict[str, object]] = []
    for item in correspondences:
        pair = build_local_pair(
            pij_micro=pij_micro, s_t=s_t, s_tp=s_tp, source_cluster=item.source_cluster,
            target_cluster=item.target_cluster, min_local_spots=min_local_spots,
        )
        pair_root = output_root / "pairs" / f"source_{item.source_cluster:03d}__target_{item.target_cluster:03d}"
        config = {"source_cluster": item.source_cluster, "target_cluster": item.target_cluster, "seed": global_seed + item.source_cluster * 1000 + item.target_cluster, "k_global": k_global, "k_local": choose_local_k(k_global, pair.n_source_used, pair.n_target), "transition_probability": item.transition_probability, "n_source_spots": pair.n_source_used, "n_target_spots": pair.n_target}
        write_pair_inputs(output_root, pair, config)
        row = _summary_row(item, pair, k_global)
        if pair.status == "ready":
            try:
                result = run_local_optimal_cg(pair=pair, inputs=inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, k_global=k_global, device=device, pair_root=pair_root, global_seed=global_seed)
                row.update(result)
                row["status"] = "completed"
            except Exception as error:  # One pair must not prevent the rest of the analysis.
                row["status"] = "failed"
                row["warning"] = _join_warning(row.get("warning"), f"{type(error).__name__}: {error}")
                write_json(pair_root / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        write_json(pair_root / "summary.json", row)
        rows.append(row)
    write_csv(output_root / "pair_summary.csv", rows, SUMMARY_FIELDS)
    payload = {"k_global": k_global, "global_seed": global_seed, "pair_count": len(rows), "completed_count": sum(row["status"] == "completed" for row in rows), "rows": rows}
    write_json(output_root / "pair_summary.json", payload)
    write_json(output_root / "diagnostics.json", {"pij_micro_shape": list(pij_micro.shape), "pij_macro_shape": list(pij_macro.shape), "S_t_shape": list(s_t.shape), "S_tp_shape": list(s_tp.shape), "min_local_spots": min_local_spots})
    return payload


def _summary_row(item, pair, k_global: int) -> dict[str, object]:
    retained = pair.retained_mass
    return {
        "source_cluster": item.source_cluster, "target_cluster": item.target_cluster,
        "global_macro_transition_probability": item.transition_probability, "target_in_degree": item.target_in_degree,
        "n_source_spots_original": pair.n_source_original, "n_source_spots_used": pair.n_source_used,
        "n_source_spots_dropped": pair.n_source_dropped, "n_target_spots": pair.n_target,
        "mean_pair_retained_mass": float(retained.mean()) if retained.size else None,
        "median_pair_retained_mass": float(np.median(retained)) if retained.size else None,
        "min_pair_retained_mass": float(retained.min()) if retained.size else None,
        "max_pair_retained_mass": float(retained.max()) if retained.size else None,
        "k_global": k_global, "k_local": None, "local_micro_ei": None, "local_macro_ei": None,
        "local_delta_ei": None, "local_i_available": None, "local_i_retained": None, "local_cq": None,
        "status": pair.status, "warning": pair.warning,
    }


def _join_warning(existing: object, extra: str) -> str:
    return f"{existing}; {extra}" if existing else extra
