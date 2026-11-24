"""Read global arrays and report conditional group Micro EI without training."""
from pathlib import Path

import numpy as np

from mignet_ce.metrics import effective_information
from .spot_grouping import build_source_groups, hard_assignments, assign_best_target_spots
from .group_micro_builder import build_group_micro, validate_options
from .group_micro_reporting import write_csv, write_json, write_group_inputs


SUMMARY_FIELDS = [
    "source_cluster", "target_cluster", "n_source_group_original", "n_source_group_used",
    "n_source_group_dropped", "n_target_cluster_spots", "group_micro_ei_bits",
    "mean_target_cluster_mass", "min_target_cluster_mass", "max_target_cluster_mass",
    "status", "warning",
]
MAPPING_FIELDS = ["source_spot_index", "source_cluster", "best_target_spot_index",
                  "best_target_spot_probability", "best_target_cluster"]


def run_group_micro_analysis(*, training_root: Path, output_root: Path | None = None,
                             eps: float = 1e-12,
                             export_matrices: bool = True) -> dict[str, object]:
    validate_options(eps)
    training_root = Path(training_root)
    output_root = Path(output_root) if output_root is not None else training_root.parent / "group_micro_analysis"
    # Never overwrite inputs, even if an explicit output directory is supplied.
    if output_root.resolve() == training_root.resolve() or output_root.resolve() in training_root.resolve().parents:
        raise ValueError("output_root must be separate from training_root and its ancestors.")
    p, s_t, s_tp = [np.load(training_root / name, allow_pickle=False) for name in
                       ("PIJ_micro_train.npy", "S_t.npy", "S_tp.npy")]
    groups, diagnostics = build_source_groups(p, s_t, s_tp)
    source_labels, target_labels = hard_assignments(s_t, s_tp)
    best, clusters = assign_best_target_spots(p, target_labels)
    mapping = ({"source_spot_index": i, "source_cluster": int(source_labels[i]),
                "best_target_spot_index": int(best[i]), "best_target_spot_probability": float(p[i, best[i]]),
                "best_target_cluster": int(clusters[i])} for i in range(len(p)))
    write_csv(output_root / "spot_target_mapping.csv", mapping, MAPPING_FIELDS)
    rows = []
    for group in groups:
        directory = output_root / "groups" / f"source_{group.source_cluster:03d}__target_{group.target_cluster:03d}"
        row = dict.fromkeys(SUMMARY_FIELDS)
        row.update(source_cluster=group.source_cluster, target_cluster=group.target_cluster,
                   n_source_group_original=int(group.source_indices.size), status="failed", warning="")
        try:
            data = build_group_micro(pij_micro=p, target_labels=target_labels, group=group,
                                     eps=eps)
            row.update(n_source_group_used=int(data.source_indices_used.size),
                       n_source_group_dropped=int(data.dropped_mask.sum()), n_target_cluster_spots=int(data.target_indices.size),
                       mean_target_cluster_mass=float(data.retained_mass.mean()),
                       min_target_cluster_mass=float(data.retained_mass.min()), max_target_cluster_mass=float(data.retained_mass.max()),
                       status=data.status, warning=data.warning)
            write_group_inputs(directory, data, export_matrices)
            write_json(directory / "dropped_sources.json", [
                {"source_spot_index": int(i), "reason": "target cluster mass <= eps"}
                for i in data.source_indices_original[data.dropped_mask]])
            if data.status == "ready":
                ei = float(effective_information(data.pij_conditional.copy()))
                if not np.isfinite(ei):
                    raise ValueError("Group EI is not finite.")
                row.update(group_micro_ei_bits=ei, status="completed")
        except Exception as error:
            row.update(status="failed", group_micro_ei_bits=None,
                       warning=f"{type(error).__name__}: {error}")
        write_json(directory / "summary.json", row)
        rows.append(row)
    failed = sum(r["status"] == "failed" for r in rows)
    completed = sum(r["status"] == "completed" for r in rows)
    skipped = sum(r["status"] == "skipped" for r in rows)
    status = ("partial_failure" if completed or skipped else "failed") if failed else "completed"
    payload = {"status": status, "group_count": len(rows),
               "completed_count": completed, "skipped_count": skipped,
               "failed_count": failed, "rows": rows}
    diagnostics.update(PIJ_micro_shape=list(p.shape), S_t_shape=list(s_t.shape), S_tp_shape=list(s_tp.shape),
                       eps=eps, export_matrices=export_matrices,
                       index_convention="zero-based row/column positions in global PIJ_micro_train",
                       interpretation="Conditional subsystem EI in bits under uniform intervention on used source spots")
    write_csv(output_root / "group_micro_ei.csv", rows, SUMMARY_FIELDS)
    write_json(output_root / "group_micro_ei.json", payload)
    write_json(output_root / "diagnostics.json", diagnostics)
    return payload
