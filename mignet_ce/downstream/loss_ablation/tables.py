"""Select the recorded stage checkpoints and export machine-readable tables."""

from __future__ import annotations

import csv
import json
import math
from itertools import product
from pathlib import Path

from .variants import METHOD_LABELS, STAGES, TIME_PAIRS


INFORMATION_COLUMNS = {
    "delta_ei_bits": "ΔEI",
    "cq": "CQ",
    "i_retain_bits": "Iretain",
    "i_available_bits": "Iavailable",
}
OCCUPANCY_COLUMNS = {
    "hard_k": "HardK",
    "keff_hard": "KeffHard",
    "hard_k_minus_keff_hard": "HardK − KeffHard",
}
ENDPOINTS = ("source", "target")
IDENTITY_COLUMNS = (
    "input_scale", "method", "stage", "source_time", "target_time", "k",
    "checkpoint", "checkpoint_epoch", "checkpoint_selection", "fallback_used",
)


def checkpoint_rows(
    run_dir: Path, *, method: str, scale: str, pair: tuple[str, str], k: int,
) -> list[dict[str, object]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary["checkpoint_metric_state_consistent"] or summary["selected_checkpoint"] != "best_joint.pt":
        raise ValueError(f"Stage 2 checkpoint was not verified: {run_dir}")
    stage1_epoch = int(summary["stage1_best_epoch"])
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        stage1, = (
            row for row in csv.DictReader(stream)
            if row["stage"] == "stage1" and int(row["epoch"]) == stage1_epoch
        )

    records = []
    for stage, metrics in zip(STAGES, (stage1, summary)):
        first = stage == "stage1"
        checkpoint = "best_ei.pt" if first else "best_joint.pt"
        if not (run_dir / checkpoint).is_file():
            raise FileNotFoundError(run_dir / checkpoint)
        source, target = pair
        record = {
            "input_scale": scale, "method": method, "stage": stage,
            "source_time": source, "target_time": target, "k": k,
            "checkpoint": checkpoint,
            "checkpoint_epoch": stage1_epoch if first else int(summary["best_epoch"]),
            "checkpoint_selection": (
                "stage1_best_delta_ei_noncollapsed" if first else summary["best_joint_selection"]
            ),
            "fallback_used": not first and bool(summary["best_joint_fallback_used"]),
            "delta_ei_bits": float(metrics["delta_EI" if first else "delta_EI_best_checkpoint"]),
            "cq": float(metrics["closure_quality"]),
            "i_retain_bits": float(metrics["I_retained"]),
            "i_available_bits": float(metrics["I_available"]),
        }
        if not all(math.isfinite(record[key]) for key in INFORMATION_COLUMNS):
            raise ValueError(f"Non-finite information metric in {checkpoint}: {run_dir}")
        for endpoint, suffix in zip(ENDPOINTS, ("t", "tp")):
            hard_k = int(metrics[f"hardK_{suffix}"])
            keff_hard = float(metrics[f"KeffHard_{suffix}"])
            if not 1.0 <= keff_hard <= hard_k <= k:
                raise ValueError(f"Invalid {endpoint} hard occupancy in {checkpoint}: {run_dir}")
            record[f"hard_k_{endpoint}"] = hard_k
            record[f"keff_hard_{endpoint}"] = keff_hard
            record[f"hard_k_minus_keff_hard_{endpoint}"] = hard_k - keff_hard
        records.append(record)
    return records


def write_tables(rows: list[dict[str, object]], out_dir: Path) -> None:
    occupancy_fields = tuple(f"{metric}_{end}" for metric in OCCUPANCY_COLUMNS for end in ENDPOINTS)
    for name, fields in (
        ("information", tuple(INFORMATION_COLUMNS)),
        ("cluster_occupancy", occupancy_fields),
    ):
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*IDENTITY_COLUMNS, *fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def table_grid(
    rows: list[dict[str, object]], scales: tuple[str, ...], methods: tuple[str, ...] = METHOD_LABELS,
) -> dict:
    grid = {
        (row["input_scale"], row["method"], row["stage"], (row["source_time"], row["target_time"])): row
        for row in rows
    }
    expected = set(product(scales, methods, STAGES, TIME_PAIRS))
    if len(rows) != len(grid) or grid.keys() != expected:
        raise ValueError("Expected one table record per scale, method, stage, and time pair.")
    return grid
