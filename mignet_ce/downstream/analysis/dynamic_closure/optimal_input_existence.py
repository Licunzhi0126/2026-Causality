from __future__ import annotations

"""Post-hoc closure quality for persisted optimized coarse-graining runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import information_closure_budget


def evaluate_optimal_run(run_dir: Path) -> dict[str, object]:
    """Evaluate the saved best-checkpoint assignments against its input PIJ."""

    directory = Path(run_dir).resolve()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = ("S_t.npy", "S_tp.npy", "PIJ_micro_train.npy")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing closure inputs: {missing}")
    source = np.load(directory / "S_t.npy", allow_pickle=False)
    target = np.load(directory / "S_tp.npy", allow_pickle=False)
    micro_pij = np.load(directory / "PIJ_micro_train.npy", allow_pickle=False)
    budget = information_closure_budget(micro_pij, source, target)
    micro_ei = float(summary["EI_micro_fixed"])
    macro_ei = float(summary["EI_macro_best_checkpoint"])
    delta_ei = float(summary["delta_EI_best_checkpoint"])
    if not np.isclose(macro_ei - micro_ei, delta_ei, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Best-checkpoint DeltaEI does not match EI difference: {directory}")
    status = str(budget["signal_status"])
    return {
        "run_dir": str(directory),
        "method": str(summary.get("method", "")),
        "K": int(summary["K"]),
        "best_epoch": int(summary["best_epoch"]),
        "EI_micro_fixed": micro_ei,
        "EI_macro_best_checkpoint": macro_ei,
        "delta_EI_best_checkpoint": delta_ei,
        "I_available_bits": float(budget["I_available_bits"]),
        "I_macro_retained_bits": float(budget["I_macro_retained_bits"]),
        "closure_leakage_bits": float(budget["closure_leakage_bits"]),
        "closure_quality": float(budget["closure_quality"]),
        "closure_quality_for_claim": (
            float(budget["closure_quality"]) if status == "informative" else np.nan
        ),
        "signal_status": status,
        "signal_threshold_bits": float(budget["signal_threshold_bits"]),
        "information_identity_error": float(budget["information_identity_error"]),
        "weighting": str(budget["weighting"]),
        "n_input_t": int(source.shape[0]),
        "n_input_tp": int(target.shape[0]),
    }


def evaluate_optimal_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Evaluate one selected run per input scale and time pair."""

    rows: list[dict[str, object]] = []
    for run in runs.to_dict(orient="records"):
        result = evaluate_optimal_run(Path(str(run["run_dir"])))
        for key in ("input_scale", "time_pair", "seed"):
            result[key] = run[key]
        if int(result["K"]) != int(run["K"]):
            raise ValueError(f"K disagrees with run selection: {run['run_dir']}")
        if not np.isclose(
            float(result["delta_EI_best_checkpoint"]),
            float(run["delta_EI_best_checkpoint"]), rtol=1e-8, atol=1e-8,
        ):
            raise ValueError(f"DeltaEI disagrees with run selection: {run['run_dir']}")
        rows.append(result)
    return pd.DataFrame(rows)
