"""Audit selected domain results and compare each variant with its baseline."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL


INFORMATION = ("delta_ei", "cq", "i_available", "i_retained")
OCCUPANCY = tuple(f"{name}_{side}" for name in ("hard_k", "keff_hard", "hard_k_minus_keff_hard")
                  for side in ("t", "tp"))
METRICS = INFORMATION + OCCUPANCY
DIAGNOSTICS = {
    "ei_macro": "EI_macro_best_checkpoint", "closure_js_mean": "closure_js_mean",
    "closure_js_median": "closure_js_median", "closure_js_p90": "closure_js_p90",
    "development_loss": "L_dev_best_checkpoint", "stage1_delta_ei": "stage1_best_delta_EI",
    "stage1_i_available": "stage1_reference_I_available", "stage1_i_retained": "stage1_reference_I_retained",
}
EXTRA_COMPARISONS = ("ei_macro", "closure_js_mean", "closure_js_median", "closure_js_p90", "development_loss")


def hard_occupancy(soft: np.ndarray) -> tuple[int, float]:
    values = np.asarray(soft)
    if (values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("Invalid soft assignment probabilities")
    counts = np.bincount(values.argmax(axis=1), minlength=values.shape[1])
    occupied = counts[counts > 0]
    frequencies = occupied / occupied.sum(dtype=np.float64)
    return len(occupied), min(float(len(occupied)), float(np.exp(-(frequencies * np.log(frequencies)).sum())))


def read_run(run_dir: Path, *, variant: str, pair: tuple[str, str], k: int = 40) -> dict:
    """Use the final verified best_joint state; never select a new epoch."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    consistency = json.loads((run_dir / "checkpoint_consistency.json").read_text(encoding="utf-8"))
    if (summary.get("method") != "formal_two_stage" or summary.get("loss_protocol") != FORMAL_TWO_STAGE_PROTOCOL
            or config.get("objective_version") != FORMAL_TWO_STAGE_PROTOCOL
            or summary.get("K") != k or config.get("k") != k
            or summary.get("selected_checkpoint") != "best_joint.pt"
            or summary.get("checkpoint_metric_state_consistent") is not True
            or consistency.get("passed") is not True
            or consistency.get("selected_epoch") != summary.get("best_epoch")
            or consistency.get("selected_checkpoint") != "best_joint.pt"):
        raise ValueError(f"Unverified final domain checkpoint: {run_dir}")
    for name in ("best_joint.pt", "best_ei.pt"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)
    from .variants import VARIANTS, FACTORIAL_VARIANTS

    specification = next(item for item in (*VARIANTS, *FACTORIAL_VARIANTS) if item.name == variant)
    if any(config.get(key) != value for key, value in specification.weights.items()):
        raise ValueError(f"Saved weights differ from {variant}: {run_dir}")
    record = {
        "variant": variant, "source_time": pair[0], "target_time": pair[1], "input_scale": "spot", "k": k,
        "run_dir": str(run_dir.resolve()), "checkpoint_epoch": int(summary["best_epoch"]),
        "checkpoint_selection": summary["best_joint_selection"],
        "fallback_used": bool(summary["best_joint_fallback_used"]),
        "delta_ei": float(summary["delta_EI_best_checkpoint"]), "cq": float(summary["closure_quality"]),
        "i_available": float(summary["I_available"]), "i_retained": float(summary["I_retained"]),
    }
    for name, key in DIAGNOSTICS.items():
        value = summary.get(key)
        record[name] = None if value is None else float(value)
    signal = summary.get("signal_status_reference")
    record["dynamic_pair_weight"] = None if signal is None else (1.0 if signal == "informative" else config["low_signal_pair_weight"])
    record.update(specification.weights)
    if variant in {item.name for item in FACTORIAL_VARIANTS} and any(
        record[name] is None or not np.isfinite(record[name]) for name in (*DIAGNOSTICS, "dynamic_pair_weight")
    ):
        raise ValueError(f"Missing finite factorial diagnostics: {run_dir}")
    if not all(np.isfinite(record[name]) for name in INFORMATION):
        raise ValueError(f"Non-finite information metrics: {run_dir}")
    if not (0 <= record["cq"] <= 1 + 1e-5 and record["i_available"] >= 0 and record["i_retained"] >= 0):
        raise ValueError(f"Invalid information budget: {run_dir}")
    if not np.isclose(record["delta_ei"], summary["EI_macro_best_checkpoint"] - summary["EI_micro_fixed"], atol=1e-5):
        raise ValueError(f"Delta EI identity mismatch: {run_dir}")
    if record["i_available"] > 1e-8 and not np.isclose(record["cq"], record["i_retained"] / record["i_available"], atol=1e-5):
        raise ValueError(f"CQ identity mismatch: {run_dir}")
    for side in ("t", "tp"):
        soft = np.load(run_dir / f"S_{side}.npy", allow_pickle=False)
        hard_k, keff = hard_occupancy(soft)
        if soft.shape[1] != k or summary[f"hardK_{side}"] != hard_k or not np.isclose(summary[f"KeffHard_{side}"], keff, atol=1e-5):
            raise ValueError(f"Saved occupancy differs from assignments ({side}): {run_dir}")
        record.update({f"hard_k_{side}": hard_k, f"keff_hard_{side}": keff,
                       f"hard_k_minus_keff_hard_{side}": hard_k - keff})
    return record


def baseline_differences(rows: list[dict], reference: str = "baseline") -> list[dict]:
    identities = ("input_scale", "source_time", "target_time", "k")
    lookup = {}
    baselines = {}
    for row in rows:
        key = tuple(row[field] for field in identities)
        identity = (row["variant"], *key)
        if identity in lookup:
            raise ValueError(f"Duplicate domain result: {identity}")
        lookup[identity] = row
        if row["variant"] == reference:
            baselines[key] = row
    result = []
    for row in rows:
        key = tuple(row[field] for field in identities)
        if key not in baselines:
            raise ValueError(f"Missing baseline for {key}")
        record = dict(row, comparison_reference=reference)
        optional = tuple(name for name in EXTRA_COMPARISONS
                         if row.get(name) is not None and baselines[key].get(name) is not None)
        for metric in (*METRICS, *optional):
            value, reference_value = row[metric], baselines[key][metric]
            if not np.isfinite(value) or not np.isfinite(reference_value):
                raise ValueError(f"Non-finite comparison metric {metric}")
            record[f"delta_vs_baseline_{metric}"] = value - reference_value
        result.append(record)
    return result


def write_records(records: list[dict], path: Path) -> None:
    if not records:
        raise ValueError("Cannot export an empty domain report")
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_metrics(rows: list[dict], path: Path, reference: str = "baseline") -> None:
    write_records(baseline_differences(rows, reference), path)


def history_diagnostics(run: Path, *, variant: str, pair: tuple[str, str]) -> dict:
    """Audit raw Stage 2 penalty activation, independently of selected-epoch metrics."""
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != config["epochs"] or any(int(row["epoch"]) != index for index, row in enumerate(history, 1)):
        raise ValueError(f"Incomplete training history: {run}")
    stage2 = [row for row in history if row["stage"] == "stage2"]
    if not stage2:
        raise ValueError(f"Missing Stage 2 history: {run}")
    record = dict(variant=variant, source_time=pair[0], target_time=pair[1], stage2_epochs=len(stage2),
                  run_dir=str(run.resolve()))

    def values(name):
        result = np.array([float(row[name]) for row in stage2])
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite history column {name}: {run}")
        return result

    for name in ("ei_constraint", "retain_floor"):
        raw = values(f"L_{name}")
        active = int((raw > 0).sum())
        record.update({f"{name}_active_epochs": active, f"{name}_active_fraction": active / len(stage2),
                       f"{name}_mean": float(raw.mean()), f"{name}_max": float(raw.max())})
    for name in ("weighted_ei_constraint", "weighted_retain_floor", "weighted_dev", "weighted_closure", "weighted_retain_norm"):
        series = values(name)
        record[f"{name}_mean"] = float(series.mean())
        record[f"{name}_abs_mean"] = float(np.abs(series).mean())
    return record


def pareto_report(rows: list[dict], reference: str = "s8") -> dict:
    """Retain tradeoffs across individual time pairs; never average CQ or pick a winner."""
    from .variants import FACTORIAL_VARIANTS, TIME_PAIRS

    lookup = {(row["variant"], (row["source_time"], row["target_time"])): row for row in rows}
    names = [variant.name for variant in FACTORIAL_VARIANTS]
    if len(lookup) != len(rows) or set(lookup) != {(name, pair) for name in names for pair in TIME_PAIRS}:
        raise ValueError("Pareto analysis requires the complete 8 × 3 factorial grid")

    def objectives(row):
        vector = [row["delta_ei"], row["cq"], -row["closure_js_mean"]]
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite Pareto objectives")
        return vector

    def compare(vectors):
        result = []
        for name, vector in vectors.items():
            vector = np.asarray(vector)
            dominators = [other for other, alternative in vectors.items() if other != name
                          and np.all(np.asarray(alternative) >= vector) and np.any(np.asarray(alternative) > vector)]
            result.append(dict(variant=name, pareto_optimal=not dominators, dominated_by=dominators))
        return result

    return dict(reference=reference, maximize=["delta_ei", "cq"], minimize=["closure_js_mean"],
                selection="Report candidates only; no tolerance gates or automatic winner.",
                per_time_pair=[dict(time_pair=list(pair), candidates=compare({name: objectives(lookup[name, pair]) for name in names}))
                               for pair in TIME_PAIRS],
                cross_time_pair=compare({name: [value for pair in TIME_PAIRS for value in objectives(lookup[name, pair])]
                                        for name in names}))


def factorial_effects(rows: list[dict]) -> list[dict]:
    """Conventional two-level effects: 2 * mean(product of coded factor signs * response)."""
    from .variants import FACTORIAL_VARIANTS, TIME_PAIRS

    # Also validates missing/duplicate configurations and finite primary objectives.
    pareto_report(rows)
    specifications = {variant.name: variant.weights for variant in FACTORIAL_VARIANTS}
    effects = []
    for pair in TIME_PAIRS:
        group = [row for row in rows if (row["source_time"], row["target_time"]) == pair]
        for size in (1, 2, 3):
            for term in combinations(("C", "D", "R"), size):
                signs = []
                for row in group:
                    weights = specifications[row["variant"]]
                    code = {"C": 1 if weights["lambda_closure"] == 1 else -1,
                            "D": 1 if weights["lambda_dev"] == 2 else -1,
                            "R": 1 if weights["lambda_retain_norm"] == 1 else -1}
                    signs.append(np.prod([code[factor] for factor in term]))
                record = dict(source_time=pair[0], target_time=pair[1], term=":".join(term),
                              definition="2 * mean(coded_sign_product * response)")
                for metric in (*METRICS, *EXTRA_COMPARISONS):
                    values = np.asarray([row[metric] for row in group], dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite factorial response: {metric}")
                    record[f"effect_{metric}"] = float(2 * np.mean(np.asarray(signs) * values))
                effects.append(record)
    return effects


def metric_lines(row: dict | None, *, baseline: bool) -> list[str]:
    if row is None:
        return [f"{label}: unavailable" for label in ("ΔEI", "CQ", "I_available", "HardK", "HardK − KeffHard")]
    prefix = "" if baseline else "delta_vs_baseline_"

    def number(metric, *, integer=False):
        value = row[prefix + metric]
        value = 0 if abs(value) < 1e-12 else value
        if integer:
            return format(int(value), "d" if baseline else "+d")
        if value and abs(value) < 1e-4:
            return format(value, ".2e" if baseline else "+.2e")
        return format(value, ".4f" if baseline else "+.4f")

    return [f"{label}: {number(metric)}" for label, metric in
            (("ΔEI", "delta_ei"), ("CQ", "cq"), ("I_available", "i_available"))] + [
        "HardK: " + " / ".join(number(f"hard_k_{side}", integer=True) for side in ("t", "tp")),
        "HardK − KeffHard: " + " / ".join(number(f"hard_k_minus_keff_hard_{side}") for side in ("t", "tp")),
    ]
