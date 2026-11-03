#!/usr/bin/env python3
"""Safely promote a verified diagnostic Method 10 run into Formal Core optimal_runs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.formal_core_analysis.tables import inspect_optimal_run


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def promote(source: Path, target: Path) -> dict:
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target:
        raise ValueError("Source and target run directories must be different.")
    if not source.is_dir():
        raise FileNotFoundError(f"Diagnostic source directory does not exist: {source}")

    verified = inspect_optimal_run(source)
    expected = ("shr_method_10", "spot", "11.5->13.5")
    observed = (verified["method"], verified["input_scale"], verified["time_pair"])
    if observed != expected:
        raise ValueError(f"Expected {expected}, got {observed}; refusing promotion.")

    config_path = source / "config.json"
    config = _read_json(config_path)
    if float(config.get("checkpoint_keff_min", float("nan"))) != 1e-6:
        raise ValueError("Diagnostic run was not trained with checkpoint_keff_min=1e-6.")
    if float(config.get("keff_min", float("nan"))) != 12.0:
        raise ValueError("Diagnostic run does not record the expected K=40 keff_min=12.")

    summary = _read_json(source / "summary.json")
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch <= 0 or float(verified["trained_delta_EI"]) <= 0:
        raise ValueError("Diagnostic run lacks a positive-Delta-EI selected checkpoint.")

    if not (source / "best_model.pt").is_file():
        raise FileNotFoundError(f"Missing selected Method 10 model checkpoint: {source / 'best_model.pt'}")
    history_path = source / "training_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    import pandas as pd
    history = pd.read_csv(history_path)
    selected = history.loc[pd.to_numeric(history["epoch"], errors="coerce").eq(best_epoch)]
    if len(selected) != 1 or not bool(selected.iloc[0]["checkpoint_eligible"]):
        raise ValueError("The saved best epoch is not marked checkpoint_eligible in training_history.csv.")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.name == "K40" and target.parents[1].name == "spot" and target.parents[2].name == "shr_method_10" and target.parents[3].name == "optimal_runs":
        optimal_root = target.parents[3]
        backup = optimal_root.parent / "_incomplete_run_backups" / target.relative_to(optimal_root)
    else:
        backup = target.with_name(target.name + ".pre_ckpt1e-6_backup")
    temporary = target.with_name(target.name + ".promotion_tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary promotion directory already exists: {temporary}")
    if target.exists():
        if (target / "summary.json").is_file():
            try:
                existing = inspect_optimal_run(target)
            except Exception:
                existing = None
            if existing is not None:
                try:
                    existing_config = _read_json(target / "config.json")
                    existing_checkpoint_floor = float(existing_config.get("checkpoint_keff_min", float("nan")))
                    existing_training_floor = existing_config.get("keff_min")
                    if existing_training_floor is None:
                        existing_training_floor = 0.30 * int(existing_config.get("k", existing.get("K", 0)))
                    already_matches = (
                        abs(existing_checkpoint_floor - 1e-6) <= 1e-12
                        and abs(float(existing_training_floor) - 12.0) <= 1e-12
                    )
                except (OSError, ValueError, TypeError):
                    already_matches = False
                if already_matches:
                    raise FileExistsError(f"Target already contains a valid run with the approved Method 10 protocol: {target}")
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)

    try:
        shutil.copytree(source, temporary)
        copied_config = _read_json(temporary / "config.json")
        copied_config["out_dir"] = str(target)
        (temporary / "config.json").write_text(
            json.dumps(copied_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        promotion = {
            "promotion_type": "verified_diagnostic_run_to_formal_optimal_run",
            "source_run": str(source),
            "target_run": str(target),
            "method": verified["method"],
            "input_scale": verified["input_scale"],
            "time_pair": verified["time_pair"],
            "best_epoch": best_epoch,
            "trained_delta_EI": float(verified["trained_delta_EI"]),
            "keff_source": float(verified["Keff_t"]),
            "keff_target": float(verified["Keff_tp"]),
            "keff_min": 12.0,
            "checkpoint_keff_min": 1e-6,
            "scientific_arrays_copied_without_modification": True,
        }
        (temporary / "promotion_manifest.json").write_text(
            json.dumps(promotion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(target)
        post = inspect_optimal_run(target)
        if not (post["method"] == verified["method"]
                and post["input_scale"] == verified["input_scale"]
                and post["time_pair"] == verified["time_pair"]
                and abs(float(post["trained_delta_EI"]) - float(verified["trained_delta_EI"])) <= 1e-8):
            raise RuntimeError("Post-promotion audit did not reproduce the source scientific result.")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if target.exists():
            failed = backup.with_name(backup.name + ".promotion_failed")
            if failed.exists():
                raise FileExistsError(f"Promotion failed and backup path exists: {failed}")
            failed.parent.mkdir(parents=True, exist_ok=True)
            target.rename(failed)
        if backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            backup.rename(target)
        raise

    return {**promotion, "target_verified": True,
            "preserved_previous_incomplete_run_at": str(backup) if backup.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.source, args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
