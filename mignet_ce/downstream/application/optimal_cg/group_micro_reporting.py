"""Stable group CSV, strict JSON, and optional binary array exports."""
import csv
import json
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_group_inputs(directory, data, export_matrices=True):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "source_indices_original": data.source_indices_original,
        "source_indices_used": data.source_indices_used,
        "target_indices": data.target_indices,
        "retained_mass": data.retained_mass,
        "dropped_mask": data.dropped_mask,
    }
    if export_matrices:
        arrays.update(PIJ_local_raw=data.pij_raw, PIJ_local_conditional=data.pij_conditional)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
