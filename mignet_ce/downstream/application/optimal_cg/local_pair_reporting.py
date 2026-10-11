"""Stable CSV/JSON and array reporting for Plan 6 pair analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .local_pair_builder import LocalPairData


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_directory(root: Path, source_cluster: int, target_cluster: int) -> Path:
    return root / "pairs" / f"source_{source_cluster:03d}__target_{target_cluster:03d}"


def write_pair_inputs(root: Path, pair: LocalPairData, config: dict[str, object]) -> Path:
    directory = pair_directory(root, pair.source_cluster, pair.target_cluster)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "source_indices.npy", pair.source_indices_used)
    np.save(directory / "source_indices_original.npy", pair.source_indices_original)
    np.save(directory / "target_indices.npy", pair.target_indices)
    np.save(directory / "PIJ_local_raw.npy", pair.pij_raw)
    np.save(directory / "PIJ_local_conditional.npy", pair.pij_conditional)
    np.save(directory / "retained_mass.npy", pair.retained_mass)
    write_json(directory / "local_config.json", config)
    return directory


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
