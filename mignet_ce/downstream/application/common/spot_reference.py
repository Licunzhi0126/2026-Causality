"""Cached spot-level PIJ references built only through the existing frontend."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


REQUIRED_CACHE_FILES = ("PIJ_spot.npy", "source_spots.csv", "target_spots.csv", "spot_reference_summary.json", "spot_reference_provenance.json")


def common_spot_reference_dir(output_root: Path) -> Path:
    """Return the sole authoritative spot-reference cache for one workflow run."""
    return Path(output_root) / "common" / "spot_reference"


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _request_provenance(request: object) -> dict[str, object]:
    values = asdict(request) if is_dataclass(request) else vars(request)
    return _jsonable(values)  # type: ignore[return-value]


def _load(cache_dir: Path) -> dict[str, object]:
    source_units = np.loadtxt(cache_dir / "source_spots.csv", dtype=str, delimiter=",", skiprows=1, ndmin=1).tolist()
    target_units = np.loadtxt(cache_dir / "target_spots.csv", dtype=str, delimiter=",", skiprows=1, ndmin=1).tolist()
    summary = json.loads((cache_dir / "spot_reference_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((cache_dir / "spot_reference_provenance.json").read_text(encoding="utf-8"))
    return {"pij": np.load(cache_dir / "PIJ_spot.npy"), "source_units": source_units, "target_units": target_units, "micro_ei": summary["micro_ei"], "cached": True, "provenance": provenance}


def load_spot_reference(cache_dir: Path) -> dict[str, object]:
    """Load a validated prebuilt reference; never rebuild it implicitly."""
    missing = [name for name in REQUIRED_CACHE_FILES if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Shared spot reference is incomplete in {cache_dir}: {missing}")
    return _load(cache_dir)


def build_spot_reference(request: object, cache_dir: Path) -> dict[str, object]:
    """Build or load one spot PIJ reference without reimplementing its features."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested_provenance = _request_provenance(request)
    provenance_path = cache_dir / "spot_reference_provenance.json"
    if all((cache_dir / name).is_file() for name in REQUIRED_CACHE_FILES):
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("request") == requested_provenance:
            return _load(cache_dir)
    from mignet_ce.coarse_frontends.complete_combined_coarse import prepare

    prepared = prepare(request)
    np.save(cache_dir / "PIJ_spot.npy", prepared.micro_pij)
    np.savetxt(cache_dir / "source_spots.csv", np.asarray(prepared.unit_ids_t, dtype=str), fmt="%s", delimiter=",", header="spot_id", comments="")
    np.savetxt(cache_dir / "target_spots.csv", np.asarray(prepared.unit_ids_tp, dtype=str), fmt="%s", delimiter=",", header="spot_id", comments="")
    units = {"source_units": list(prepared.unit_ids_t), "target_units": list(prepared.unit_ids_tp), "micro_ei": float(prepared.micro_ei)}
    summary = {"micro_ei": units["micro_ei"], "shape": list(np.asarray(prepared.micro_pij).shape), "n_source_spots": len(units["source_units"]), "n_target_spots": len(units["target_units"])}
    provenance = {"request": requested_provenance, "frontend": _jsonable(dict(prepared.provenance))}
    (cache_dir / "spot_reference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pij": prepared.micro_pij, **units, "cached": False, "provenance": provenance}
