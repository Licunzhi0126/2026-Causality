from __future__ import annotations

"""Developmental maturity input alignment and soft-assignment loss."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

EPS = 1e-12


@dataclass(frozen=True)
class AlignedMaturity:
    values: np.ndarray
    confidence: np.ndarray | None
    provenance: dict[str, object]


def _read_delimited(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Maturity file has no header: {path}")
        return [dict(row) for row in reader]


def _normalize(values: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "none":
        return values.astype(np.float32, copy=True)
    if normalization == "minmax":
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum - minimum < EPS:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - minimum) / (maximum - minimum)).astype(np.float32)
    if normalization == "rank":
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(values, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        return ranks
    raise ValueError("normalization must be one of none, minmax, rank.")


def load_maturity_csv(
    path: Path | str,
    unit_ids: Sequence[str],
    *,
    id_column: str = "spot_id",
    maturity_column: str,
    confidence_column: str | None = None,
    normalization: str = "minmax",
    direction: str = "higher_is_more_mature",
) -> AlignedMaturity:
    source = Path(path)
    rows = _read_delimited(source)
    expected = [str(unit_id) for unit_id in unit_ids]
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise ValueError("unit_ids must be unique for maturity alignment.")
    if not rows:
        raise ValueError(f"Maturity file is empty: {source}")
    fieldnames = set(rows[0])
    required = {id_column, maturity_column}
    if confidence_column is not None:
        required.add(confidence_column)
    missing_columns = sorted(required - fieldnames)
    if missing_columns:
        raise ValueError(f"Maturity file is missing columns: {missing_columns}.")

    by_id: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        unit_id = str(row[id_column])
        if unit_id in by_id:
            duplicate_ids.append(unit_id)
        by_id[unit_id] = row
    if duplicate_ids:
        raise ValueError(f"Duplicate maturity IDs: {sorted(set(duplicate_ids))}.")

    observed_set = set(by_id)
    missing_ids = sorted(expected_set - observed_set)
    unknown_ids = sorted(observed_set - expected_set)
    if missing_ids or unknown_ids:
        raise ValueError(
            "Maturity IDs must exactly match unit_ids; "
            f"missing maturity IDs: {missing_ids}; unknown maturity IDs: {unknown_ids}."
        )

    values = np.asarray(
        [float(by_id[unit_id][maturity_column]) for unit_id in expected],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Maturity values must be finite.")
    raw_min = float(values.min())
    raw_max = float(values.max())
    values = _normalize(values, normalization)
    if direction == "higher_is_more_primitive":
        values = 1.0 - values
    elif direction != "higher_is_more_mature":
        raise ValueError(
            "direction must be one of higher_is_more_mature, higher_is_more_primitive."
        )

    confidence = None
    if confidence_column is not None:
        confidence = np.asarray(
            [float(by_id[unit_id][confidence_column]) for unit_id in expected],
            dtype=np.float32,
        )
        if not np.isfinite(confidence).all() or np.any(confidence < 0.0):
            raise ValueError("Maturity confidence values must be finite and non-negative.")

    return AlignedMaturity(
        values=values.astype(np.float32, copy=False),
        confidence=confidence,
        provenance={
            "path": str(source),
            "id_column": id_column,
            "maturity_column": maturity_column,
            "confidence_column": confidence_column,
            "normalization": normalization,
            "direction": direction,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "unit_count": len(expected),
            "alignment": "exact_unit_id_match",
            "effective_training_scope": "within_timepoint_cluster_variance",
        },
    )


def developmental_loss(
    assignment: torch.Tensor,
    maturity: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    min_state_mass: float = 1e-8,
) -> torch.Tensor:
    if assignment.ndim != 2:
        raise ValueError("assignment must be a 2D tensor.")
    if maturity.ndim != 1 or maturity.shape[0] != assignment.shape[0]:
        raise ValueError("maturity must be 1D and match assignment rows.")
    values = maturity.to(dtype=assignment.dtype, device=assignment.device)
    weights = assignment
    if confidence is not None:
        if confidence.ndim != 1 or confidence.shape[0] != assignment.shape[0]:
            raise ValueError("confidence must be 1D and match assignment rows.")
        weights = weights * confidence.to(
            dtype=assignment.dtype,
            device=assignment.device,
        ).unsqueeze(1)
    mass = weights.sum(dim=0).clamp_min(float(min_state_mass))
    center = (weights * values.unsqueeze(1)).sum(dim=0) / mass
    variance = (
        weights * (values.unsqueeze(1) - center.unsqueeze(0)).pow(2)
    ).sum(dim=0) / mass
    mass_weight = mass / mass.sum().clamp_min(EPS)
    return (variance * mass_weight).sum()


def hard_assignment_maturity_summary(
    assignment: np.ndarray,
    maturity: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(assignment).argmax(axis=1)
    values = np.asarray(maturity, dtype=np.float32)
    spans: list[float] = []
    variances: list[float] = []
    for label in range(np.asarray(assignment).shape[1]):
        selected = values[labels == label]
        if selected.size == 0:
            continue
        spans.append(float(selected.max() - selected.min()))
        variances.append(float(selected.var()))
    return {
        "maturity_hard_span_mean": float(np.mean(spans)) if spans else 0.0,
        "maturity_hard_span_max": float(np.max(spans)) if spans else 0.0,
        "maturity_hard_variance_mean": float(np.mean(variances)) if variances else 0.0,
    }
