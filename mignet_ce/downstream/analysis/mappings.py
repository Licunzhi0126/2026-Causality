from __future__ import annotations

"""Soft-native records shared by the five-representation downstream suite."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .config import (
    MAPPING_COMPLETE,
    MAPPING_K150,
    MAPPING_K40,
    MAPPING_MATURITY,
    MAPPING_TWO_STAGE,
    UNIFIED_MAPPINGS,
    UnifiedDownstreamConfig,
)
from .io import (
    cci_index_path,
    layer_h5ad,
    load_domain_map,
    load_npz_matrix,
    read_h5ad_units_coords,
    read_index,
)
from .metrics import row_normalize


MAPPINGS = UNIFIED_MAPPINGS
LAYER_BY_MAPPING = {
    MAPPING_K150: "seurat_k150",
    MAPPING_K40: "seurat_k40",
}
OPTIMIZED_METHOD_BY_MAPPING = {
    MAPPING_COMPLETE: "complete_combined_coarse",
    MAPPING_MATURITY: "complete_combined_coarse_maturity_cci_grn",
    MAPPING_TWO_STAGE: "maturity_cci_grn_two_stage",
}
COLORS = {
    MAPPING_K150: "#2F74B8",
    MAPPING_K40: "#D45959",
    MAPPING_COMPLETE: "#4D9A73",
    MAPPING_MATURITY: "#8B5FBF",
    MAPPING_TWO_STAGE: "#C58A4A",
}
MARKERS = {
    MAPPING_K150: "o",
    MAPPING_K40: "s",
    MAPPING_COMPLETE: "D",
    MAPPING_MATURITY: "^",
    MAPPING_TWO_STAGE: "P",
}
DISPLAY_NAMES = {
    MAPPING_K150: "Seurat K150",
    MAPPING_K40: "Seurat K40",
    MAPPING_COMPLETE: "Complete",
    MAPPING_MATURITY: "Maturity+CCI+GRN",
    MAPPING_TWO_STAGE: "Maturity+CCI+GRN Two-stage",
}


def is_optimized(mapping: str) -> bool:
    return mapping in OPTIMIZED_METHOD_BY_MAPPING


@dataclass(frozen=True)
class MappingRecord:
    mapping: str
    pair: str
    p: np.ndarray
    source_assignment: np.ndarray
    target_assignment: np.ndarray
    q_direct: np.ndarray
    spots_s: tuple[str, ...]
    spots_t: tuple[str, ...]
    coords_s: np.ndarray
    coords_t: np.ndarray
    summary: dict[str, Any]
    method: str

    def __post_init__(self) -> None:
        p = np.asarray(self.p, dtype=float)
        source = np.asarray(self.source_assignment, dtype=float)
        target = np.asarray(self.target_assignment, dtype=float)
        q_direct = np.asarray(self.q_direct, dtype=float)
        if p.ndim != 2 or source.ndim != 2 or target.ndim != 2 or q_direct.ndim != 2:
            raise ValueError("P, assignments, and Q must all be two-dimensional")
        if p.shape != (source.shape[0], target.shape[0]):
            raise ValueError(
                f"P {p.shape} is incompatible with assignments {source.shape}/{target.shape}"
            )
        if q_direct.shape != (source.shape[1], target.shape[1]):
            raise ValueError(
                f"Q {q_direct.shape} is incompatible with assignments {source.shape}/{target.shape}"
            )
        for label, matrix in (
            ("P", p),
            ("source_assignment", source),
            ("target_assignment", target),
            ("Q", q_direct),
        ):
            if not np.isfinite(matrix).all() or np.any(matrix < -1e-10):
                raise ValueError(f"{label} contains invalid probabilities")
            row_sums = matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                raise ValueError(f"{label} rows must sum to one")
        if len(self.spots_s) != source.shape[0] or len(self.spots_t) != target.shape[0]:
            raise ValueError("spot IDs do not match assignment rows")
        if np.asarray(self.coords_s).shape != (source.shape[0], 2):
            raise ValueError("source coordinates must have shape (N_source, 2)")
        if np.asarray(self.coords_t).shape != (target.shape[0], 2):
            raise ValueError("target coordinates must have shape (N_target, 2)")

    @property
    def p_model_micro(self) -> np.ndarray:
        return self.p

    @property
    def q_model_full(self) -> np.ndarray:
        return self.q_direct


def natural_cache_path(cfg: UnifiedDownstreamConfig, layer: str, pair: str) -> Path:
    source, target = pair.split("->")
    root = cfg.full_cache_root / "natural"
    if layer == "spot":
        return root / "matrices" / f"pij_spot_{source}_to_{target}.npz"
    return root / "tables" / f"q_direct_{layer}_{source}_to_{target}.npy"


def natural_cache_manifest_path(cfg: UnifiedDownstreamConfig, layer: str, pair: str) -> Path:
    output = natural_cache_path(cfg, layer, pair)
    return output.with_suffix(output.suffix + ".full_manifest.json")


def optimized_pair_dir(cfg: UnifiedDownstreamConfig, mapping: str, pair: str) -> Path:
    try:
        method = OPTIMIZED_METHOD_BY_MAPPING[mapping]
    except KeyError as exc:
        raise ValueError(f"{mapping!r} is not an optimized mapping") from exc
    source, target = pair.split("->")
    return cfg.full_cache_root / "optimized_coarse" / method / f"{source}_to_{target}"


def _coords_in_order(cfg: UnifiedDownstreamConfig, time: str, units: list[str]) -> np.ndarray:
    h5_units, coords = read_h5ad_units_coords(layer_h5ad(cfg.data_root, "spot", time, cfg.organ))
    lookup = {unit: index for index, unit in enumerate(h5_units)}
    missing = [unit for unit in units if unit not in lookup]
    if missing:
        raise ValueError(f"Spot units are missing from H5AD at {time}: {missing[:10]}")
    return np.asarray(coords[[lookup[unit] for unit in units]], dtype=float)


def natural_assignment(
    cfg: UnifiedDownstreamConfig,
    layer: str,
    time: str,
    *,
    spot_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    spots = (
        list(map(str, spot_order))
        if spot_order is not None
        else read_index(cci_index_path(cfg.data_root, "spot", time, cfg.organ))
    )
    states = read_index(cci_index_path(cfg.data_root, layer, time, cfg.organ))
    frame = load_domain_map(cfg.data_root, layer, time, cfg.organ)
    lookup = frame.drop_duplicates("spot_id").set_index("spot_id")["domain_id"].astype(str)
    state_index = {state: index for index, state in enumerate(states)}
    missing_spots = [spot for spot in spots if spot not in lookup.index]
    if missing_spots:
        raise ValueError(f"Domain map {layer} {time} misses spots: {missing_spots[:10]}")
    unknown_states = sorted({str(lookup.loc[spot]) for spot in spots} - set(state_index))
    if unknown_states:
        raise ValueError(f"Domain map {layer} {time} contains states absent from CCI index: {unknown_states[:10]}")
    labels = np.asarray([state_index[str(lookup.loc[spot])] for spot in spots], dtype=int)
    assignment = np.zeros((len(spots), len(states)), dtype=float)
    assignment[np.arange(len(spots)), labels] = 1.0
    return assignment, tuple(spots), tuple(states)


def _optimized_record(cfg: UnifiedDownstreamConfig, mapping: str, pair: str) -> MappingRecord:
    root = optimized_pair_dir(cfg, mapping, pair)
    required = [
        "S_t.npy",
        "S_tp.npy",
        "PIJ_macro_train.npy",
        "PIJ_micro_train.npy",
        "assignments_t.csv",
        "assignments_tp.csv",
        "summary.json",
        "downstream_full_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete full DeltaEI cache {root}: {missing}")
    source_soft_full = row_normalize(np.load(root / "S_t.npy"))
    target_soft_full = row_normalize(np.load(root / "S_tp.npy"))
    direct = row_normalize(np.load(root / "PIJ_macro_train.npy"))
    if source_soft_full.shape[1] != direct.shape[0] or target_soft_full.shape[1] != direct.shape[1]:
        raise ValueError(
            f"Training assignments {source_soft_full.shape}/{target_soft_full.shape} do not match "
            f"PIJ_macro_train {direct.shape} in {root}"
        )
    source_frame = pd.read_csv(root / "assignments_t.csv")
    target_frame = pd.read_csv(root / "assignments_tp.csv")
    source_id = "spot_id" if "spot_id" in source_frame else source_frame.columns[0]
    target_id = "spot_id" if "spot_id" in target_frame else target_frame.columns[0]
    spots_s = tuple(source_frame[source_id].astype(str))
    spots_t = tuple(target_frame[target_id].astype(str))
    source_time, target_time = pair.split("->")
    coords_s = np.load(root / "coords_t.npy") if (root / "coords_t.npy").exists() else _coords_in_order(cfg, source_time, list(spots_s))
    coords_t = np.load(root / "coords_tp.npy") if (root / "coords_tp.npy").exists() else _coords_in_order(cfg, target_time, list(spots_t))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return MappingRecord(
        mapping=mapping,
        pair=pair,
        p=row_normalize(np.load(root / "PIJ_micro_train.npy")),
        source_assignment=source_soft_full,
        target_assignment=target_soft_full,
        q_direct=direct,
        spots_s=spots_s,
        spots_t=spots_t,
        coords_s=np.asarray(coords_s, dtype=float),
        coords_t=np.asarray(coords_t, dtype=float),
        summary=summary,
        method=OPTIMIZED_METHOD_BY_MAPPING[mapping],
    )


def _natural_record(cfg: UnifiedDownstreamConfig, mapping: str, pair: str) -> MappingRecord:
    layer = LAYER_BY_MAPPING[mapping]
    source, target = pair.split("->")
    source_spots = read_index(cci_index_path(cfg.data_root, "spot", source, cfg.organ))
    target_spots = read_index(cci_index_path(cfg.data_root, "spot", target, cfg.organ))
    source_assignment, spots_s, _ = natural_assignment(cfg, layer, source, spot_order=source_spots)
    target_assignment, spots_t, _ = natural_assignment(cfg, layer, target, spot_order=target_spots)
    direct_full = row_normalize(np.load(natural_cache_path(cfg, layer, pair)))
    if direct_full.shape != (source_assignment.shape[1], target_assignment.shape[1]):
        raise ValueError(
            f"Direct Q {direct_full.shape} does not match {layer} assignments "
            f"{source_assignment.shape}/{target_assignment.shape} for {pair}"
        )
    p_matrix = row_normalize(load_npz_matrix(natural_cache_path(cfg, "spot", pair)))
    if p_matrix.shape != (len(spots_s), len(spots_t)):
        raise ValueError(
            f"Spot P {p_matrix.shape} does not match assignments {(len(spots_s), len(spots_t))} for {pair}"
        )
    return MappingRecord(
        mapping=mapping,
        pair=pair,
        p=p_matrix,
        source_assignment=source_assignment,
        target_assignment=target_assignment,
        q_direct=direct_full,
        spots_s=spots_s,
        spots_t=spots_t,
        coords_s=_coords_in_order(cfg, source, list(spots_s)),
        coords_t=_coords_in_order(cfg, target, list(spots_t)),
        summary={},
        method=layer,
    )


def load_mapping_record(cfg: UnifiedDownstreamConfig, mapping: str, pair: str) -> MappingRecord:
    if mapping in OPTIMIZED_METHOD_BY_MAPPING:
        return _optimized_record(cfg, mapping, pair)
    if mapping in LAYER_BY_MAPPING:
        return _natural_record(cfg, mapping, pair)
    raise ValueError(f"Unknown unified mapping {mapping!r}")


def load_all_mapping_records(
    cfg: UnifiedDownstreamConfig,
) -> dict[tuple[str, str], MappingRecord]:
    return {
        (mapping, pair): load_mapping_record(cfg, mapping, pair)
        for pair in cfg.adjacent_pairs
        for mapping in MAPPINGS
    }
