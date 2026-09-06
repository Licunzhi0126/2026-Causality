from __future__ import annotations

"""Shared four-representation records for the topic-oriented downstream suite."""

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
    UNIFIED_MAPPINGS,
    UnifiedDownstreamConfig,
)
from .dynamic_closure.analysis import to_hard_assignment
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
}
COLORS = {
    MAPPING_K150: "#2F74B8",
    MAPPING_K40: "#D45959",
    MAPPING_COMPLETE: "#4D9A73",
    MAPPING_MATURITY: "#8B5FBF",
}
MARKERS = {
    MAPPING_K150: "o",
    MAPPING_K40: "s",
    MAPPING_COMPLETE: "D",
    MAPPING_MATURITY: "^",
}
DISPLAY_NAMES = {
    MAPPING_K150: "K150",
    MAPPING_K40: "K40",
    MAPPING_COMPLETE: "Opt-Complete",
    MAPPING_MATURITY: "Opt-Maturity",
}


def is_optimized(mapping: str) -> bool:
    return mapping in OPTIMIZED_METHOD_BY_MAPPING


@dataclass(frozen=True)
class MappingRecord:
    """One mapping/pair in the complete model state space.

    The historical field names are retained for compatibility, but their
    formal contract is now explicit: ``p`` is the matched model micro P,
    ``q_direct`` is the complete model Q, ``soft_s``/``soft_t`` retain every
    model prototype, and ``hs``/``ht`` are full-width argmax projections.
    Hard-active compaction belongs exclusively to the closure module.
    """

    mapping: str
    pair: str
    p: np.ndarray
    hs: np.ndarray
    ht: np.ndarray
    q_direct: np.ndarray
    spots_s: tuple[str, ...]
    spots_t: tuple[str, ...]
    soft_s: np.ndarray
    soft_t: np.ndarray
    coords_s: np.ndarray
    coords_t: np.ndarray
    summary: dict[str, Any]
    method: str

    def __post_init__(self) -> None:
        p = np.asarray(self.p)
        hs = np.asarray(self.hs)
        ht = np.asarray(self.ht)
        q = np.asarray(self.q_direct)
        soft_s = np.asarray(self.soft_s)
        soft_t = np.asarray(self.soft_t)
        if p.shape != (len(self.spots_s), len(self.spots_t)):
            raise ValueError(
                f"Matched micro P {p.shape} does not match spot substrate "
                f"{(len(self.spots_s), len(self.spots_t))} for {self.mapping} {self.pair}"
            )
        if hs.shape != soft_s.shape or ht.shape != soft_t.shape:
            raise ValueError(
                f"Hard/soft assignment shapes disagree for {self.mapping} {self.pair}: "
                f"source={hs.shape}/{soft_s.shape}, target={ht.shape}/{soft_t.shape}"
            )
        if hs.shape[0] != len(self.spots_s) or ht.shape[0] != len(self.spots_t):
            raise ValueError(f"Assignment rows do not match spots for {self.mapping} {self.pair}")
        if q.shape != (soft_s.shape[1], soft_t.shape[1]):
            raise ValueError(
                f"Full model Q {q.shape} does not match full assignment widths "
                f"{(soft_s.shape[1], soft_t.shape[1])} for {self.mapping} {self.pair}"
            )

    @property
    def p_model_micro(self) -> np.ndarray:
        return self.p

    @property
    def q_model_full(self) -> np.ndarray:
        return self.q_direct

    @property
    def hard_s_full(self) -> np.ndarray:
        return self.hs

    @property
    def hard_t_full(self) -> np.ndarray:
        return self.ht

    @property
    def soft_s_full(self) -> np.ndarray:
        return self.soft_s

    @property
    def soft_t_full(self) -> np.ndarray:
        return self.soft_t


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
    source_hard = to_hard_assignment(source_soft_full)
    target_hard = to_hard_assignment(target_soft_full)
    direct = row_normalize(np.load(root / "PIJ_macro_train.npy"))
    if direct.shape != (source_soft_full.shape[1], target_soft_full.shape[1]):
        raise ValueError(
            f"Full direct Q {direct.shape} does not match optimized assignment widths "
            f"{(source_soft_full.shape[1], target_soft_full.shape[1])} in {root}"
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
        hs=source_hard,
        ht=target_hard,
        q_direct=direct,
        spots_s=spots_s,
        spots_t=spots_t,
        soft_s=source_soft_full,
        soft_t=target_soft_full,
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
            f"Full direct Q {direct_full.shape} does not match natural assignment widths "
            f"{(source_assignment.shape[1], target_assignment.shape[1])} for {layer} {pair}"
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
        hs=source_assignment,
        ht=target_assignment,
        q_direct=direct_full,
        spots_s=spots_s,
        spots_t=spots_t,
        soft_s=source_assignment,
        soft_t=target_assignment,
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
