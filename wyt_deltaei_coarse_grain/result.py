from __future__ import annotations

"""Artifact writer for the v40-derived learner.

Adaptation: emits the report's stable manifest, real-ID assignment and sparse
PIJ/network artifact contract in addition to WYT-compatible ``.npy`` files.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.assignment import assignment_rows


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if sp.issparse(value):
        return {
            "sparse_format": value.getformat(),
            "shape": list(value.shape),
            "nnz": int(value.nnz),
        }
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_static_manifests(
    out_dir: Path,
    prepared: PreparedCoarseInput,
    config: object,
) -> None:
    write_json(out_dir / "config.json", asdict(config))
    write_json(out_dir / "input_manifest.json", prepared.manifest())
    write_json(
        out_dir / "feature_manifest.json",
        {
            "method": prepared.method,
            "encoder_features_t": list(prepared.encoder_features_t.shape),
            "encoder_features_tp": list(prepared.encoder_features_tp.shape),
            "micro_features_t": list(prepared.micro_features_t.shape),
            "micro_features_tp": list(prepared.micro_features_tp.shape),
            "feature_blocks": {
                name: {
                    "t": list(np.asarray(prepared.feature_blocks_t[name]).shape),
                    "tp": list(np.asarray(prepared.feature_blocks_tp[name]).shape),
                }
                for name in prepared.feature_blocks_t
            },
            "provenance": dict(prepared.provenance),
        },
    )


def write_final_arrays(
    out_dir: Path,
    prepared: PreparedCoarseInput,
    *,
    assignment_t: np.ndarray,
    assignment_tp: np.ndarray,
    macro_network_t: np.ndarray,
    macro_network_tp: np.ndarray,
    macro_features_t: np.ndarray,
    macro_features_tp: np.ndarray,
    macro_pij: np.ndarray,
    summary: dict[str, object],
) -> None:
    arrays = {
        "S_t.npy": assignment_t,
        "S_tp.npy": assignment_tp,
        "P_macro_t.npy": macro_network_t,
        "P_macro_tp.npy": macro_network_tp,
        "Z_micro_t.npy": prepared.micro_features_t,
        "Z_micro_tp.npy": prepared.micro_features_tp,
        "Z_macro_t.npy": macro_features_t,
        "Z_macro_tp.npy": macro_features_tp,
        "PIJ_micro_train.npy": prepared.micro_pij,
        "PIJ_macro_train.npy": macro_pij,
    }
    for name, values in arrays.items():
        np.save(out_dir / name, np.asarray(values, dtype=np.float32))
    sp.save_npz(out_dir / "P_macro_t.npz", sp.csr_matrix(macro_network_t))
    sp.save_npz(out_dir / "P_macro_tp.npz", sp.csr_matrix(macro_network_tp))
    sp.save_npz(out_dir / "PIJ_micro.npz", sp.csr_matrix(prepared.micro_pij))
    sp.save_npz(out_dir / "PIJ_macro.npz", sp.csr_matrix(macro_pij))
    sp.save_npz(out_dir / "PIJ_macro_train.npz", sp.csr_matrix(macro_pij))
    write_csv(out_dir / "assignments_t.csv", assignment_rows(prepared.unit_ids_t, assignment_t))
    write_csv(
        out_dir / "assignments_tp.csv",
        assignment_rows(prepared.unit_ids_tp, assignment_tp),
    )
    if prepared.coords_t is not None:
        np.save(out_dir / "coords_t.npy", np.asarray(prepared.coords_t, dtype=np.float32))
    if prepared.coords_tp is not None:
        np.save(out_dir / "coords_tp.npy", np.asarray(prepared.coords_tp, dtype=np.float32))
    write_json(out_dir / "summary.json", summary)
