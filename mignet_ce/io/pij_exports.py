from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.pij.base import TransitionKernels
from mignet_ce.utils.matrix import save_transition_npz, serialize_metadata, transition_topk_table


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def pij_archive_directory(
    cfg: TemporalRunConfig,
    organ: str,
    pair: VerticalPairSpec,
) -> Path:
    return (
        cfg.effective_pij_archive_root()
        / f"network={cfg.network_method}"
        / f"pij={cfg.effective_pij_method()}"
        / f"organ={organ}"
        / f"pair={pair.label()}"
    )


def export_pij_sparse_archive(
    *,
    cfg: TemporalRunConfig,
    organ: str,
    pair: VerticalPairSpec,
    stable_upper_units: Sequence[str],
    kernels: TransitionKernels,
) -> Path:
    archive_dir = pij_archive_directory(cfg, organ, pair)
    archive_dir.mkdir(parents=True, exist_ok=True)
    units = list(map(str, stable_upper_units))
    pd.DataFrame({"index": range(len(units)), "unit": units}).to_csv(
        archive_dir / "units.csv",
        index=False,
    )

    for space, matrices in (("lower", kernels.p_lower), ("upper", kernels.p_upper)):
        diagnostics_by_pair = kernels.kernel_diagnostics.get(space, {})
        for (t0, t1), matrix in matrices.items():
            source_stage = str(cfg.time_points[t0])
            target_stage = str(cfg.time_points[t1])
            label = f"{source_stage}_to_{target_stage}"
            matrix_array = np.asarray(matrix, dtype=float)
            expected_shape = (len(units), len(units))
            if matrix_array.shape != expected_shape:
                raise ValueError(
                    f"{space} Pij matrix for {source_stage}->{target_stage} has shape "
                    f"{matrix_array.shape}; expected {expected_shape} from units.csv."
                )
            if not np.all(np.isfinite(matrix_array)):
                raise ValueError(f"{space} Pij matrix for {source_stage}->{target_stage} contains non-finite values.")

            save_transition_npz(archive_dir / f"{label}_{space}_P.npz", matrix_array)

            if int(cfg.export_pij_topk) > 0:
                transition_topk_table(
                    matrix_array,
                    source_units=units,
                    target_units=units,
                    time_pair=f"{source_stage}->{target_stage}",
                    space=space,
                    top_k=cfg.export_pij_topk,
                    pij_method=cfg.effective_pij_method(),
                    diagnostic_costs=diagnostics_by_pair.get((t0, t1)),
                ).to_csv(archive_dir / f"{label}_{space}_P_topk.csv", index=False)

    metadata = {
        "archive_root": str(cfg.effective_pij_archive_root()),
        "data_root": str(cfg.data_root),
        "output_root": str(cfg.output_root),
        "network_method": cfg.network_method,
        "pij_method": cfg.effective_pij_method(),
        "organ": str(organ),
        "lower_layer": pair.lower_layer,
        "upper_layer": pair.upper_layer,
        "time_points": list(map(str, cfg.time_points)),
        "top_k": int(cfg.export_pij_topk),
        "matrix_format": "scipy.sparse.csr_matrix saved by scipy.sparse.save_npz",
        "full_matrix": True,
        "topk_csv_written": int(cfg.export_pij_topk) > 0,
        "unit_mapping_file": "units.csv",
        "matrix_convention": (
            "For each *_P.npz, P[i,j] means transition probability from units.csv row i "
            "at source stage to units.csv row j at target stage."
        ),
        "kernel_metadata": serialize_metadata(kernels.kernel_metadata),
    }
    with (archive_dir / "kernel_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=_json_default)
    return archive_dir


def export_pij_csv_archive(
    *,
    cfg: TemporalRunConfig,
    organ: str,
    pair: VerticalPairSpec,
    stable_upper_units: Sequence[str],
    kernels: TransitionKernels,
) -> Path:
    """Compatibility alias for the sparse archive exporter."""
    return export_pij_sparse_archive(
        cfg=cfg,
        organ=organ,
        pair=pair,
        stable_upper_units=stable_upper_units,
        kernels=kernels,
    )
