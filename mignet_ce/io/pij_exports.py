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
    lower_units_by_time: Sequence[Sequence[str]] | None = None,
    upper_units_by_time: Sequence[Sequence[str]] | None = None,
    feature_alignment_space: str = "stable_upper_units",
) -> Path:
    archive_dir = pij_archive_directory(cfg, organ, pair)
    archive_dir.mkdir(parents=True, exist_ok=True)
    native_units = feature_alignment_space == "native_units"
    units = list(map(str, stable_upper_units))
    unit_mapping_files: dict[str, dict[str, str]] = {}
    if native_units:
        if lower_units_by_time is None or upper_units_by_time is None:
            raise ValueError("Native Pij export requires lower_units_by_time and upper_units_by_time.")
        if len(lower_units_by_time) != len(cfg.time_points) or len(upper_units_by_time) != len(cfg.time_points):
            raise ValueError("Native Pij unit mappings must match the configured number of time points.")
        units_dir = archive_dir / "units"
        units_dir.mkdir(parents=True, exist_ok=True)
        for stage, lower_stage_units, upper_stage_units in zip(
            map(str, cfg.time_points),
            lower_units_by_time,
            upper_units_by_time,
        ):
            pd.DataFrame(
                {
                    "index": range(len(lower_stage_units)),
                    "unit": list(map(str, lower_stage_units)),
                }
            ).to_csv(units_dir / f"lower_{stage}_units.csv", index=False)
            pd.DataFrame(
                {
                    "index": range(len(upper_stage_units)),
                    "unit": list(map(str, upper_stage_units)),
                }
            ).to_csv(units_dir / f"upper_{stage}_units.csv", index=False)
    else:
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
            if native_units:
                unit_lists = lower_units_by_time if space == "lower" else upper_units_by_time
                assert unit_lists is not None
                source_units = list(map(str, unit_lists[t0]))
                target_units = list(map(str, unit_lists[t1]))
                source_mapping = f"units/{space}_{source_stage}_units.csv"
                target_mapping = f"units/{space}_{target_stage}_units.csv"
            else:
                source_units = units
                target_units = units
                source_mapping = "units.csv"
                target_mapping = "units.csv"
            expected_shape = (len(source_units), len(target_units))
            if matrix_array.shape != expected_shape:
                raise ValueError(
                    f"{space} Pij matrix for {source_stage}->{target_stage} has shape "
                    f"{matrix_array.shape}; expected {expected_shape} from "
                    f"{source_mapping} and {target_mapping}."
                )
            if not np.all(np.isfinite(matrix_array)):
                raise ValueError(f"{space} Pij matrix for {source_stage}->{target_stage} contains non-finite values.")

            save_transition_npz(archive_dir / f"{label}_{space}_P.npz", matrix_array)

            if int(cfg.export_pij_topk) > 0:
                transition_topk_table(
                    matrix_array,
                    source_units=source_units,
                    target_units=target_units,
                    time_pair=f"{source_stage}->{target_stage}",
                    space=space,
                    top_k=cfg.export_pij_topk,
                    pij_method=cfg.effective_pij_method(),
                    diagnostic_costs=diagnostics_by_pair.get((t0, t1)),
                ).to_csv(archive_dir / f"{label}_{space}_P_topk.csv", index=False)
            unit_mapping_files[f"{label}_{space}_P.npz"] = {
                "source_units": source_mapping,
                "target_units": target_mapping,
            }

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
        "feature_alignment_space": feature_alignment_space,
        "unit_mapping_file": None if native_units else "units.csv",
        "unit_mapping_files": unit_mapping_files,
        "matrix_convention": (
            "For each *_P.npz, P[i,j] maps source_units row i to target_units row j "
            "using the files recorded in unit_mapping_files."
            if native_units
            else
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
    lower_units_by_time: Sequence[Sequence[str]] | None = None,
    upper_units_by_time: Sequence[Sequence[str]] | None = None,
    feature_alignment_space: str = "stable_upper_units",
) -> Path:
    """Compatibility alias for the sparse archive exporter."""
    return export_pij_sparse_archive(
        cfg=cfg,
        organ=organ,
        pair=pair,
        stable_upper_units=stable_upper_units,
        kernels=kernels,
        lower_units_by_time=lower_units_by_time,
        upper_units_by_time=upper_units_by_time,
        feature_alignment_space=feature_alignment_space,
    )
