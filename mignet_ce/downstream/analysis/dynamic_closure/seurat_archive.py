from __future__ import annotations

"""Read native-unit PIJ exports without silently changing their row order."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class ArchivedTransition:
    values: np.ndarray
    source_units: tuple[str, ...]
    target_units: tuple[str, ...]
    matrix_path: Path
    source_units_path: Path
    target_units_path: Path


@dataclass(frozen=True)
class SeuratPairArchive:
    directory: Path
    metadata: dict[str, object]

    def transition(self, source_time: str, target_time: str, side: str) -> ArchivedTransition:
        if side not in {"lower", "upper"}:
            raise ValueError(f"Invalid PIJ side: {side}")
        label = f"{source_time}_to_{target_time}_{side}_P.npz"
        mappings = self.metadata.get("unit_mapping_files")
        if not isinstance(mappings, dict) or label not in mappings:
            raise ValueError(f"Missing unit mapping for {label} in {self.directory}")
        mapping = mappings[label]
        if not isinstance(mapping, dict) or not {"source_units", "target_units"} <= mapping.keys():
            raise ValueError(f"Incomplete unit mapping for {label} in {self.directory}")
        matrix_path = self.directory / label
        source_path = self.directory / str(mapping["source_units"])
        target_path = self.directory / str(mapping["target_units"])
        source_units = _read_units(source_path)
        target_units = _read_units(target_path)
        values = np.asarray(sp.load_npz(matrix_path).toarray(), dtype=np.float64)
        if values.shape != (len(source_units), len(target_units)):
            raise ValueError(f"PIJ shape {values.shape} does not match units for {matrix_path}")
        if not np.isfinite(values).all() or np.any(values < -1e-10):
            raise ValueError(f"PIJ contains invalid probabilities: {matrix_path}")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"PIJ rows are not stochastic: {matrix_path}")
        return ArchivedTransition(
            values=values,
            source_units=source_units,
            target_units=target_units,
            matrix_path=matrix_path,
            source_units_path=source_path,
            target_units_path=target_path,
        )


def _read_units(path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(path, dtype={"unit": str})
    if list(frame.columns) != ["index", "unit"] or frame.empty:
        raise ValueError(f"Expected nonempty index/unit table: {path}")
    indices = pd.to_numeric(frame["index"], errors="coerce").to_numpy()
    if not np.array_equal(indices, np.arange(len(frame))):
        raise ValueError(f"Unit indices are not sequential in {path}")
    if frame["unit"].isna().any() or frame["unit"].duplicated().any():
        raise ValueError(f"Unit names are missing or duplicated in {path}")
    return tuple(frame["unit"].astype(str))


def load_seurat_pair_archive(
    archive_root: Path,
    *,
    organ: str,
    macro_layer: str,
    network_method: str,
    pij_method: str,
    times: tuple[str, ...],
) -> SeuratPairArchive:
    directory = (
        Path(archive_root)
        / f"network={network_method}"
        / f"pij={pij_method}"
        / f"organ={organ}"
        / f"pair=spot_to_{macro_layer}"
    )
    metadata_path = directory / "kernel_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "network_method": network_method,
        "pij_method": pij_method,
        "organ": organ,
        "lower_layer": "spot",
        "upper_layer": macro_layer,
        "feature_alignment_space": "native_units",
        "full_matrix": True,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"PIJ archive {directory} has {field}={metadata.get(field)!r}; expected {value!r}"
            )
    if tuple(map(str, metadata.get("time_points", []))) != times:
        raise ValueError(f"PIJ archive {directory} has different time points")
    return SeuratPairArchive(directory=directory, metadata=metadata)


def file_descriptor(path: Path) -> dict[str, object]:
    source = Path(path).resolve()
    stat = source.stat()
    return {"path": str(source), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
