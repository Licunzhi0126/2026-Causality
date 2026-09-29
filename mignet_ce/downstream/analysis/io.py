from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

LAYER_PREFIXES: dict[str, tuple[str, ...]] = {
    "spot": ("spot",),
    "seurat_k150": ("seurat150",),
    "seurat_k40": ("seurat", "seurat40"),
    "seurat_k10": ("seurat10",),
}


def decode_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in {"O", "S", "U"}:
        return arr
    out = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in arr.reshape(-1)]
    return np.asarray(out, dtype=object).reshape(arr.shape)


def read_h5ad_index(group: h5py.Group) -> pd.Index:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    if key not in group:
        for candidate in ("_index", "cell_name", "gene_short_name", "index"):
            if candidate in group:
                key = candidate
                break
    if key not in group:
        raise ValueError(f"Could not locate an H5AD index in {group.name}")
    return pd.Index(decode_array(group[key][()]).astype(str), name=str(key))


def read_h5ad_sparse_matrix(obj: h5py.Dataset | h5py.Group) -> sp.csr_matrix:
    if isinstance(obj, h5py.Dataset):
        return sp.csr_matrix(np.asarray(obj[()], dtype=float))
    encoding = obj.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding not in {"csr_matrix", "csc_matrix"} and not {"data", "indices", "indptr"}.issubset(obj.keys()):
        raise ValueError(f"Unsupported H5AD matrix encoding: {encoding!r}")
    shape = tuple(int(value) for value in obj.attrs["shape"])
    matrix_class = sp.csc_matrix if encoding == "csc_matrix" else sp.csr_matrix
    matrix = matrix_class(
        (
            np.asarray(obj["data"][()], dtype=float),
            np.asarray(obj["indices"][()], dtype=np.int64),
            np.asarray(obj["indptr"][()], dtype=np.int64),
        ),
        shape=shape,
    )
    return matrix.tocsr()


def read_h5ad_expression(path: Path, *, prefer_counts: bool = True) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    with h5py.File(path, "r") as handle:
        matrix_obj = handle["X"]
        if prefer_counts and "layers" in handle:
            for key in ("count", "counts"):
                if key in handle["layers"]:
                    matrix_obj = handle["layers"][key]
                    break
        matrix = read_h5ad_sparse_matrix(matrix_obj)
        units = read_h5ad_index(handle["obs"])
        genes = read_h5ad_index(handle["var"])
    if matrix.shape != (len(units), len(genes)):
        raise ValueError(f"H5AD matrix shape {matrix.shape} does not match obs/var for {path}")
    return matrix, units, genes


def read_h5ad_units_coords(path: Path) -> tuple[list[str], np.ndarray]:
    """Read spot identifiers and two-dimensional coordinates without AnnData."""

    source = Path(path)
    with h5py.File(source, "r") as handle:
        units = read_h5ad_index(handle["obs"]).astype(str).tolist()
        if "obsm" in handle and "spatial" in handle["obsm"]:
            coords = np.asarray(handle["obsm"]["spatial"][()], dtype=float)[:, :2]
        elif "x" in handle["obs"] and "y" in handle["obs"]:
            coords = np.column_stack(
                [
                    np.asarray(handle["obs"]["x"][()], dtype=float),
                    np.asarray(handle["obs"]["y"][()], dtype=float),
                ]
            )
        else:
            coords = np.zeros((len(units), 2), dtype=float)
    if coords.shape != (len(units), 2):
        raise ValueError(f"Coordinate shape {coords.shape} does not match {len(units)} units in {source}")
    return units, coords


def load_npz_matrix(path: Path) -> np.ndarray:
    """Load a scipy sparse archive or a regular NumPy archive as a dense matrix."""

    source = Path(path)
    try:
        return np.asarray(sp.load_npz(source).toarray(), dtype=float)
    except (OSError, ValueError, KeyError):
        archive = np.load(source)
        if isinstance(archive, np.ndarray):
            return np.asarray(archive, dtype=float)
        try:
            key = "pij" if "pij" in archive.files else archive.files[0]
            return np.asarray(archive[key], dtype=float)
        finally:
            archive.close()


def _layer_stems(layer: str, organ: str, time: str) -> tuple[str, ...]:
    try:
        prefixes = LAYER_PREFIXES[layer]
    except KeyError as exc:
        raise ValueError(f"Unsupported layer {layer!r}") from exc
    return tuple(f"{prefix}_{organ}_{time}" for prefix in prefixes)


def canonical_layer_stem(layer: str, organ: str, time: str) -> str:
    """Return the primary on-disk stem used by the production data factory."""

    return _layer_stems(layer, organ, time)[0]


def _first_existing(candidates: Iterable[Path], description: str) -> Path:
    paths = list(candidates)
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {description}; checked: {', '.join(map(str, paths))}")


def layer_h5ad(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    return _first_existing(
        (Path(data_root) / layer / organ / f"{stem}.h5ad" for stem in _layer_stems(layer, organ, time)),
        f"{layer} H5AD for {organ} {time}",
    )


def domain_map_path(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    if layer not in {"seurat_k150", "seurat_k40", "seurat_k10"}:
        raise ValueError("domain maps are supported only for seurat_k150, seurat_k40, and seurat_k10")
    return _first_existing(
        (
            Path(data_root) / layer / organ / f"{stem}_spot_domain_map.csv"
            for stem in _layer_stems(layer, organ, time)
        ),
        f"{layer} domain map for {organ} {time}",
    )


def load_domain_map(data_root: Path, layer: str, time: str, organ: str = "heart") -> pd.DataFrame:
    path = domain_map_path(data_root, layer, time, organ)
    frame = pd.read_csv(path)
    required = {"spot_id", "domain_id", "x", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["domain_id"] = frame["domain_id"].astype(str)
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    return frame.dropna(subset=["x", "y"])


def cci_path(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    stems = _layer_stems(layer, organ, time)
    return _first_existing(
        (
            Path(data_root) / folder / layer / f"{stem}_CCI_total.npz"
            for folder in ("cci", "cci_clean")
            for stem in stems
        ),
        f"{layer} CCI for {organ} {time}",
    )


def cci_index_path(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    stems = _layer_stems(layer, organ, time)
    return _first_existing(
        (
            Path(data_root) / folder / layer / f"{stem}_index.tsv"
            for folder in ("cci", "cci_clean")
            for stem in stems
        ),
        f"{layer} CCI index for {organ} {time}",
    )


def read_index(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"CCI index is empty: {path}")
    return frame.iloc[:, 0].astype(str).tolist()


def grn_path(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    return _first_existing(
        (Path(data_root) / "grn" / layer / stem / "grn_edges.csv" for stem in _layer_stems(layer, organ, time)),
        f"{layer} GRN for {organ} {time}",
    )
