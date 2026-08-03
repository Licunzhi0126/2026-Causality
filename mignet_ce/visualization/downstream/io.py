from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DownstreamConfig


LAYER_PREFIXES: dict[str, tuple[str, ...]] = {
    "spot": ("spot",),
    "seurat_k150": ("seurat150",),
    "seurat_k40": ("seurat", "seurat40"),
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


def _layer_stems(layer: str, organ: str, time: str) -> tuple[str, ...]:
    try:
        prefixes = LAYER_PREFIXES[layer]
    except KeyError as exc:
        raise ValueError(f"Unsupported layer {layer!r}") from exc
    return tuple(f"{prefix}_{organ}_{time}" for prefix in prefixes)


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
    if layer not in {"seurat_k150", "seurat_k40"}:
        raise ValueError("domain maps are supported only for seurat_k150 and seurat_k40")
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


def grn_path(data_root: Path, layer: str, time: str, organ: str = "heart") -> Path:
    return _first_existing(
        (Path(data_root) / "grn" / layer / stem / "grn_edges.csv" for stem in _layer_stems(layer, organ, time)),
        f"{layer} GRN for {organ} {time}",
    )


def load_units(pair_dir: Path, space: str, time: str) -> list[str]:
    path = Path(pair_dir) / "units" / f"{space}_{time}_units.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Unit mapping is empty: {path}")
    column = "unit" if "unit" in frame.columns else frame.columns[-1]
    return frame[column].astype(str).tolist()


def load_pij(pair_dir: Path, time_pair: str, space: str) -> np.ndarray:
    left, right = time_pair.split("->")
    path = Path(pair_dir) / f"{left}_to_{right}_{space}_P.npz"
    return sp.load_npz(path).toarray().astype(float, copy=False)


def load_filtered_metrics(cfg: DownstreamConfig) -> pd.DataFrame:
    frame = pd.read_csv(cfg.metrics_csv)
    rename = {}
    if "EI_local" in frame.columns and "EI_lower" not in frame.columns:
        rename["EI_local"] = "EI_lower"
    if "EI_global" in frame.columns and "EI_upper" not in frame.columns:
        rename["EI_global"] = "EI_upper"
    frame = frame.rename(columns=rename)
    required = {"organ", "lower_layer", "upper_layer", "time_pair", "EI_lower", "EI_upper"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{cfg.metrics_csv} is missing columns {sorted(missing)}")
    mask = (
        (frame["organ"].astype(str) == cfg.organ)
        & (frame["lower_layer"].astype(str) == cfg.lower_layer)
        & (frame["upper_layer"].astype(str) == cfg.upper_layer)
    )
    if "network_method" in frame.columns:
        mask &= frame["network_method"].astype(str) == cfg.network_method
    if "pij_method" in frame.columns:
        mask &= frame["pij_method"].astype(str) == cfg.pij_method
    frame = frame.loc[mask].copy()
    frame = frame[frame["time_pair"].astype(str).isin(cfg.all_pairs)].copy()
    if frame.empty:
        raise ValueError(
            "No metrics match the downstream selection: "
            f"network={cfg.network_method}, pij={cfg.pij_method}, organ={cfg.organ}, "
            f"pair={cfg.lower_layer}->{cfg.upper_layer}"
        )
    duplicates = frame["time_pair"].astype(str).duplicated(keep=False)
    if duplicates.any():
        repeated = sorted(frame.loc[duplicates, "time_pair"].astype(str).unique())
        raise ValueError(f"Metrics selection is not unique for time pairs: {repeated}")
    expected = set(cfg.all_pairs)
    observed = set(frame["time_pair"].astype(str))
    if expected != observed:
        raise ValueError(f"Metrics time-pair mismatch; missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    if "EI_gain" not in frame.columns:
        frame["EI_gain"] = pd.to_numeric(frame["EI_upper"]) - pd.to_numeric(frame["EI_lower"])
    order = {pair: index for index, pair in enumerate(cfg.all_pairs)}
    frame["time_pair"] = frame["time_pair"].astype(str)
    frame["lag"] = frame["time_pair"].map(
        lambda pair: cfg.times.index(pair.split("->")[1]) - cfg.times.index(pair.split("->")[0])
    )
    return frame.sort_values("time_pair", key=lambda values: values.map(order)).reset_index(drop=True)


def validate_pair_archive(cfg: DownstreamConfig) -> None:
    for time in cfg.times:
        for space in ("lower", "upper"):
            load_units(cfg.pair_archive, space, time)
    for pair in cfg.all_pairs:
        source, target = pair.split("->")
        for space in ("lower", "upper"):
            matrix = load_pij(cfg.pair_archive, pair, space)
            expected = (
                len(load_units(cfg.pair_archive, space, source)),
                len(load_units(cfg.pair_archive, space, target)),
            )
            if matrix.shape != expected:
                raise ValueError(f"{pair} {space} Pij shape {matrix.shape} != unit mapping {expected}")
