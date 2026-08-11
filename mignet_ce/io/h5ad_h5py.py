from __future__ import annotations

"""Small h5py readers for H5AD fields used by lightweight preparation scripts."""

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def decode_values(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def read_axis_index(handle: h5py.File, axis: str, path: Path) -> list[str]:
    group = handle[axis]
    index_key = group.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if str(index_key) in group:
        return decode_values(group[str(index_key)][()])
    if "_index" in group:
        return decode_values(group["_index"][()])
    raise ValueError(f"{path} must contain an encoded {axis} index.")


def read_h5ad_axis_names(path: Path | str) -> tuple[list[str], list[str]]:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        return read_axis_index(handle, "obs", source), read_axis_index(handle, "var", source)


def read_h5ad_spatial(path: Path | str) -> np.ndarray | None:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        if "obsm" in handle and "spatial" in handle["obsm"]:
            return np.asarray(handle["obsm"]["spatial"][()], dtype=np.float32)[:, :2]
        if "obs" in handle and "x" in handle["obs"] and "y" in handle["obs"]:
            return np.column_stack(
                [
                    np.asarray(handle["obs"]["x"][()], dtype=np.float32),
                    np.asarray(handle["obs"]["y"][()], dtype=np.float32),
                ]
            )
        return None


def _encoding_type(node: h5py.Group) -> str:
    value = node.attrs.get("encoding-type", "csr_matrix")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def read_h5ad_csr_matrix(path: Path | str, *, layer: str = "count") -> sp.csr_matrix:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        layers = handle.get("layers")
        if layers is not None and layer in layers:
            node = layers[layer]
        elif layers is not None and layer == "count" and "counts" in layers:
            node = layers["counts"]
        else:
            node = handle["X"]
        if isinstance(node, h5py.Dataset):
            return sp.csr_matrix(np.asarray(node[()], dtype=np.float32))
        if {"data", "indices", "indptr"}.issubset(node.keys()):
            shape = tuple(int(value) for value in node.attrs["shape"])
            values = (
                np.asarray(node["data"][()], dtype=np.float32),
                np.asarray(node["indices"][()], dtype=np.int64),
                np.asarray(node["indptr"][()], dtype=np.int64),
            )
            encoding = _encoding_type(node)
            if encoding == "csr_matrix":
                return sp.csr_matrix(values, shape=shape)
            if encoding == "csc_matrix":
                return sp.csc_matrix(values, shape=shape).tocsr()
            raise ValueError(f"Unsupported sparse H5AD encoding {encoding!r} in {source}.")
    raise ValueError(f"Cannot read X/layers/{layer} as dense, CSR, or CSC from {source}.")
