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
            spatial = _read_h5ad_2d_node(handle["obsm"]["spatial"], source, "obsm/spatial")
            if spatial.shape[1] < 2:
                raise ValueError(
                    f"{source} obsm/spatial must contain at least two columns; got {spatial.shape}."
                )
            return spatial[:, :2]
        if "obs" in handle and "x" in handle["obs"] and "y" in handle["obs"]:
            return np.column_stack(
                [
                    np.asarray(handle["obs"]["x"][()], dtype=np.float32),
                    np.asarray(handle["obs"]["y"][()], dtype=np.float32),
                ]
            )
        return None


def _read_h5ad_2d_node(
    node: h5py.Dataset | h5py.Group,
    source: Path,
    field: str,
) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[()], dtype=np.float32)
    elif {"data", "indices", "indptr"}.issubset(node.keys()):
        values = _read_sparse_group(node, source, field).toarray().astype(np.float32, copy=False)
    elif _encoding_type(node) == "dataframe":
        values = _read_dataframe_group(node, source, field)
    else:
        raise ValueError(
            f"Unsupported H5AD encoding {_encoding_type(node)!r} for {field} in {source}."
        )
    if values.ndim != 2:
        raise ValueError(f"{source} {field} must be two-dimensional; got shape {values.shape}.")
    return values


def _read_dataframe_group(group: h5py.Group, source: Path, field: str) -> np.ndarray:
    raw_columns = group.attrs.get("column-order")
    if raw_columns is None:
        index_key = group.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        columns = [str(key) for key in group.keys() if str(key) != str(index_key)]
    else:
        if isinstance(raw_columns, (str, bytes)):
            raw_columns = [raw_columns]
        columns = decode_values(raw_columns)
    if not columns:
        raise ValueError(f"H5AD dataframe {field} in {source} contains no value columns.")

    arrays: list[np.ndarray] = []
    expected_rows: int | None = None
    for column in columns:
        if column not in group or not isinstance(group[column], h5py.Dataset):
            raise ValueError(
                f"H5AD dataframe {field} column {column!r} in {source} is missing or not a numeric dataset."
            )
        values = np.asarray(group[column][()], dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"H5AD dataframe {field} column {column!r} in {source} must be one-dimensional."
            )
        if expected_rows is None:
            expected_rows = int(values.shape[0])
        elif values.shape[0] != expected_rows:
            raise ValueError(f"H5AD dataframe {field} columns have inconsistent lengths in {source}.")
        arrays.append(values)
    return np.column_stack(arrays).astype(np.float32, copy=False)


def _encoding_type(node: h5py.Group) -> str:
    value = node.attrs.get("encoding-type", "csr_matrix")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _read_sparse_group(node: h5py.Group, source: Path, field: str) -> sp.csr_matrix:
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
    raise ValueError(f"Unsupported sparse H5AD encoding {encoding!r} for {field} in {source}.")


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
            return _read_sparse_group(node, source, f"X/layers/{layer}")
    raise ValueError(f"Cannot read X/layers/{layer} as dense, CSR, or CSC from {source}.")
