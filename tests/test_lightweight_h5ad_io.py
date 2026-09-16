from __future__ import annotations

import h5py
import numpy as np
import scipy.sparse as sp

from mignet_ce.io.h5ad_h5py import read_h5ad_spatial


def test_lightweight_coordinate_reader_preserves_obs_xy_fallback(tmp_path) -> None:
    path = tmp_path / "xy_only.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray([b"a", b"b"]))
        obs.create_dataset("x", data=np.asarray([1.0, 2.0]))
        obs.create_dataset("y", data=np.asarray([3.0, 4.0]))
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray([b"g1"]))
    expected = np.asarray([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(read_h5ad_spatial(path), expected)


def test_downstream_coordinate_reader_preserves_obs_xy_fallback(tmp_path) -> None:
    from mignet_ce.downstream.analysis.io import read_h5ad_units_coords

    path = tmp_path / "xy_only_downstream.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray([b"a", b"b"]))
        obs.create_dataset("x", data=np.asarray([1.0, 2.0]))
        obs.create_dataset("y", data=np.asarray([3.0, 4.0]))
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray([b"g1"]))
    expected = np.asarray([[1.0, 3.0], [2.0, 4.0]])
    units, coords = read_h5ad_units_coords(path)
    assert units == ["a", "b"]
    assert np.allclose(coords, expected)


def test_lightweight_coordinate_reader_supports_dense_obsm_dataset(tmp_path) -> None:
    path = tmp_path / "dense_spatial.h5ad"
    expected = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=expected)
    assert np.allclose(read_h5ad_spatial(path), expected)


def test_lightweight_coordinate_reader_supports_anndata_dataframe_group(tmp_path) -> None:
    path = tmp_path / "dataframe_spatial.h5ad"
    expected = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        spatial = handle.create_group("obsm").create_group("spatial")
        spatial.attrs["encoding-type"] = "dataframe"
        spatial.attrs["encoding-version"] = "0.2.0"
        spatial.attrs["_index"] = "_index"
        spatial.attrs["column-order"] = np.asarray(["x", "y"], dtype=h5py.string_dtype())
        spatial.create_dataset("_index", data=np.asarray(["a", "b"], dtype=h5py.string_dtype()))
        spatial.create_dataset("x", data=expected[:, 0])
        spatial.create_dataset("y", data=expected[:, 1])
    assert np.allclose(read_h5ad_spatial(path), expected)


def test_lightweight_coordinate_reader_supports_sparse_obsm_group(tmp_path) -> None:
    path = tmp_path / "sparse_spatial.h5ad"
    expected = np.asarray([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    matrix = sp.csr_matrix(expected)
    with h5py.File(path, "w") as handle:
        spatial = handle.create_group("obsm").create_group("spatial")
        spatial.attrs["encoding-type"] = "csr_matrix"
        spatial.attrs["encoding-version"] = "0.1.0"
        spatial.attrs["shape"] = matrix.shape
        spatial.create_dataset("data", data=matrix.data)
        spatial.create_dataset("indices", data=matrix.indices)
        spatial.create_dataset("indptr", data=matrix.indptr)
    assert np.allclose(read_h5ad_spatial(path), expected)
