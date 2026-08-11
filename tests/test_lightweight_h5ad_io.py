from __future__ import annotations

import h5py
import numpy as np

from mignet_ce.io.h5ad_h5py import read_h5ad_spatial
from mignet_ce.visualization.downstream.io import read_h5ad_units_coords


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
    units, coords = read_h5ad_units_coords(path)
    assert units == ["a", "b"]
    assert np.allclose(coords, expected)
