from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))


@pytest.fixture()
def spatial_builder(monkeypatch):
    import importlib

    fake_scanpy = types.SimpleNamespace(settings=types.SimpleNamespace(verbosity=0))
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    sys.modules.pop("domain_builder_louvain", None)
    sys.modules.pop("spatial_domain_builder", None)
    module = importlib.import_module("spatial_domain_builder")
    yield module
    sys.modules.pop("spatial_domain_builder", None)
    sys.modules.pop("domain_builder_louvain", None)


def test_build_spatial_connectivity_returns_symmetric_sparse_graph(spatial_builder) -> None:
    spatial = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )

    conn = spatial_builder.build_spatial_connectivity(spatial, n_neighbors=2)

    assert sp.isspmatrix_csr(conn)
    assert conn.shape == (5, 5)
    assert conn.diagonal().sum() == 0
    assert conn.nnz > 0
    assert np.isfinite(conn.data).all()
    assert (conn != conn.T).nnz == 0


def test_build_spatial_augmented_features_has_expected_shape_and_no_nan(spatial_builder) -> None:
    spatial = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pca = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ],
        dtype=np.float32,
    )
    conn = spatial_builder.build_spatial_connectivity(spatial, n_neighbors=2)

    augmented = spatial_builder.build_spatial_augmented_features(pca, conn, smooth_weight=0.3)

    assert augmented.shape == pca.shape
    assert np.isfinite(augmented).all()


def test_fuse_expression_spatial_connectivity_normalizes_sparse_result(spatial_builder) -> None:
    spatial = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    spatial_conn = spatial_builder.build_spatial_connectivity(spatial, n_neighbors=2)
    expr_conn = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0, 0.5],
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.5, 0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    fused = spatial_builder.fuse_expression_spatial_connectivity(
        expr_conn,
        spatial_conn,
        expr_weight=0.5,
        spatial_weight=0.5,
    )

    assert sp.isspmatrix_csr(fused)
    assert fused.shape == expr_conn.shape
    assert fused.diagonal().sum() == 0
    assert (fused != fused.T).nnz == 0
    assert fused.data.max() <= 1.0


def test_build_spatial_merge_features_includes_spatial_signal(spatial_builder) -> None:
    x_aug = np.zeros((4, 3), dtype=np.float32)
    spatial = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=np.float32)

    features = spatial_builder.build_spatial_merge_features(x_aug, spatial, merge_spatial_weight=0.25)

    assert features.shape == (4, 5)
    assert np.isfinite(features).all()
    assert not np.allclose(features[:, -2:], 0.0)
