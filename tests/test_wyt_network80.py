from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mignet_ce.representations.wyt_network80 import build_network80_pair


def test_network80_pair_is_deterministic_and_jointly_aligned() -> None:
    rng = np.random.default_rng(7)
    source = sp.csr_matrix(rng.random((9, 9)))
    target = sp.csr_matrix(rng.random((8, 8)))
    first = build_network80_pair(source, target, svd_dim=3, pca_dim=5, random_state=11)
    second = build_network80_pair(source, target, svd_dim=3, pca_dim=5, random_state=11)
    assert first.features_t.shape == (9, 22)
    assert first.features_tp.shape == (8, 22)
    assert first.latent_t.shape == (9, 5)
    assert first.latent_tp.shape == (8, 5)
    np.testing.assert_allclose(first.features_t, second.features_t)
    np.testing.assert_allclose(first.latent_tp, second.latent_tp)
    combined = np.vstack([first.features_t, first.features_tp])
    assert np.max(np.abs(combined.mean(axis=0))) < 0.05
