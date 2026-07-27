from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
)


def test_regsim_network_and_integration_are_row_normalized() -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    regsim = build_regsim_similarity_network(features, k=2)
    assert regsim.shape == (4, 4)
    np.testing.assert_allclose(regsim.toarray().sum(axis=1), 1.0, atol=1e-6)
    cci = sp.csr_matrix(
        [[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0]]
    )
    integrated = integrate_cci_regsim(cci, regsim, regsim_weight=0.2)
    np.testing.assert_allclose(integrated.toarray().sum(axis=1), 1.0, atol=1e-6)
    assert np.all(integrated.data >= 0.0)
