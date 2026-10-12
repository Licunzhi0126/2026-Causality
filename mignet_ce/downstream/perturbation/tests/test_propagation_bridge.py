from __future__ import annotations

import numpy as np

from mignet_ce.downstream.perturbation.propagation_bridge import propagate_delta


def test_propagation_uses_transposed_source_to_target_pij() -> None:
    pij = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.allclose(propagate_delta(np.array([1.0, 2.0]), pij), np.array([1.0, 2.0]))
