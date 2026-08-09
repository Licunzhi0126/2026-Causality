from __future__ import annotations

import numpy as np
import pytest

from mignet_ce.visualization.downstream.dynamic_closure.ultradeep import (
    conditional_mi_triplet,
    macro_triplet_joint,
)


def test_identity_macrostate_preserves_first_order_markovity() -> None:
    p01 = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    p12 = np.asarray([[0.9, 0.1], [0.25, 0.75]], dtype=float)
    identity = np.eye(2)
    joint = macro_triplet_joint(p01, p12, identity, identity, identity)
    cmi, per_state, state_mass = conditional_mi_triplet(joint)
    assert joint.sum() == pytest.approx(1.0)
    assert cmi == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(per_state, 0.0, atol=1e-12)
    assert state_mass.sum() == pytest.approx(1.0)
