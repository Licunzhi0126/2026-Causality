from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.coarse_frontends.registry import COARSE_FRONTEND_REGISTRY
from mignet_ce.representations.coarse_input import PreparedCoarseInput


def test_coarse_frontend_registry_has_exact_planned_methods() -> None:
    assert set(COARSE_FRONTEND_REGISTRY) == {
        "wyt_cg_cci",
        "wyt_cg_cci_regsim",
        "wyt_cg_regsim_v7",
        "wyt_cg_regsim_v9",
    }


def test_prepared_coarse_input_rejects_nonstochastic_pij() -> None:
    prepared = PreparedCoarseInput(
        method="synthetic",
        unit_ids_t=["a", "b"],
        unit_ids_tp=["c", "d"],
        network_t=sp.eye(2, format="csr"),
        network_tp=sp.eye(2, format="csr"),
        encoder_features_t=np.ones((2, 2)),
        encoder_features_tp=np.ones((2, 2)),
        micro_features_t=np.ones((2, 2)),
        micro_features_tp=np.ones((2, 2)),
        micro_pij=np.ones((2, 2)),
        micro_ei=0.0,
        macro_pij_builder=lambda _: None,
    )
    with pytest.raises(ValueError, match="row-stochastic"):
        prepared.validate()
