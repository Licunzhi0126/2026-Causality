from __future__ import annotations

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.downstream.analysis.dynamic_closure.optimal import ngklot_pij_numpy
from mignet_ce.downstream.sensitive_analysis.kernel import (
    balanced_pij_from_cost,
    build_component_costs,
    mix_cost,
)
from mignet_ce.pij.compare.NG_KLot import NGKLotPijMethod
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_numpy
from wyt_deltaei_coarse_grain.complete_combined import canonical_ng_pij_numpy as complete_pij


def test_all_production_adapters_share_the_canonical_transition() -> None:
    rng = np.random.default_rng(920)
    n_t = rng.normal(size=(5, 4))
    n_tp = rng.normal(size=(6, 4))
    g_t = rng.normal(size=(5, 7))
    g_tp = rng.normal(size=(6, 7))

    _, expected, _ = canonical_ng_pij_numpy(n_t, n_tp, g_t, g_tp)
    _, complete, _ = complete_pij(n_t, n_tp, g_t, g_tp)
    dynamic, _ = ngklot_pij_numpy(n_t, n_tp, g_t, g_tp)

    d_n, d_g, _ = build_component_costs(
        n_t,
        n_tp,
        g_t,
        g_tp,
        beta_n=0.05,
        beta_g=0.05,
    )
    sensitivity_cost, _ = mix_cost(d_n, d_g, alpha_cci=0.1)
    sensitivity, sensitivity_metadata = balanced_pij_from_cost(sensitivity_cost, tau=0.1)

    method = NGKLotPijMethod()
    _, _, ngklot, diagnostics = method._build_pair_kernel(
        source=n_t,
        target=n_tp,
        cfg=TemporalRunConfig(),
        grn_source=g_t,
        grn_target=g_tp,
    )

    for actual in (complete, dynamic, sensitivity, ngklot):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert sensitivity_metadata["transition_protocol"] == "canonical_ng_v1"
    assert diagnostics["transition_protocol"] == "canonical_ng_v1"
    assert diagnostics["legacy_cfg_pij_temperature_received_but_not_used"] == 1.0
