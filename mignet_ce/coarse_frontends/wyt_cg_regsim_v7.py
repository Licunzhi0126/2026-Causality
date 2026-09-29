from __future__ import annotations

import numpy as np

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    build_n_pair,
    build_regsim_pair,
    compute_micro_ei,
    fixed_micro_features,
    load_spot_pair,
    pairwise_zscore,
    provenance_base,
)
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
)
from mignet_ce.pij.compare.compare_NR_kl_sinkhorn_regsim_v7 import (
    regsim_v7_pij_numpy,
    regsim_v7_pij_torch,
)
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput


def _macro_pij(inputs: MacroPijInputs):
    return regsim_v7_pij_torch(
        inputs.feature_blocks_t["N"],
        inputs.feature_blocks_tp["N"],
        inputs.feature_blocks_t["R"],
        inputs.feature_blocks_tp["R"],
    )


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    pair = load_spot_pair(request)
    regsim_t, regsim_tp = build_regsim_pair(request, pair)
    r_t, r_tp = pairwise_zscore(regsim_t.values, regsim_tp.values)
    n_t, n_tp, n_metadata = build_n_pair(pair.cci_t, pair.cci_tp, request)
    network_t = integrate_cci_regsim(
        pair.cci_t,
        build_regsim_similarity_network(regsim_t.values, k=request.regsim_knn_k),
        regsim_weight=request.regsim_weight,
    )
    network_tp = integrate_cci_regsim(
        pair.cci_tp,
        build_regsim_similarity_network(regsim_tp.values, k=request.regsim_knn_k),
        regsim_weight=request.regsim_weight,
    )
    encoder_t, encoder_tp = pairwise_zscore(np.hstack([n_t, r_t]), np.hstack([n_tp, r_tp]))
    micro_t, micro_tp = fixed_micro_features(encoder_t, encoder_tp, request.mid_dim)
    _, micro_pij, _, pij_metadata = regsim_v7_pij_numpy(n_t, n_tp, r_t, r_tp)
    prepared = PreparedCoarseInput(
        method="wyt_cg_regsim_v7",
        unit_ids_t=pair.unit_ids_t,
        unit_ids_tp=pair.unit_ids_tp,
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=encoder_t,
        encoder_features_tp=encoder_tp,
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=compute_micro_ei(micro_pij),
        macro_pij_builder=_macro_pij,
        feature_blocks_t={"N": n_t, "R": r_t},
        feature_blocks_tp={"N": n_tp, "R": r_tp},
        coords_t=pair.coords_t,
        coords_tp=pair.coords_tp,
        provenance={
            **provenance_base(request, pair),
            "network": "CCI_RegSim_integrated",
            "encoder": "pairwise_zscore(concat(N,R))",
            "N": n_metadata,
            "R_t": regsim_t.metadata,
            "R_tp": regsim_tp.metadata,
            "pij": "RegSim_V7_balanced_Sinkhorn",
            "pij_metadata": pij_metadata,
        },
    )
    prepared.validate()
    return prepared
