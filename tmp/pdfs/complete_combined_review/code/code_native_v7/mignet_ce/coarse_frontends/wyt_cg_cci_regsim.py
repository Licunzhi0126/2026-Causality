from __future__ import annotations

from functools import partial

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    build_regsim_pair,
    compute_micro_ei,
    load_spot_pair,
    provenance_base,
)
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
)
from mignet_ce.pij.wyt_single_kl import single_kl_pij_numpy, single_kl_pij_torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from mignet_ce.representations.wyt_network80 import build_network80_pair


def _macro_pij(inputs: MacroPijInputs, *, temperature: float):
    return single_kl_pij_torch(
        inputs.z_macro_t,
        inputs.z_macro_tp,
        temperature=temperature,
    )


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    pair = load_spot_pair(request)
    regsim_t, regsim_tp = build_regsim_pair(request, pair)
    r_graph_t = build_regsim_similarity_network(regsim_t.values, k=request.regsim_knn_k)
    r_graph_tp = build_regsim_similarity_network(regsim_tp.values, k=request.regsim_knn_k)
    network_t = integrate_cci_regsim(
        pair.cci_t,
        r_graph_t,
        regsim_weight=request.regsim_weight,
    )
    network_tp = integrate_cci_regsim(
        pair.cci_tp,
        r_graph_tp,
        regsim_weight=request.regsim_weight,
    )
    network80 = build_network80_pair(
        network_t,
        network_tp,
        svd_dim=request.network_svd_dim,
        pca_dim=request.mid_dim,
        random_state=request.seed,
    )
    _, micro_pij = single_kl_pij_numpy(
        network80.latent_t,
        network80.latent_tp,
        temperature=request.pij_temperature,
    )
    prepared = PreparedCoarseInput(
        method="wyt_cg_cci_regsim",
        unit_ids_t=pair.unit_ids_t,
        unit_ids_tp=pair.unit_ids_tp,
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=network80.features_t,
        encoder_features_tp=network80.features_tp,
        micro_features_t=network80.latent_t,
        micro_features_tp=network80.latent_tp,
        micro_pij=micro_pij.astype("float32"),
        micro_ei=compute_micro_ei(micro_pij),
        macro_pij_builder=partial(_macro_pij, temperature=request.pij_temperature),
        feature_blocks_t={"R": regsim_t.values},
        feature_blocks_tp={"R": regsim_tp.values},
        coords_t=pair.coords_t,
        coords_tp=pair.coords_tp,
        provenance={
            **provenance_base(request, pair),
            "network": "RowNorm((1-regsim_weight)*RowNorm(CCI)+regsim_weight*RowNorm(RegSim_kNN))",
            "regsim_weight": float(request.regsim_weight),
            "regsim_knn_k": int(request.regsim_knn_k),
            "regsim_t": regsim_t.metadata,
            "regsim_tp": regsim_tp.metadata,
            "pij": "single_direction_KL_on_integrated_network80_fixed_PCA",
            "network80": network80.metadata,
        },
    )
    prepared.validate()
    return prepared
