from __future__ import annotations

from functools import partial

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    compute_micro_ei,
    load_spot_pair,
    provenance_base,
)
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
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
    network80 = build_network80_pair(
        pair.cci_t,
        pair.cci_tp,
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
        method="wyt_cg_cci",
        unit_ids_t=pair.unit_ids_t,
        unit_ids_tp=pair.unit_ids_tp,
        network_t=row_normalize_sparse(pair.cci_t),
        network_tp=row_normalize_sparse(pair.cci_tp),
        encoder_features_t=network80.features_t,
        encoder_features_tp=network80.features_tp,
        micro_features_t=network80.latent_t,
        micro_features_tp=network80.latent_tp,
        micro_pij=micro_pij.astype("float32"),
        micro_ei=compute_micro_ei(micro_pij),
        macro_pij_builder=partial(_macro_pij, temperature=request.pij_temperature),
        coords_t=pair.coords_t,
        coords_tp=pair.coords_tp,
        provenance={
            **provenance_base(request, pair),
            "network": "raw_CCI",
            "pij": "single_direction_KL_on_network80_fixed_PCA",
            "network80": network80.metadata,
        },
    )
    prepared.validate()
    return prepared
