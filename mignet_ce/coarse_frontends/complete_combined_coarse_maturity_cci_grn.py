from __future__ import annotations

"""Sparse complete-combined LightCCI-GRN preparation with required maturity."""

from functools import partial

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    load_spot_pair,
    provenance_base,
)
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.complete_combined import (
    build_macro_pij_builder,
    prepare_complete_pair,
    prepare_complete_stage,
    strict_complete_combined_evaluation,
)
from wyt_deltaei_coarse_grain.complete_combined_maturity import _attach_required_maturity


METHOD = "complete_combined_coarse_maturity_cci_grn"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    if request.grn_t is None or request.grn_tp is None:
        raise ValueError(f"{METHOD} requires --grn-t and --grn-tp.")
    if request.maturity_t is None or request.maturity_tp is None:
        raise ValueError(f"{METHOD} requires --maturity-t and --maturity-tp.")

    pair_data = load_spot_pair(request)
    stage_t = prepare_complete_stage(
        h5ad_path=request.h5ad_t,
        grn_path=request.grn_t,
        units=pair_data.unit_ids_t,
        cci=pair_data.cci_t,
        top_k_targets=request.grn_topk_targets,
        state_dim=request.grn_state_dim,
        projection_seed=request.grn_projection_seed,
    )
    stage_tp = prepare_complete_stage(
        h5ad_path=request.h5ad_tp,
        grn_path=request.grn_tp,
        units=pair_data.unit_ids_tp,
        cci=pair_data.cci_tp,
        top_k_targets=request.grn_topk_targets,
        state_dim=request.grn_state_dim,
        projection_seed=request.grn_projection_seed,
    )
    pair = prepare_complete_pair(
        stage_t,
        stage_tp,
        nmf_components=request.nmf_components,
        nmf_max_iter=request.nmf_max_iter,
        seed=request.seed,
        mid_dim=request.mid_dim,
    )
    prepared = PreparedCoarseInput(
        method=METHOD,
        unit_ids_t=stage_t.units,
        unit_ids_tp=stage_tp.units,
        network_t=row_normalize_sparse(stage_t.cci),
        network_tp=row_normalize_sparse(stage_tp.cci),
        encoder_features_t=pair.encoder_t,
        encoder_features_tp=pair.encoder_tp,
        micro_features_t=pair.micro_features_t,
        micro_features_tp=pair.micro_features_tp,
        micro_pij=pair.micro_pij,
        micro_ei=pair.micro_ei,
        macro_pij_builder=build_macro_pij_builder(stage_t, stage_tp),
        feature_blocks_t={"N": pair.n_t, "X": stage_t.expression_grn},
        feature_blocks_tp={"N": pair.n_tp, "X": stage_tp.expression_grn},
        independent_width_feature_blocks=frozenset({"X"}),
        coords_t=pair_data.coords_t,
        coords_tp=pair_data.coords_tp,
        provenance={
            **provenance_base(request, pair_data),
            "coarse_method": METHOD,
            "source_network_method": "light_cci_grn",
            "uses_cci": True,
            "uses_grn": True,
            "coarse_core": "complete_combined_N_plus_G",
            "network_adjacency_policy": "row_normalized_true_CCI",
            "macro_feature_mode": "pool_expression_then_recompute_GRN_G",
            "feature_extractor": "Native_V7_N_plus_true_GRN_G",
            "nmf_components": int(request.nmf_components),
            "nmf_max_iter_used": int(request.nmf_max_iter),
            "feature_seed": int(request.seed),
            "N": pair.n_metadata,
            "Native_V7": pair.v7_metadata,
            "stage_t": stage_t.metadata,
            "stage_tp": stage_tp.metadata,
            "loader_optimization": "sparse_complete_stage_equivalent_to_registered_light_cci_grn_core",
            "strict_posthoc_filename": "strict_complete_combined_maturity_evaluation.json",
        },
        posthoc_evaluator=partial(
            strict_complete_combined_evaluation,
            stage_t,
            stage_tp,
            pair,
            nmf_components=request.nmf_components,
            nmf_max_iter=request.nmf_max_iter,
            seed=request.seed,
        ),
    )
    prepared.validate()
    return _attach_required_maturity(prepared, request)
