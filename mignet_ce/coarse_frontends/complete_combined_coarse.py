from __future__ import annotations

"""Complete Native-V7 and WYT coarse-graining frontend."""

from functools import partial

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    load_spot_pair,
    provenance_base,
)
from wyt_deltaei_coarse_grain.complete_combined import (
    build_macro_pij_builder,
    prepare_complete_pair,
    prepare_complete_stage,
    strict_complete_combined_evaluation,
)
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    if request.grn_t is None or request.grn_tp is None:
        raise ValueError(
            "complete_combined_coarse requires --grn-t and --grn-tp."
        )
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
    network_t = integrate_cci_regsim(
        stage_t.cci,
        build_regsim_similarity_network(pair.g_t, k=request.grn_knn_k),
        regsim_weight=request.grn_graph_weight,
    )
    network_tp = integrate_cci_regsim(
        stage_tp.cci,
        build_regsim_similarity_network(pair.g_tp, k=request.grn_knn_k),
        regsim_weight=request.grn_graph_weight,
    )
    prepared = PreparedCoarseInput(
        method="complete_combined_coarse",
        unit_ids_t=stage_t.units,
        unit_ids_tp=stage_tp.units,
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=pair.encoder_t,
        encoder_features_tp=pair.encoder_tp,
        micro_features_t=pair.micro_features_t,
        micro_features_tp=pair.micro_features_tp,
        micro_pij=pair.micro_pij,
        micro_ei=pair.micro_ei,
        macro_pij_builder=build_macro_pij_builder(stage_t, stage_tp),
        feature_blocks_t={
            "N": pair.n_t,
            "X": stage_t.expression_grn,
        },
        feature_blocks_tp={
            "N": pair.n_tp,
            "X": stage_tp.expression_grn,
        },
        independent_width_feature_blocks=frozenset({"X"}),
        coords_t=pair_data.coords_t,
        coords_tp=pair_data.coords_tp,
        provenance={
            **provenance_base(request, pair_data),
            "feature_extractor": "Native_V7_N_plus_true_GRN_G",
            "uses_true_commot_cci": True,
            "uses_original_grn_edges_csv": True,
            "uses_true_grn": True,
            "network": "row_normalized_CCI_plus_true_GRN_G_cosine_knn",
            "network_mode": "cci_g_integrated",
            "grn_knn_k": int(request.grn_knn_k),
            "grn_graph_weight": float(request.grn_graph_weight),
            "encoder": "pairwise_zscore(concat(N,G))",
            "macro_feature_mode": "pool_expression_then_recompute_true_GRN_G",
            "macro_N_training_interface": "pool_spot_N_then_pairwise_zscore",
            "strict_posthoc_protocol": (
                "raw_and_rownorm_S_transpose_A_S_reextract_N_recompute_G"
            ),
            "N": pair.n_metadata,
            "Native_V7": pair.v7_metadata,
            "stage_t": stage_t.metadata,
            "stage_tp": stage_tp.metadata,
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
    return prepared
