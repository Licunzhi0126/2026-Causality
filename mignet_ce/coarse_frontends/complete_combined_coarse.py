from __future__ import annotations

"""Complete Native-V7 and WYT coarse-graining frontend."""

from functools import partial

import numpy as np
import torch

from mignet_ce.coarse_frontends._common import (
    CoarseFrontendRequest,
    load_spot_pair,
    provenance_base,
)
from mignet_ce.coarse_frontends._complete_combined_core import (
    CompleteCombinedStage,
    prepare_complete_pair,
    prepare_complete_stage,
    strict_complete_combined_evaluation,
)
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
)
from mignet_ce.pij.compare.native_v7_torch import native_v7_pij_torch
from mignet_ce.representations.coarse_input import (
    MacroPijInputs,
    PreparedCoarseInput,
)


def _pairwise_zscore_torch(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([source, target], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    values = (combined - mean) / std
    return values[: source.shape[0]], values[source.shape[0] :]


def _macro_pij_builder(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
):
    cache: dict[tuple[str, str], torch.Tensor] = {}

    def torch_assets(stage: CompleteCombinedStage, label: str, device: torch.device):
        device_key = str(device)
        adjacency_key = (f"{label}_adjacency", device_key)
        if adjacency_key not in cache:
            coo = stage.grn_adjacency.tocoo()
            indices = torch.tensor(
                np.vstack([coo.row, coo.col]),
                dtype=torch.long,
                device=device,
            )
            values = torch.tensor(coo.data, dtype=torch.float32, device=device)
            cache[adjacency_key] = torch.sparse_coo_tensor(
                indices,
                values,
                size=coo.shape,
                dtype=torch.float32,
                device=device,
                check_invariants=True,
            ).coalesce()
            cache[(f"{label}_projection_reg", device_key)] = torch.tensor(
                stage.projection_reg,
                dtype=torch.float32,
                device=device,
            )
            cache[(f"{label}_projection_tar", device_key)] = torch.tensor(
                stage.projection_tar,
                dtype=torch.float32,
                device=device,
            )
        return (
            cache[adjacency_key],
            cache[(f"{label}_projection_reg", device_key)],
            cache[(f"{label}_projection_tar", device_key)],
        )

    def project(
        expression: torch.Tensor,
        stage: CompleteCombinedStage,
        label: str,
    ) -> torch.Tensor:
        values = torch.clamp(
            torch.nan_to_num(expression, nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        adjacency, projection_reg, projection_tar = torch_assets(
            stage,
            label,
            values.device,
        )
        regulator_program = torch.sparse.mm(adjacency, values.T).T
        target_program = torch.sparse.mm(adjacency.transpose(0, 1), values.T).T
        return (
            (values * regulator_program) @ projection_reg
            + (values * target_program) @ projection_tar
        )

    def build(inputs: MacroPijInputs) -> torch.Tensor:
        n_t, n_tp = _pairwise_zscore_torch(
            inputs.feature_blocks_t["N"],
            inputs.feature_blocks_tp["N"],
        )
        g_t, g_tp = _pairwise_zscore_torch(
            project(inputs.feature_blocks_t["X"], stage_t, "t"),
            project(inputs.feature_blocks_tp["X"], stage_tp, "tp"),
        )
        return native_v7_pij_torch(n_t, n_tp, g_t, g_tp)

    return build


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
        macro_pij_builder=_macro_pij_builder(stage_t, stage_tp),
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
