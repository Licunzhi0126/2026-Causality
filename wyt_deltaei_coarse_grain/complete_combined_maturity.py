from __future__ import annotations

"""Complete-combined coarse preparation backed by registered network methods."""

from dataclasses import replace
from functools import partial

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.config import TemporalRunConfig
from mignet_ce.metrics import effective_information
from mignet_ce.networks.base import (
    CoarseTemporalNetworkPair,
    CoarseTemporalNetworkRequest,
    CoarseTemporalStageRequest,
    build_registered_coarse_temporal_pair,
)
from mignet_ce.networks.light_cci_grn import deterministic_projection_matrix
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
from mignet_ce.pij.wyt_single_kl import single_kl_pij_numpy, single_kl_pij_torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from mignet_ce.representations.wyt_network80 import joint_fixed_pca
from wyt_deltaei_coarse_grain.complete_combined import (
    CompleteCombinedStage,
    build_macro_pij_builder,
    pairwise_zscore,
    pool_features,
    prepare_complete_pair,
    project_macro_cci_raw,
    sparse_shared_core_directed_nmf,
    strict_complete_combined_evaluation,
)
from wyt_deltaei_coarse_grain.development import load_maturity_csv


def _network_config(request: CoarseFrontendRequest, network_method: str) -> TemporalRunConfig:
    return TemporalRunConfig(
        time_points=("t", "tp"),
        network_method=network_method,
        pij_method="compare_N_kl",
        cci_min=request.cci_min,
        grn_topk_targets=request.grn_topk_targets,
        grn_state_dim=request.grn_state_dim,
        grn_projection_seed=request.grn_projection_seed,
        grn_gate_mode="double_end",
        nmf_components=request.nmf_components,
        nmf_max_iter=request.nmf_max_iter,
        nmf_seed=request.seed,
        pij_temperature=request.pij_temperature,
    )


def _build_network_pair(
    request: CoarseFrontendRequest,
    *,
    network_method: str,
) -> CoarseTemporalNetworkPair:
    if request.cci_index_t is None or request.cci_index_tp is None:
        raise ValueError(
            "complete_combined maturity methods require explicit "
            "--cci-index-t and --cci-index-tp."
        )
    if network_method == "light_cci_grn" and (
        request.grn_t is None or request.grn_tp is None
    ):
        raise ValueError(
            "complete_combined_coarse_maturity_cci_grn requires --grn-t and --grn-tp."
        )
    network_request = CoarseTemporalNetworkRequest(
        stages=(
            CoarseTemporalStageRequest(
                time_point="t",
                h5ad=request.h5ad_t,
                cci_total=request.cci_t,
                cci_index=request.cci_index_t,
                grn_edges=request.grn_t if network_method == "light_cci_grn" else None,
            ),
            CoarseTemporalStageRequest(
                time_point="tp",
                h5ad=request.h5ad_tp,
                cci_total=request.cci_tp,
                cci_index=request.cci_index_tp,
                grn_edges=request.grn_tp if network_method == "light_cci_grn" else None,
            ),
        ),
        config=_network_config(request, network_method),
    )
    builder = get_network_builder(network_method)
    pair = build_registered_coarse_temporal_pair(
        builder,
        network_request,
        retain_joint_inputs=network_method == "light_cci_grn",
    )
    if pair.network_method != network_method:
        raise ValueError(
            f"Network builder returned {pair.network_method!r}; expected {network_method!r}."
        )
    for stage in (pair.source, pair.target):
        actual = stage.graph.metadata.get("network_method")
        if actual != network_method:
            raise ValueError(
                f"Network stage {stage.time_point} reports method {actual!r}; "
                f"expected {network_method!r}."
            )
    return pair


def _stage_adjacency(stage) -> sp.csr_matrix:
    stored = stage.graph.metadata.get("adjacency_csr")
    if stored is None:
        raise ValueError(
            f"{stage.graph.metadata.get('network_method')} did not return adjacency_csr."
        )
    matrix = stored.tocsr() if sp.issparse(stored) else sp.csr_matrix(stored)
    expected = len(stage.graph.units)
    if matrix.shape != (expected, expected):
        raise ValueError(
            f"Network adjacency {matrix.shape} does not match {expected} graph units."
        )
    return matrix.astype(np.float32)


def _attach_required_maturity(
    prepared: PreparedCoarseInput,
    request: CoarseFrontendRequest,
) -> PreparedCoarseInput:
    if request.maturity_t is None or request.maturity_tp is None:
        raise ValueError(
            f"{prepared.method} requires both --maturity-t and --maturity-tp."
        )
    source = load_maturity_csv(
        request.maturity_t,
        prepared.unit_ids_t,
        id_column=request.maturity_id_column,
        maturity_column=request.maturity_column,
        confidence_column=request.maturity_confidence_column,
        normalization=request.maturity_normalization,
        direction=request.maturity_direction,
    )
    target = load_maturity_csv(
        request.maturity_tp,
        prepared.unit_ids_tp,
        id_column=request.maturity_id_column,
        maturity_column=request.maturity_column,
        confidence_column=request.maturity_confidence_column,
        normalization=request.maturity_normalization,
        direction=request.maturity_direction,
    )
    attached = replace(
        prepared,
        maturity_t=source.values,
        maturity_tp=target.values,
        maturity_confidence_t=source.confidence,
        maturity_confidence_tp=target.confidence,
        maturity_provenance={"t": source.provenance, "tp": target.provenance},
    )
    attached.validate()
    return attached


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


def _cci_only_macro_pij(inputs: MacroPijInputs, *, temperature: float) -> torch.Tensor:
    n_t, n_tp = _pairwise_zscore_torch(
        inputs.feature_blocks_t["N"],
        inputs.feature_blocks_tp["N"],
    )
    return single_kl_pij_torch(n_t, n_tp, temperature=temperature)


def _evaluate_cci_only_macro(
    network_t: sp.csr_matrix,
    network_tp: sp.csr_matrix,
    micro_ei: float,
    micro_n_t: np.ndarray,
    micro_n_tp: np.ndarray,
    assignment_t: np.ndarray,
    assignment_tp: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    temperature: float,
) -> dict[str, object]:
    pooled_t, pooled_tp = pairwise_zscore(
        pool_features(micro_n_t, assignment_t),
        pool_features(micro_n_tp, assignment_tp),
    )
    _, pooled_pij = single_kl_pij_numpy(
        pooled_t,
        pooled_tp,
        temperature=temperature,
    )
    pooled_ei = float(effective_information(pooled_pij.copy()))

    strict_values: dict[str, float] = {}
    for label, normalize in (("raw", False), ("rownorm", True)):
        macro_t = project_macro_cci_raw(network_t, assignment_t)
        macro_tp = project_macro_cci_raw(network_tp, assignment_tp)
        if normalize:
            macro_t = row_normalize_sparse(macro_t)
            macro_tp = row_normalize_sparse(macro_tp)
        exact_t, exact_tp, _ = sparse_shared_core_directed_nmf(
            macro_t,
            macro_tp,
            components=nmf_components,
            max_iter=nmf_max_iter,
            seed=seed,
        )
        _, exact_pij = single_kl_pij_numpy(
            exact_t,
            exact_tp,
            temperature=temperature,
        )
        strict_values[label] = float(effective_information(exact_pij.copy()))
    return {
        "evaluation_protocol": "complete_combined_coarse_light_cci_n_only",
        "EI_micro_light_cci_N": float(micro_ei),
        "EI_macro_training_interface_pool_N": pooled_ei,
        "deltaEI_training_interface_pool_N": pooled_ei - float(micro_ei),
        "EI_macro_strict_raw_projected_CCI_reextract_N": strict_values["raw"],
        "deltaEI_strict_raw_projected_CCI_reextract_N": (
            strict_values["raw"] - float(micro_ei)
        ),
        "EI_macro_strict_rownorm_projected_CCI_reextract_N": strict_values["rownorm"],
        "deltaEI_strict_rownorm_projected_CCI_reextract_N": (
            strict_values["rownorm"] - float(micro_ei)
        ),
    }


def _prepare_cci_only(
    request: CoarseFrontendRequest,
    *,
    method: str,
) -> PreparedCoarseInput:
    network_pair = _build_network_pair(request, network_method="light_cci")
    source = network_pair.source
    target = network_pair.target
    cci_t = _stage_adjacency(source)
    cci_tp = _stage_adjacency(target)
    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        cci_t,
        cci_tp,
        components=request.nmf_components,
        max_iter=request.nmf_max_iter,
        seed=request.seed,
    )
    _, micro_pij = single_kl_pij_numpy(
        n_t,
        n_tp,
        temperature=request.pij_temperature,
    )
    micro_t, micro_tp = joint_fixed_pca(n_t, n_tp, output_dim=request.mid_dim)
    micro_ei = float(effective_information(micro_pij.copy()))
    prepared = PreparedCoarseInput(
        method=method,
        unit_ids_t=list(map(str, source.graph.units)),
        unit_ids_tp=list(map(str, target.graph.units)),
        network_t=row_normalize_sparse(cci_t),
        network_tp=row_normalize_sparse(cci_tp),
        encoder_features_t=n_t.astype(np.float32),
        encoder_features_tp=n_tp.astype(np.float32),
        micro_features_t=micro_t.astype(np.float32),
        micro_features_tp=micro_tp.astype(np.float32),
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=micro_ei,
        macro_pij_builder=partial(
            _cci_only_macro_pij,
            temperature=request.pij_temperature,
        ),
        feature_blocks_t={"N": n_t.astype(np.float32)},
        feature_blocks_tp={"N": n_tp.astype(np.float32)},
        coords_t=source.coords,
        coords_tp=target.coords,
        provenance={
            **network_pair.metadata,
            "coarse_method": method,
            "source_network_method": "light_cci",
            "uses_cci": True,
            "uses_grn": False,
            "coarse_core": "complete_combined_N_only",
            "network_adjacency_policy": "exact_registered_network_output",
            "N": n_metadata,
            "strict_posthoc_filename": "strict_complete_combined_maturity_evaluation.json",
        },
        posthoc_evaluator=partial(
            _evaluate_cci_only_macro,
            cci_t,
            cci_tp,
            micro_ei,
            n_t,
            n_tp,
            nmf_components=request.nmf_components,
            nmf_max_iter=request.nmf_max_iter,
            seed=request.seed,
            temperature=request.pij_temperature,
        ),
    )
    prepared.validate()
    return _attach_required_maturity(prepared, request)


def _complete_stage_from_registered_network(stage, request: CoarseFrontendRequest):
    metadata = stage.graph.metadata
    required = {
        "grn_state_csr",
        "grn_genes",
        "grn_adjacency_csr",
        "grn_expression_csr",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(
            f"light_cci_grn coarse payload is missing fields: {missing}."
        )
    genes = list(map(str, metadata["grn_genes"]))
    expression = sp.csr_matrix(metadata["grn_expression_csr"]).toarray().astype(np.float32)
    adjacency = sp.csr_matrix(metadata["grn_adjacency_csr"]).astype(np.float32)
    projection_reg = deterministic_projection_matrix(
        genes,
        role="reg",
        output_dim=request.grn_state_dim,
        seed=request.grn_projection_seed,
    ).astype(np.float32)
    projection_tar = deterministic_projection_matrix(
        genes,
        role="tar",
        output_dim=request.grn_state_dim,
        seed=request.grn_projection_seed,
    ).astype(np.float32)
    return CompleteCombinedStage(
        units=list(map(str, stage.graph.units)),
        cci=_stage_adjacency(stage),
        grn_genes=genes,
        expression_grn=expression,
        grn_adjacency=adjacency,
        projection_reg=projection_reg,
        projection_tar=projection_tar,
        g_raw=sp.csr_matrix(metadata["grn_state_csr"]).toarray().astype(np.float32),
        metadata={
            "source_network_method": "light_cci_grn",
            "network_builder_class": "LightCCIGRNNetworkBuilder",
            "network_stage_metadata": {
                key: value
                for key, value in metadata.items()
                if key
                not in {
                    "adjacency_csr",
                    "grn_state_csr",
                    "grn_adjacency_csr",
                    "grn_expression_csr",
                }
            },
        },
    )


def _prepare_cci_grn(
    request: CoarseFrontendRequest,
    *,
    method: str,
) -> PreparedCoarseInput:
    network_pair = _build_network_pair(request, network_method="light_cci_grn")
    stage_t = _complete_stage_from_registered_network(network_pair.source, request)
    stage_tp = _complete_stage_from_registered_network(network_pair.target, request)
    pair = prepare_complete_pair(
        stage_t,
        stage_tp,
        nmf_components=request.nmf_components,
        nmf_max_iter=request.nmf_max_iter,
        seed=request.seed,
        mid_dim=request.mid_dim,
    )
    prepared = PreparedCoarseInput(
        method=method,
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
        coords_t=network_pair.source.coords,
        coords_tp=network_pair.target.coords,
        provenance={
            **network_pair.metadata,
            "coarse_method": method,
            "source_network_method": "light_cci_grn",
            "uses_cci": True,
            "uses_grn": True,
            "coarse_core": "complete_combined_N_plus_G",
            "network_adjacency_policy": "exact_registered_network_output",
            "macro_feature_mode": "pool_expression_then_recompute_GRN_G",
            "N": pair.n_metadata,
            "Canonical_NG": pair.canonical_ng_metadata,
            "stage_t": stage_t.metadata,
            "stage_tp": stage_tp.metadata,
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


def prepare_complete_combined_maturity(
    request: CoarseFrontendRequest,
    *,
    method: str,
    network_method: str,
) -> PreparedCoarseInput:
    request.validate()
    if network_method == "light_cci":
        return _prepare_cci_only(request, method=method)
    if network_method == "light_cci_grn":
        return _prepare_cci_grn(request, method=method)
    raise ValueError(
        "complete-combined maturity methods support only light_cci and light_cci_grn."
    )
