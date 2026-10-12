"""Direct GRN-state frontend for perturbation experiments.

This adapter deliberately reuses the registered CCI/N preparation and global
two-stage objective.  The only changed input is the GRN state: spot states are
provided by the perturbation pipeline and macro states are pooled directly.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest, load_spot_pair, provenance_base
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.complete_combined import (
    CANONICAL_ALPHA_CCI,
    CANONICAL_FEATURE_BETA_G,
    CANONICAL_FEATURE_BETA_N,
    CANONICAL_TEMPERATURE,
    canonical_ng_pij_torch,
    prepare_complete_pair,
    prepare_complete_stage,
)
from wyt_deltaei_coarse_grain.complete_combined_maturity import _attach_required_maturity


def _pairwise_zscore_torch(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([source, target], dim=0)
    std = values.std(dim=0, unbiased=False, keepdim=True)
    standardized = (values - values.mean(dim=0, keepdim=True)) / torch.where(std < 1e-8, torch.ones_like(std), std)
    return standardized[: source.shape[0]], standardized[source.shape[0] :]


def _pool(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    return assignment.T @ values / (assignment.sum(dim=0, keepdim=True).T + 1e-12)


def build_direct_state_macro_pij_builder():
    """Return the registered canonical N+G PIJ builder with pooled direct G."""
    def build(inputs: MacroPijInputs) -> torch.Tensor:
        n_t, n_tp = _pairwise_zscore_torch(inputs.feature_blocks_t["N"], inputs.feature_blocks_tp["N"])
        g_t, g_tp = _pairwise_zscore_torch(inputs.feature_blocks_t["G"], inputs.feature_blocks_tp["G"])
        return canonical_ng_pij_torch(
            n_t, n_tp, g_t, g_tp,
            beta_n=CANONICAL_FEATURE_BETA_N, beta_g=CANONICAL_FEATURE_BETA_G,
            alpha_cci=CANONICAL_ALPHA_CCI, temperature=CANONICAL_TEMPERATURE,
        )
    return build


def prepare_direct_state_two_stage(*, baseline_inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, g_raw_t: np.ndarray, g_raw_tp: np.ndarray, state_dim: int = 64, projection_seed: int = 20260713) -> PreparedCoarseInput:
    """Prepare the formal two-stage input while preserving supplied GRN states."""
    request = CoarseFrontendRequest(
        h5ad_t=baseline_inputs["h5ad_t"], h5ad_tp=baseline_inputs["h5ad_tp"],
        cci_t=baseline_inputs["cci_t"], cci_tp=baseline_inputs["cci_tp"],
        cci_index_t=baseline_inputs["index_t"], cci_index_tp=baseline_inputs["index_tp"],
        grn_t=baseline_inputs["grn_t"], grn_tp=baseline_inputs["grn_tp"],
        maturity_t=maturity_t, maturity_tp=maturity_tp,
        maturity_id_column="unit_id", maturity_column="pseudotime",
        grn_state_dim=state_dim, grn_projection_seed=projection_seed,
    )
    pair_data = load_spot_pair(request)
    stage_t = prepare_complete_stage(h5ad_path=request.h5ad_t, grn_path=request.grn_t, units=pair_data.unit_ids_t, cci=pair_data.cci_t, top_k_targets=request.grn_topk_targets, state_dim=state_dim, projection_seed=projection_seed)
    stage_tp = prepare_complete_stage(h5ad_path=request.h5ad_tp, grn_path=request.grn_tp, units=pair_data.unit_ids_tp, cci=pair_data.cci_tp, top_k_targets=request.grn_topk_targets, state_dim=state_dim, projection_seed=projection_seed)
    g_raw_t = np.asarray(g_raw_t, dtype=np.float32)
    g_raw_tp = np.asarray(g_raw_tp, dtype=np.float32)
    if g_raw_t.shape != stage_t.g_raw.shape or g_raw_tp.shape != stage_tp.g_raw.shape:
        raise ValueError(f"Direct GRN state shapes must match aligned spots: {g_raw_t.shape}/{g_raw_tp.shape} vs {stage_t.g_raw.shape}/{stage_tp.g_raw.shape}.")
    if not np.isfinite(g_raw_t).all() or not np.isfinite(g_raw_tp).all():
        raise ValueError("Direct GRN states must be finite.")
    stage_t, stage_tp = replace(stage_t, g_raw=g_raw_t), replace(stage_tp, g_raw=g_raw_tp)
    pair = prepare_complete_pair(stage_t, stage_tp, nmf_components=request.nmf_components, nmf_max_iter=request.nmf_max_iter, seed=request.seed, mid_dim=request.mid_dim)
    prepared = PreparedCoarseInput(
        method="maturity_cci_grn_two_stage", unit_ids_t=stage_t.units, unit_ids_tp=stage_tp.units,
        network_t=row_normalize_sparse(stage_t.cci), network_tp=row_normalize_sparse(stage_tp.cci),
        encoder_features_t=pair.encoder_t, encoder_features_tp=pair.encoder_tp,
        micro_features_t=pair.micro_features_t, micro_features_tp=pair.micro_features_tp,
        micro_pij=pair.micro_pij, micro_ei=pair.micro_ei,
        macro_pij_builder=build_direct_state_macro_pij_builder(),
        feature_blocks_t={"N": pair.n_t, "G": pair.g_t},
        feature_blocks_tp={"N": pair.n_tp, "G": pair.g_tp},
        coords_t=pair_data.coords_t, coords_tp=pair_data.coords_tp,
        provenance={**provenance_base(request, pair_data), "coarse_method": "maturity_cci_grn_two_stage", "macro_feature_mode": "pool_direct_grn_state", "grn_state_source": "perturbation_direct_state", "Canonical_NG": pair.canonical_ng_metadata},
    )
    prepared.validate()
    return _attach_required_maturity(prepared, request)
