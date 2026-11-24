"""Direct GRN-state frontend for perturbation experiments.

This adapter deliberately reuses the registered CCI/N preparation and global
two-stage objective.  The only changed input is the GRN state: spot states are
computed from the actual graph and macro expression with propagated residuals.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from mignet_ce.coarse_frontends._common import provenance_base
from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from .formal_macro_adapter import (
    build_direct_state_macro_pij_builder, PERTURBATION_MACRO_SCHEMA,
)
from .state_bridge import PerturbedGRNStates, make_request


def prepare_direct_state_two_stage(
    *, baseline_inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path,
    frontend_options: dict | None = None, states: PerturbedGRNStates | None = None,
) -> PreparedCoarseInput:
    """Prepare the formal two-stage input while preserving supplied GRN states."""
    from wyt_deltaei_coarse_grain.complete_combined import prepare_complete_pair
    from wyt_deltaei_coarse_grain.complete_combined_maturity import _attach_required_maturity
    from mignet_ce.coarse_frontends.formal_two_stage import prepare as formal_prepare

    request = make_request(baseline_inputs, frontend_options,
        maturity_t=maturity_t, maturity_tp=maturity_tp,
        maturity_id_column="unit_id", maturity_column="pseudotime")
    if states is None:
        prepared = formal_prepare(request)
        return replace(prepared, provenance={**prepared.provenance,
            "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA})
    if request.grn_feature_method != "legacy":
        raise NotImplementedError("Perturbation propagation supports legacy GRN features only.")
    pair_data = states.pair_data
    stage_t = states.stage_t
    stage_tp_base = states.stage_tp_baseline
    stage_tp_spot_state = replace(stage_tp_base, g_raw=states.g_raw_tp_perturbed)
    delta_g_raw_tp = states.delta_g_raw_tp
    pair = prepare_complete_pair(stage_t, stage_tp_spot_state,
        nmf_components=request.nmf_components, nmf_max_iter=request.nmf_max_iter,
        seed=request.seed, mid_dim=request.mid_dim,
        cci_feature_method=request.cci_feature_method,
        cci_laplacian_components=request.cci_laplacian_components,
        cci_laplacian_normalized=request.cci_laplacian_normalized, alpha_cci=request.alpha_cci)
    blocks_t = {"N": pair.n_t, "X": stage_t.expression_grn}
    blocks_tp = {"N": pair.n_tp, "X": stage_tp_base.expression_grn}
    blocks_t["DG_TP"] = np.zeros_like(stage_t.g_raw)
    blocks_tp["DG_TP"] = delta_g_raw_tp
    prepared = PreparedCoarseInput(
        method="formal_two_stage", unit_ids_t=stage_t.units, unit_ids_tp=stage_tp_base.units,
        network_t=row_normalize_sparse(stage_t.cci), network_tp=row_normalize_sparse(stage_tp_base.cci),
        encoder_features_t=pair.encoder_t, encoder_features_tp=pair.encoder_tp,
        micro_features_t=pair.micro_features_t, micro_features_tp=pair.micro_features_tp,
        micro_pij=pair.micro_pij, micro_ei=pair.micro_ei,
        macro_pij_builder=build_direct_state_macro_pij_builder(stage_t, stage_tp_base, delta_g_raw_tp=delta_g_raw_tp, alpha_cci=request.alpha_cci),
        feature_blocks_t=blocks_t, feature_blocks_tp=blocks_tp,
        independent_width_feature_blocks=frozenset({"X"}),
        coords_t=pair_data.coords_t, coords_tp=pair_data.coords_tp,
        provenance={**provenance_base(request, pair_data), "coarse_method": "formal_two_stage", "loss_protocol": FORMAL_TWO_STAGE_PROTOCOL, "macro_feature_mode": "pool_expression_then_recompute_GRN_G", "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA, "grn_state_source": "perturbation_direct_state", "Canonical_NG": pair.canonical_ng_metadata},
    )
    prepared.validate()
    return _attach_required_maturity(prepared, request)
