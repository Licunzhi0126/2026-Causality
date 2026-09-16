from __future__ import annotations

"""NG_KLot preparation shared by natural caches and optimized DeltaEI inputs."""

from typing import Any

import numpy as np

from mignet_ce.metrics import effective_information
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.ng_kl_ot import build_ng_kl_cost_numpy
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn
from mignet_ce.representations.wyt_network80 import joint_fixed_pca
from wyt_deltaei_coarse_grain.complete_combined import (
    CompleteCombinedPair,
    CompleteCombinedStage,
    pairwise_zscore,
    sparse_shared_core_directed_nmf,
)

from ..metrics import row_normalize


NG_BETA_N = 0.05
NG_BETA_G = 0.05
NG_G_SCALE = 1.55
NG_N_WEIGHT = 0.05
NG_TEMPERATURE = 1.0


def ngklot_pij_numpy(
    n_t: np.ndarray,
    n_tp: np.ndarray,
    g_t: np.ndarray,
    g_tp: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    cost, metadata = build_ng_kl_cost_numpy(
        n_t,
        n_tp,
        g_t,
        g_tp,
        beta_n=NG_BETA_N,
        beta_g=NG_BETA_G,
        g_scale=NG_G_SCALE,
        n_weight=NG_N_WEIGHT,
    )
    kernel, prebalanced = row_normalized_kernel_from_cost(cost, tau=NG_TEMPERATURE)
    _joint, pij, sinkhorn = balance_kernel_sinkhorn(kernel)
    metadata = {
        **metadata,
        "pij_method": "NG_KLot",
        "network_method": "light_cci_grn",
        "prebalanced_ei": float(effective_information(prebalanced.copy())),
        "sinkhorn": sinkhorn,
    }
    return row_normalize(pij), metadata


def prepare_ngklot_pair(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    mid_dim: int = 32,
) -> CompleteCombinedPair:
    """Build the full N+G NG_KLot micro transition used by downstream analyses."""

    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        stage_t.cci,
        stage_tp.cci,
        components=nmf_components,
        max_iter=nmf_max_iter,
        seed=seed,
    )
    g_t, g_tp = pairwise_zscore(stage_t.g_raw, stage_tp.g_raw)
    micro_pij, ng_metadata = ngklot_pij_numpy(n_t, n_tp, g_t, g_tp)
    encoder_t, encoder_tp = pairwise_zscore(
        np.hstack([n_t, g_t]),
        np.hstack([n_tp, g_tp]),
    )
    micro_t, micro_tp = joint_fixed_pca(encoder_t, encoder_tp, output_dim=mid_dim)
    return CompleteCombinedPair(
        n_t=n_t,
        n_tp=n_tp,
        g_t=g_t,
        g_tp=g_tp,
        encoder_t=encoder_t,
        encoder_tp=encoder_tp,
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=float(effective_information(micro_pij.copy())),
        n_metadata=n_metadata,
        v7_metadata=ng_metadata,
    )

