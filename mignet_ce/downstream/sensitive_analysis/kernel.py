from __future__ import annotations

"""Temporary compatibility adapter for the new sensitivity module.

IMPORTANT FOR INTEGRATION:
This file currently mirrors the targeted canonical N/G equation so the module can
be reviewed against the 2026-09-20 tree before the production refactor. After
`mignet_ce.pij.compare._shared.ng_kl_ot` is upgraded, this module must delegate
to that shared implementation rather than retaining duplicate math.
"""

import numpy as np

from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.distances import robust_normalize_cost, summarize_dense_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare._shared.log_balanced_ot import balance_cost_log_sinkhorn
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn

EPS = 1e-12


def robust_span(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot estimate a robust span from an empty/non-finite matrix.")
    q_low = float(np.percentile(finite, lower))
    q_high = float(np.percentile(finite, upper))
    span = q_high - q_low
    if not np.isfinite(span) or span <= EPS:
        span = float(finite.max() - finite.min())
    if not np.isfinite(span) or span <= EPS:
        span = 1.0
    return q_low, q_high, span


def build_component_costs(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta_n: float,
    beta_g: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compute raw N/G KL then put both components on the same Robust5-95 scale."""
    d_n_raw = pairwise_feature_kl(n_source, n_target, beta=float(beta_n))
    d_g_raw = pairwise_feature_kl(g_source, g_target, beta=float(beta_g))
    if d_n_raw.shape != d_g_raw.shape:
        raise ValueError(f"N/G cost shape mismatch: {d_n_raw.shape} vs {d_g_raw.shape}")
    d_n, n_norm = robust_normalize_cost(d_n_raw, copy=True)
    d_g, g_norm = robust_normalize_cost(d_g_raw, copy=True)
    return d_n, d_g, {
        "component_normalization": "independent_robust_5_95",
        "d_n_raw": summarize_dense_cost(d_n_raw),
        "d_g_raw": summarize_dense_cost(d_g_raw),
        "d_n_normalization": n_norm,
        "d_g_normalization": g_norm,
        "d_n_normalized": summarize_dense_cost(d_n),
        "d_g_normalized": summarize_dense_cost(d_g),
    }


def mix_cost(
    d_n: np.ndarray,
    d_g: np.ndarray,
    *,
    alpha_cci: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Separate relative biological role (alpha) from global transition sharpness.

    alpha_cci=0 -> pure GRN; alpha_cci=1 -> pure CCI.
    Both components are already on the same Robust5-95 scale.  After mixing,
    divide by the mixed matrix's own Robust5-95 span WITHOUT clipping.  This
    prevents alpha from changing the global cost scale; tau remains the sole
    sharpness parameter.
    """
    alpha = float(alpha_cci)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha_cci must lie in [0, 1].")
    g_term = (1.0 - alpha) * np.asarray(d_g, dtype=float)
    n_term = alpha * np.asarray(d_n, dtype=float)
    mixed = g_term + n_term
    q05, q95, span = robust_span(mixed)
    controlled = mixed / span
    denominator = float(np.mean(mixed))
    g_share = float(np.mean(g_term) / denominator) if denominator > EPS else float("nan")
    return controlled, {
        "alpha_cci": alpha,
        "nominal_grn_weight": 1.0 - alpha,
        "nominal_cci_weight": alpha,
        "actual_grn_mean_cost_share": g_share,
        "mixed_q05": q05,
        "mixed_q95": q95,
        "mixed_robust_span": span,
        "combined_scale_control": "divide_by_mixed_q95_minus_q05_no_clipping",
    }


def balanced_pij_from_cost(cost: np.ndarray, *, tau: float) -> tuple[np.ndarray, dict[str, object]]:
    kernel, prebalanced = row_normalized_kernel_from_cost(cost, tau=float(tau))
    try:
        joint, pij, sinkhorn = balance_kernel_sinkhorn(kernel)
        sinkhorn = {**sinkhorn, "log_domain_fallback_used": False}
    except RuntimeError as error:
        joint, pij, sinkhorn = balance_cost_log_sinkhorn(np.asarray(cost, dtype=float) / float(tau))
        sinkhorn = {
            **sinkhorn,
            "log_domain_fallback_used": True,
            "standard_sinkhorn_error": str(error),
        }
    return pij, {
        "tau": float(tau),
        "prebalanced_row_sum_min": float(prebalanced.sum(axis=1).min()),
        "prebalanced_row_sum_max": float(prebalanced.sum(axis=1).max()),
        "sinkhorn": sinkhorn,
    }
