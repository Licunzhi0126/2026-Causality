from __future__ import annotations

import numpy as np

from mignet_ce.pij.compare._shared.distances import robust_normalize_cost, summarize_dense_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare.common import ComparePijMethodBase


def build_block_kl_cost(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    weight_n: float = 0.5,
    weight_g: float = 0.5,
    beta_n: float = 1.0,
    beta_g: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    weight_n = float(weight_n)
    weight_g = float(weight_g)
    if weight_n < 0.0 or weight_g < 0.0:
        raise ValueError("weight_n and weight_g must be nonnegative.")
    if abs(weight_n + weight_g - 1.0) > 1e-9:
        raise ValueError("weight_n + weight_g must equal 1.")

    n_cost = pairwise_feature_kl(n_source, n_target, beta=beta_n)
    if weight_g == 0.0:
        return n_cost, {
            "mode": "n_only_exact_fallback",
            "weight_n": weight_n,
            "weight_g": weight_g,
            "beta_n": float(beta_n),
            "beta_g": float(beta_g),
            "n_cost": summarize_dense_cost(n_cost),
            "g_cost": None,
            "combined_cost": summarize_dense_cost(n_cost),
            "normalization": "none_to_preserve_original_compare_N_kl",
        }

    g_cost = pairwise_feature_kl(g_source, g_target, beta=beta_g)
    if n_cost.shape != g_cost.shape:
        raise ValueError(f"N and G KL cost shapes differ: {n_cost.shape} vs {g_cost.shape}.")
    if weight_n == 0.0:
        return g_cost, {
            "mode": "g_only_exact",
            "weight_n": weight_n,
            "weight_g": weight_g,
            "beta_n": float(beta_n),
            "beta_g": float(beta_g),
            "n_cost": summarize_dense_cost(n_cost),
            "g_cost": summarize_dense_cost(g_cost),
            "combined_cost": summarize_dense_cost(g_cost),
            "normalization": "none_for_single_active_block",
        }

    normalized_n, n_normalization = robust_normalize_cost(n_cost, copy=True)
    normalized_g, g_normalization = robust_normalize_cost(g_cost, copy=True)
    combined = weight_n * normalized_n + weight_g * normalized_g
    combined = np.maximum(np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return combined, {
        "mode": "weighted_independently_normalized_block_kl",
        "weight_n": weight_n,
        "weight_g": weight_g,
        "beta_n": float(beta_n),
        "beta_g": float(beta_g),
        "n_cost": summarize_dense_cost(n_cost),
        "g_cost": summarize_dense_cost(g_cost),
        "n_normalization": n_normalization,
        "g_normalization": g_normalization,
        "combined_cost": summarize_dense_cost(combined),
        "normalization": "independent_robust_5_95_per_block",
    }


class CompareNKlPijMethod(ComparePijMethodBase):
    name = "compare_N_kl"
    feature_keys = ("N",)
    pij_key = "kl"
    supports_block_kl = True

    def build_kl_cost(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        beta: float,
        weight_n: float,
        weight_g: float,
        grn_source: np.ndarray | None = None,
        grn_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object] | None]:
        if grn_source is None or grn_target is None:
            return super().build_kl_cost(
                source,
                target,
                beta=beta,
                weight_n=weight_n,
                weight_g=weight_g,
                grn_source=grn_source,
                grn_target=grn_target,
            )
        return build_block_kl_cost(
            source,
            target,
            grn_source,
            grn_target,
            weight_n=weight_n,
            weight_g=weight_g,
            beta_n=beta,
            beta_g=beta,
        )
