from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mignet_ce.pij.compare._shared.distances import summarize_dense_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl

from .config import AblationConfig
from .features import PairFeatureBlocks


@dataclass(frozen=True)
class AblationCostResult:
    cost: np.ndarray
    metadata: dict[str, object]


def _raw_kl(source: np.ndarray, target: np.ndarray, beta: float) -> np.ndarray:
    return pairwise_feature_kl(np.asarray(source, float), np.asarray(target, float), beta=float(beta))


def build_controlled_cost(blocks: PairFeatureBlocks, cfg: AblationConfig) -> AblationCostResult:
    """Build the controlled raw-KL cost used by every ablation row.

    N/G and L/G fusion uses the same full-model contract:
        C = (1-alpha_CCI) * D_G + alpha_CCI * D_CCI
    where both terms are raw KL costs with no per-matrix normalization or clipping.
    """

    cfg.validate()
    cci_cost: np.ndarray | None = None
    grn_cost: np.ndarray | None = None
    cci_beta: float | None = None
    if blocks.cci_source is not None:
        cci_beta = cfg.beta_n if blocks.cci_representation == "N" else cfg.beta_l
        cci_cost = _raw_kl(blocks.cci_source, blocks.cci_target, cci_beta)
    if blocks.grn_source is not None:
        grn_cost = _raw_kl(blocks.grn_source, blocks.grn_target, cfg.beta_g)

    if cci_cost is None and grn_cost is None:
        raise ValueError("At least one of CCI or GRN must be present.")
    if cci_cost is not None and grn_cost is not None and cci_cost.shape != grn_cost.shape:
        raise ValueError(f"CCI/GRN KL cost shapes differ: {cci_cost.shape} vs {grn_cost.shape}.")

    if cci_cost is None:
        cost = grn_cost.copy()
        fusion = "G_only_raw_KL"
        nominal_cci_weight = 0.0
        nominal_grn_weight = 1.0
    elif grn_cost is None:
        cost = cci_cost.copy()
        fusion = f"{blocks.cci_representation}_only_raw_KL"
        nominal_cci_weight = 1.0
        nominal_grn_weight = 0.0
    else:
        alpha = float(cfg.alpha_cci)
        cost = (1.0 - alpha) * grn_cost + alpha * cci_cost
        fusion = f"{blocks.cci_representation}G_raw_KL_convex_fusion"
        nominal_cci_weight = alpha
        nominal_grn_weight = 1.0 - alpha

    if not np.isfinite(cost).all() or np.any(cost < 0.0):
        raise ValueError("Controlled ablation cost must be finite and nonnegative.")

    metadata = {
        "fusion_mode": fusion,
        "component_normalization": "none",
        "combined_cost_clipping": False,
        "cci_representation": blocks.cci_representation,
        "beta_cci": cci_beta,
        "beta_g": float(cfg.beta_g) if grn_cost is not None else None,
        "alpha_cci": float(cfg.alpha_cci) if cci_cost is not None and grn_cost is not None else None,
        "nominal_cci_weight": nominal_cci_weight,
        "nominal_grn_weight": nominal_grn_weight,
        "CCI_raw_KL": summarize_dense_cost(cci_cost) if cci_cost is not None else None,
        "GRN_raw_KL": summarize_dense_cost(grn_cost) if grn_cost is not None else None,
        "combined_cost": summarize_dense_cost(cost),
    }
    return AblationCostResult(cost=cost, metadata=metadata)
