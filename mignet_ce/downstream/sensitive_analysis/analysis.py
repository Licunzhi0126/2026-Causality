from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mignet_ce.downstream.analysis.preparation import _load_complete_stage
from mignet_ce.metrics import effective_information
from wyt_deltaei_coarse_grain.complete_combined import pairwise_zscore, sparse_shared_core_directed_nmf

from .config import DEFAULT_COMPARISONS, SensitivityConfig
from .kernel import balanced_pij_from_cost, build_component_costs, mix_cost


@dataclass(frozen=True)
class _LoaderConfig:
    data_root: object
    organ: str


def _layer_pair_ei_curve(cfg: SensitivityConfig, layer: str, pair: str) -> pd.DataFrame:
    source_time, target_time = pair.split("->")
    loader_cfg = _LoaderConfig(data_root=cfg.data_root, organ=cfg.organ)
    stage_t = _load_complete_stage(loader_cfg, layer, source_time)
    stage_tp = _load_complete_stage(loader_cfg, layer, target_time)

    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        stage_t.cci,
        stage_tp.cci,
        components=cfg.nmf_components,
        max_iter=cfg.nmf_max_iter,
        seed=cfg.random_seed,
    )
    g_t, g_tp = pairwise_zscore(stage_t.g_raw, stage_tp.g_raw)
    d_n, d_g, cost_metadata = build_component_costs(
        n_t,
        n_tp,
        g_t,
        g_tp,
        beta_n=cfg.beta_n,
        beta_g=cfg.beta_g,
    )

    rows: list[dict[str, Any]] = []
    for alpha in cfg.alphas:
        cost, mix_metadata = mix_cost(d_n, d_g, alpha_cci=alpha)
        pij, pij_metadata = balanced_pij_from_cost(cost, tau=cfg.tau)
        rows.append(
            {
                "organ": cfg.organ,
                "layer": layer,
                "time_pair": pair,
                "source_time": source_time,
                "target_time": target_time,
                "alpha": float(alpha),
                "tau": float(cfg.tau),
                "EI_bits": float(effective_information(pij.copy())),
                "actual_grn_cost_share": float(mix_metadata["actual_grn_mean_cost_share"]),
                "mixed_robust_span": float(mix_metadata["mixed_robust_span"]),
                "sinkhorn_residual": float(
                    pij_metadata["sinkhorn"].get(
                        "max_absolute_marginal_residual",
                        pij_metadata["sinkhorn"].get("marginal_residual", np.nan),
                    )
                ),
                "nmf_components": int(cfg.nmf_components),
                "nmf_max_iter": int(cfg.nmf_max_iter),
                "random_seed": int(cfg.random_seed),
                "n_nmf_mode": str(n_metadata.get("mode", "shared_core_directed_nmf")),
                "component_normalization": str(cost_metadata["component_normalization"]),
                "combined_scale_control": str(mix_metadata["combined_scale_control"]),
            }
        )
    return pd.DataFrame(rows)


def build_layer_ei_table(cfg: SensitivityConfig) -> pd.DataFrame:
    frames = [
        _layer_pair_ei_curve(cfg, layer, pair)
        for pair in cfg.time_pairs
        for layer in cfg.layers
    ]
    return pd.concat(frames, ignore_index=True)


def build_hierarchy_sensitivity(layer_ei: pd.DataFrame) -> pd.DataFrame:
    indexed = layer_ei.set_index(["time_pair", "alpha", "layer"])
    rows: list[dict[str, object]] = []
    time_pairs = list(dict.fromkeys(layer_ei["time_pair"].astype(str)))
    alphas = sorted(layer_ei["alpha"].astype(float).unique())
    for pair in time_pairs:
        for alpha in alphas:
            for lower, upper, label in DEFAULT_COMPARISONS:
                low = indexed.loc[(pair, alpha, lower)]
                high = indexed.loc[(pair, alpha, upper)]
                rows.append(
                    {
                        "hierarchy_pair": label,
                        "alpha": float(alpha),
                        "time_pair": pair,
                        "EI_lower_bits": float(low["EI_bits"]),
                        "EI_upper_bits": float(high["EI_bits"]),
                        "delta_EI_bits": float(high["EI_bits"] - low["EI_bits"]),
                        "GRN_share_lower": float(low["actual_grn_cost_share"]),
                        "GRN_share_upper": float(high["actual_grn_cost_share"]),
                        "tau": float(low["tau"]),
                    }
                )
    return pd.DataFrame(rows)


def build_requested_wide_table(
    sensitivity_long: pd.DataFrame,
    *,
    time_pair_order: tuple[str, ...],
) -> pd.DataFrame:
    wide = sensitivity_long.pivot_table(
        index=["hierarchy_pair", "alpha"],
        columns="time_pair",
        values="delta_EI_bits",
        aggfunc="first",
    ).reset_index()
    for pair in time_pair_order:
        if pair not in wide.columns:
            wide[pair] = np.nan
    return wide[["hierarchy_pair", "alpha", *time_pair_order]].sort_values(
        ["hierarchy_pair", "alpha"]
    ).reset_index(drop=True)
