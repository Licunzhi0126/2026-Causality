from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary, row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.distances import robust_normalize_cost, summarize_dense_cost
from mignet_ce.pij.compare._shared.features import CompareFeatureSet, build_compare_feature_set
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare.common import export_compare_pair_artifacts


FIXED_FEATURE_BETA = 0.05
FIXED_KERNEL_TEMPERATURE = 1.0
N_CORRECTION_WEIGHT = 0.25


def build_grnanchored_kl_cost(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep raw GRN KL dynamic range and use normalized N KL only as a correction."""
    beta = float(beta)
    n_correction_weight = float(n_correction_weight)
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if n_correction_weight < 0.0:
        raise ValueError("n_correction_weight must be nonnegative.")

    n_cost = pairwise_feature_kl(n_source, n_target, beta=beta)
    g_cost = pairwise_feature_kl(g_source, g_target, beta=beta)
    if n_cost.shape != g_cost.shape:
        raise ValueError(f"N and G KL cost shapes differ: {n_cost.shape} vs {g_cost.shape}.")

    normalized_n, n_normalization = robust_normalize_cost(n_cost, copy=True)
    combined = g_cost + n_correction_weight * normalized_n
    if not np.isfinite(combined).all() or np.any(combined < 0.0):
        raise ValueError("GRN-anchored KL cost must be finite and nonnegative.")

    return combined, {
        "mode": "raw_grn_kl_plus_bounded_n_correction",
        "beta_n": beta,
        "beta_g": beta,
        "n_correction_weight": n_correction_weight,
        "n_cost": summarize_dense_cost(n_cost),
        "g_cost": summarize_dense_cost(g_cost),
        "n_normalization": n_normalization,
        "combined_cost": summarize_dense_cost(combined),
        "grn_cost_scale": "raw_kl_nats",
        "n_correction_scale": f"robust_5_95_times_{n_correction_weight:g}",
        "final_cost_clipped_to_unit_interval": False,
        "removes_unit_interval_gibbs_ei_bound": True,
        "parameter_selection_split": "adjacent_time_pairs_development_only",
        "heldout_split_observed": False,
    }


def _select_pair_features(
    feature_set: CompareFeatureSet,
    side: str,
    pair: TimePair,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if side == "lower":
        timewise = feature_set.lower_features
        pairwise = feature_set.pairwise_lower_features
    elif side == "upper":
        timewise = feature_set.upper_features
        pairwise = feature_set.pairwise_upper_features
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    if pairwise is not None and pair in pairwise:
        source, target = pairwise[pair]
        return np.asarray(source, dtype=float), np.asarray(target, dtype=float), True
    return (
        np.asarray(timewise[pair[0]], dtype=float),
        np.asarray(timewise[pair[1]], dtype=float),
        False,
    )


class CompareNGKlGRNAnchorV5PijMethod:
    """Frozen compare_N_kl features with a versioned GRN-anchored cost mapping."""

    name = "compare_NG_kl_grnanchor_v5"
    feature_keys = ("N",)
    pij_key = "kl"

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
            raise ValueError(f"{self.name} requires the light_cci_grn GRN feature block.")
        if not np.isclose(float(beta), FIXED_FEATURE_BETA, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{self.name} fixes pij_entropy_epsilon={FIXED_FEATURE_BETA}; got {float(beta)}."
            )
        cost, metadata = build_grnanchored_kl_cost(
            source,
            target,
            grn_source,
            grn_target,
            beta=FIXED_FEATURE_BETA,
            n_correction_weight=N_CORRECTION_WEIGHT,
        )
        metadata.update(
            {
                "entry_method": self.name,
                "algorithm_version": "lightcci_grn_baseline_cost_v5",
                "uses_frozen_compare_N_kl_feature_path": True,
                "legacy_kl_block_weight_n_received_but_not_used": float(weight_n),
                "legacy_kl_block_weight_g_received_but_not_used": float(weight_g),
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
            }
        )
        return cost, metadata

    def _build_pair_kernel(self, *, source, target, cfg: TemporalRunConfig, grn_source=None, grn_target=None):
        if not np.isclose(
            float(cfg.pij_temperature),
            FIXED_KERNEL_TEMPERATURE,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{self.name} fixes pij_temperature={FIXED_KERNEL_TEMPERATURE}; "
                f"got {float(cfg.pij_temperature)}."
            )
        cost, block_metadata = self.build_kl_cost(
            np.asarray(source, dtype=float),
            np.asarray(target, dtype=float),
            beta=float(cfg.pij_entropy_epsilon),
            weight_n=float(cfg.kl_block_weight_n),
            weight_g=float(cfg.kl_block_weight_g),
            grn_source=None if grn_source is None else np.asarray(grn_source, dtype=float),
            grn_target=None if grn_target is None else np.asarray(grn_target, dtype=float),
        )
        kernel, pij = row_normalized_kernel_from_cost(cost, tau=FIXED_KERNEL_TEMPERATURE)
        diagnostics = {
            "kind": "grnanchored_block_feature_kl_kernel",
            "beta": FIXED_FEATURE_BETA,
            "tau": FIXED_KERNEL_TEMPERATURE,
            "cost": matrix_summary(cost),
            "kernel": matrix_summary(kernel),
            "main_cost_dense": cost,
            "block_kl": block_metadata,
        }
        return sp.csr_matrix(kernel), sp.csr_matrix(pij), pij, diagnostics, None

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        feature_set = build_compare_feature_set(context, cfg, self.feature_keys)
        if not bool(feature_set.metadata.get("grn_block", {}).get("enabled", False)):
            raise ValueError(f"{self.name} requires an enabled light_cci_grn GRN block.")

        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "raw_grn_kl_plus_bounded_n_correction",
                "transition_construction": "grnanchored_block_kl",
                "cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
                "fixed_feature_beta": FIXED_FEATURE_BETA,
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                "uses_frozen_compare_N_kl_feature_path": True,
                "feature_metadata": feature_set.metadata,
                "row_stochastic": True,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
                "parameter_selection_split": "adjacent_time_pairs_development_only",
                "heldout_split_observed": False,
            }
        )
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_dict in (("lower", kernels.p_lower), ("upper", kernels.p_upper)):
                source, target, pairwise_used = _select_pair_features(feature_set, side, pair)
                grn_pairwise = (
                    feature_set.pairwise_lower_grn_features
                    if side == "lower"
                    else feature_set.pairwise_upper_grn_features
                )
                if grn_pairwise is None or pair not in grn_pairwise:
                    raise ValueError(f"{self.name} is missing {side} GRN features for time pair {pair_label}.")
                grn_source, grn_target = grn_pairwise[pair]
                raw_sparse, pij_sparse, dense_pij, diagnostics, _ = self._build_pair_kernel(
                    source=source,
                    target=target,
                    cfg=cfg,
                    grn_source=grn_source,
                    grn_target=grn_target,
                )
                target_dict[pair] = dense_pij
                block_metadata = diagnostics["block_kl"]
                kernels.kernel_metadata[pair_label][side] = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    "fusion_mode": "raw_grn_kl_plus_bounded_n_correction",
                    "transition_construction": "grnanchored_block_kl",
                    "cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
                    "feature_source": "pairwise_compare_features" if pairwise_used else "timewise_compare_features",
                    "pairwise_features_used": bool(pairwise_used),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "grn_block_used": True,
                    "grn_source_shape": list(np.asarray(grn_source).shape),
                    "grn_target_shape": list(np.asarray(grn_target).shape),
                    "final_cost_clipped_to_unit_interval": False,
                    "combined_cost": block_metadata["combined_cost"],
                }

                if should_export:
                    export_compare_pair_artifacts(
                        cfg=cfg,
                        context=context,
                        method_name=self.name,
                        feature_keys=self.feature_keys,
                        pij_key=self.pij_key,
                        feature_set=feature_set,
                        pair=pair,
                        side=side,
                        source_features=source,
                        target_features=target,
                        raw_sparse=raw_sparse,
                        pij_sparse=pij_sparse,
                        diagnostics={key: value for key, value in diagnostics.items() if key != "main_cost_dense"},
                        metadata_extra={
                            "fusion_mode": "raw_grn_kl_plus_bounded_n_correction",
                            "transition_construction": "grnanchored_block_kl",
                            "cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
                            "fixed_feature_beta": FIXED_FEATURE_BETA,
                            "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                            "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                            "final_cost_clipped_to_unit_interval": False,
                            "parameter_selection_split": "adjacent_time_pairs_development_only",
                            "heldout_split_observed": False,
                        },
                        grn_source_features=np.asarray(grn_source, dtype=float),
                        grn_target_features=np.asarray(grn_target, dtype=float),
                    )

        result = MethodResult(
            lower_features=feature_set.lower_features,
            upper_features=feature_set.upper_features,
            lower_coords=(
                context.lower_coords_by_time
                if context.feature_alignment_space == "native_units"
                else context.upper_coords_by_time
            ),
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=feature_set.pairwise_lower_features,
            pairwise_upper_features=feature_set.pairwise_upper_features,
            method_metadata={
                "pij_method": self.name,
                "representation": "lightcci_grn_baseline_cost_v5",
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "raw_grn_kl_plus_bounded_n_correction",
                "transition_construction": "grnanchored_block_kl",
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
                "uses_frozen_compare_N_kl_feature_path": True,
                "heldout_split_observed": False,
            },
        )
        return result, kernels
