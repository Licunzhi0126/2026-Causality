from __future__ import annotations

"""Experimental mean-optimal N/G KL-OT variant selected by the V7 ablation."""

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import (
    matrix_summary,
    row_normalized_kernel_from_cost,
)
from mignet_ce.pij.compare._shared.features import (
    CompareFeatureSet,
    build_compare_feature_set,
)
from mignet_ce.pij.compare._shared.log_balanced_ot import (
    balance_cost_log_sinkhorn,
)
from mignet_ce.pij.compare._shared.ng_kl_ot import build_ng_kl_cost_numpy
from mignet_ce.pij.compare.common import export_compare_pair_artifacts
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import (
    SINKHORN_MAX_ITERATIONS,
    SINKHORN_TOLERANCE,
    balance_kernel_sinkhorn,
)


FIXED_FEATURE_BETA_N = 0.05
FIXED_FEATURE_BETA_G = 0.05
FIXED_G_SCALE = 1.55
FIXED_N_WEIGHT = 0.05
FIXED_KERNEL_TEMPERATURE = 1.0
ABLATION_MEAN_DELTA_EI = 0.6304599603442601
ABLATION_MEDIAN_DELTA_EI = 0.7885116198512225
ABLATION_MIN_DELTA_EI = -0.08776963499604395
ABLATION_POSITIVE_COUNT = 8
ABLATION_TOTAL_COUNT = 9


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


def _experimental_metadata() -> dict[str, object]:
    return {
        "status": "experimental",
        "replaces_frozen_v7": False,
        "parameter_selection_dataset": (
            "heart_E11.5_E12.5_E13.5_E14.5_seurat_ablation_2026-07-28"
        ),
        "parameter_selection_metric": "maximum_mean_deltaEI",
        "parameter_selection_split": "same_heart_ablation_pairs; not an external holdout",
        "heldout_split_observed": False,
        "cross_organ_validation_completed": False,
        "ablation_mean_deltaEI": ABLATION_MEAN_DELTA_EI,
        "ablation_median_deltaEI": ABLATION_MEDIAN_DELTA_EI,
        "ablation_min_deltaEI": ABLATION_MIN_DELTA_EI,
        "ablation_positive_count": ABLATION_POSITIVE_COUNT,
        "ablation_total_count": ABLATION_TOTAL_COUNT,
        "effective_temperature_note": (
            "At fixed OT temperature 1.0, g_scale=1.55 sharpens the GRN "
            "contribution and therefore changes its effective OT temperature."
        ),
    }


class NGKLotPijMethod:
    """Balanced N/G KL-OT with the report-selected mean-optimal fixed weights."""

    name = "NG_KLot"
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
    ) -> tuple[np.ndarray, dict[str, object]]:
        if grn_source is None or grn_target is None:
            raise ValueError(f"{self.name} requires the light_cci_grn GRN feature block.")
        if not np.isclose(float(beta), FIXED_FEATURE_BETA_N, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{self.name} fixes pij_entropy_epsilon={FIXED_FEATURE_BETA_N}; "
                f"got {float(beta)}."
            )
        cost, metadata = build_ng_kl_cost_numpy(
            source,
            target,
            grn_source,
            grn_target,
            beta_n=FIXED_FEATURE_BETA_N,
            beta_g=FIXED_FEATURE_BETA_G,
            g_scale=FIXED_G_SCALE,
            n_weight=FIXED_N_WEIGHT,
        )
        metadata.update(
            {
                "entry_method": self.name,
                "algorithm_version": "NG_KLot_heart_mean_optimal_v1",
                "uses_frozen_compare_N_kl_feature_path": True,
                "legacy_kl_block_weight_n_received_but_not_used": float(weight_n),
                "legacy_kl_block_weight_g_received_but_not_used": float(weight_g),
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                **_experimental_metadata(),
            }
        )
        return cost, metadata

    def _build_pair_kernel(
        self,
        *,
        source,
        target,
        cfg: TemporalRunConfig,
        grn_source=None,
        grn_target=None,
    ):
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
        kernel, prebalanced_pij = row_normalized_kernel_from_cost(
            cost,
            tau=FIXED_KERNEL_TEMPERATURE,
        )
        try:
            joint, balanced_pij, sinkhorn_metadata = balance_kernel_sinkhorn(kernel)
            sinkhorn_metadata["log_domain_fallback_used"] = False
        except RuntimeError as error:
            joint, balanced_pij, sinkhorn_metadata = balance_cost_log_sinkhorn(
                cost / FIXED_KERNEL_TEMPERATURE
            )
            sinkhorn_metadata["log_domain_fallback_used"] = True
            sinkhorn_metadata["standard_sinkhorn_error"] = str(error)
        diagnostics = {
            "kind": "experimental_scaled_ng_kl_balanced_sinkhorn",
            "beta_n": FIXED_FEATURE_BETA_N,
            "beta_g": FIXED_FEATURE_BETA_G,
            "g_scale": FIXED_G_SCALE,
            "n_weight": FIXED_N_WEIGHT,
            "tau": FIXED_KERNEL_TEMPERATURE,
            "cost": matrix_summary(cost),
            "kernel": matrix_summary(kernel),
            "prebalanced_pij": matrix_summary(prebalanced_pij),
            "balanced_joint": matrix_summary(joint),
            "balanced_pij": matrix_summary(balanced_pij),
            "sinkhorn": sinkhorn_metadata,
            "main_cost_dense": cost,
            "block_kl": block_metadata,
        }
        return (
            sp.csr_matrix(joint),
            sp.csr_matrix(balanced_pij),
            balanced_pij,
            diagnostics,
        )

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        feature_set = build_compare_feature_set(context, cfg, self.feature_keys)
        if not bool(feature_set.metadata.get("grn_block", {}).get("enabled", False)):
            raise ValueError(f"{self.name} requires an enabled light_cci_grn GRN block.")

        common_metadata = {
            "pij_method": self.name,
            "compare_feature_keys": list(self.feature_keys),
            "compare_pij_method": self.pij_key,
            "fusion_mode": "scaled_raw_grn_kl_plus_bounded_n_correction",
            "transition_construction": "balanced_sinkhorn_scaled_ng_kl",
            "cost_source": "1.55_raw_GRN_KL_plus_0.05_robust_normalized_N_KL",
            "fixed_feature_beta_n": FIXED_FEATURE_BETA_N,
            "fixed_feature_beta_g": FIXED_FEATURE_BETA_G,
            "fixed_g_scale": FIXED_G_SCALE,
            "fixed_n_weight": FIXED_N_WEIGHT,
            "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
            "sinkhorn_max_iterations": SINKHORN_MAX_ITERATIONS,
            "sinkhorn_tolerance": SINKHORN_TOLERANCE,
            "source_marginal_policy": "uniform",
            "target_marginal_policy": "uniform",
            "uses_frozen_compare_N_kl_feature_path": True,
            "row_stochastic": True,
            "balanced_target_marginal": True,
            "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            "uses_ei_for_fitting": False,
            "uses_layer_identity": False,
            "uses_labels": False,
            "uses_third_timepoint": False,
            **_experimental_metadata(),
        }
        kernels = TransitionKernels(
            kernel_metadata={
                **common_metadata,
                "feature_metadata": feature_set.metadata,
            }
        )
        should_export = bool(
            cfg.export_pij
            or cfg.export_pair_artifacts
            or cfg.export_feature_diagnostics
        )

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
                    raise ValueError(
                        f"{self.name} is missing {side} GRN features for time pair "
                        f"{pair_label}."
                    )
                grn_source, grn_target = grn_pairwise[pair]
                raw_sparse, pij_sparse, dense_pij, diagnostics = self._build_pair_kernel(
                    source=source,
                    target=target,
                    cfg=cfg,
                    grn_source=grn_source,
                    grn_target=grn_target,
                )
                target_dict[pair] = dense_pij
                block_metadata = diagnostics["block_kl"]
                sinkhorn_metadata = diagnostics["sinkhorn"]
                pair_metadata = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    "fusion_mode": common_metadata["fusion_mode"],
                    "transition_construction": common_metadata["transition_construction"],
                    "cost_source": common_metadata["cost_source"],
                    "feature_source": (
                        "pairwise_compare_features"
                        if pairwise_used
                        else "timewise_compare_features"
                    ),
                    "pairwise_features_used": bool(pairwise_used),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "grn_block_used": True,
                    "grn_source_shape": list(np.asarray(grn_source).shape),
                    "grn_target_shape": list(np.asarray(grn_target).shape),
                    "final_cost_clipped_to_unit_interval": False,
                    "combined_cost": block_metadata["combined_cost"],
                    "sinkhorn": sinkhorn_metadata,
                    "raw_matrix_semantics": "balanced_joint_coupling",
                    "row_normalized_matrix_semantics": (
                        "conditional_transition_probability"
                    ),
                    "uses_only_current_pair_timepoints": True,
                    "uses_developmental_features": False,
                    **_experimental_metadata(),
                }
                kernels.kernel_metadata[pair_label][side] = pair_metadata

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
                        diagnostics={
                            key: value
                            for key, value in diagnostics.items()
                            if key != "main_cost_dense"
                        },
                        metadata_extra={
                            **common_metadata,
                            "raw_matrix_semantics": "balanced_joint_coupling",
                            "row_normalized_matrix_semantics": (
                                "conditional_transition_probability"
                            ),
                            "final_cost_clipped_to_unit_interval": False,
                            "uses_only_current_pair_timepoints": True,
                            "uses_developmental_features": False,
                            "uses_lower_to_upper_projection": False,
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
                **common_metadata,
                "representation": "NG_KLot_heart_mean_optimal_v1",
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
                "uses_developmental_features": False,
            },
        )
        return result, kernels
