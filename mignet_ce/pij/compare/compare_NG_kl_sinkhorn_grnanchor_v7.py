from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary, row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.features import CompareFeatureSet, build_compare_feature_set
from mignet_ce.pij.compare.common import export_compare_pair_artifacts
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    FIXED_KERNEL_TEMPERATURE,
    N_CORRECTION_WEIGHT,
    build_grnanchored_kl_cost,
)
from mignet_ce.utils.matrix import safe_row_normalize


SINKHORN_MAX_ITERATIONS = 2_000
SINKHORN_TOLERANCE = 1.0e-9
SINKHORN_CHECK_EVERY = 10
SINKHORN_DUAL_MAX_ITERATIONS = 1_000


def _entropy_bits(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def _refine_sinkhorn_scaling_in_dual(
    base: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    source_scale: np.ndarray,
    target_scale: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Accelerate an ill-conditioned Sinkhorn fixed point in the same OT dual."""
    source_count, target_count = base.shape
    source_log_scale = np.log(source_scale)
    target_log_scale = np.log(target_scale)

    # Fix the additive dual gauge by setting the final target potential to zero.
    gauge = float(target_log_scale[-1])
    source_log_scale = source_log_scale + gauge
    target_log_scale = target_log_scale - gauge
    initial = np.concatenate([source_log_scale, target_log_scale[:-1]])

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        source_potential = parameters[:source_count]
        target_potential = np.concatenate([parameters[source_count:], np.zeros(1, dtype=float)])
        with np.errstate(over="ignore", invalid="ignore"):
            current_source_scale = np.exp(source_potential)
            current_target_scale = np.exp(target_potential)
            kernel_times_target = base @ current_target_scale
            kernel_transpose_times_source = base.T @ current_source_scale
            source_mass = current_source_scale * kernel_times_target
            target_mass = current_target_scale * kernel_transpose_times_source
            objective = float(
                np.sum(source_mass)
                - source_marginal @ source_potential
                - target_marginal @ target_potential
            )
        gradient = np.concatenate(
            [source_mass - source_marginal, (target_mass - target_marginal)[:-1]]
        )
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            return float(np.finfo(float).max), np.full_like(parameters, 1.0e100)
        return objective, gradient

    optimized = minimize(
        objective_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 1.0e-30,
            "gtol": min(float(tolerance), 1.0e-12),
            "maxiter": SINKHORN_DUAL_MAX_ITERATIONS,
            "maxls": 100,
            "maxcor": 20,
        },
    )
    source_potential = optimized.x[:source_count]
    target_potential = np.concatenate([optimized.x[source_count:], np.zeros(1, dtype=float)])
    refined_source_scale = np.exp(source_potential)
    refined_target_scale = np.exp(target_potential)
    current_source = refined_source_scale * (base @ refined_target_scale)
    current_target = refined_target_scale * (base.T @ refined_source_scale)
    residual = float(
        max(
            np.max(np.abs(current_source - source_marginal)),
            np.max(np.abs(current_target - target_marginal)),
        )
    )
    return refined_source_scale, refined_target_scale, {
        "used": True,
        "optimizer": "L-BFGS-B_on_balanced_entropic_OT_dual",
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "iterations": int(optimized.nit),
        "function_evaluations": int(optimized.nfev),
        "residual": residual,
    }


def balance_kernel_sinkhorn(
    kernel: np.ndarray,
    *,
    max_iterations: int = SINKHORN_MAX_ITERATIONS,
    tolerance: float = SINKHORN_TOLERANCE,
    check_every: int = SINKHORN_CHECK_EVERY,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Balance a positive rectangular kernel and return its conditional transition matrix.

    The joint coupling has uniform source and target marginals. The returned
    conditional matrix is the joint coupling divided by the uniform source
    marginal, so every row sums to one.
    """
    values = np.asarray(kernel, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Sinkhorn kernel must be a non-empty 2D matrix; got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Sinkhorn kernel must be finite and nonnegative.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")
    if int(check_every) < 1:
        raise ValueError("check_every must be positive.")

    row_sums = values.sum(axis=1)
    column_sums = values.sum(axis=0)
    if np.any(row_sums <= 0.0) or np.any(column_sums <= 0.0):
        raise ValueError("Balanced Sinkhorn requires positive support in every row and column.")

    # Positive row scaling does not change the balanced OT solution. Starting
    # from the V5 row-stochastic kernel improves numerical conditioning while
    # keeping exactly the same cost-induced coupling.
    base = safe_row_normalize(values)
    source_count, target_count = base.shape
    source_marginal = np.full(source_count, 1.0 / source_count, dtype=float)
    target_marginal = np.full(target_count, 1.0 / target_count, dtype=float)
    target_scale = np.ones(target_count, dtype=float)

    converged = False
    marginal_residual = float("inf")
    source_scale = np.ones(source_count, dtype=float)
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        kernel_times_target = base @ target_scale
        if not np.isfinite(kernel_times_target).all() or np.any(kernel_times_target <= 0.0):
            raise RuntimeError("Sinkhorn source scaling became non-finite or unsupported.")
        source_scale = source_marginal / kernel_times_target

        kernel_transpose_times_source = base.T @ source_scale
        if not np.isfinite(kernel_transpose_times_source).all() or np.any(
            kernel_transpose_times_source <= 0.0
        ):
            raise RuntimeError("Sinkhorn target scaling became non-finite or unsupported.")
        target_scale = target_marginal / kernel_transpose_times_source
        if not np.isfinite(source_scale).all() or not np.isfinite(target_scale).all():
            raise RuntimeError("Sinkhorn scaling vectors became non-finite.")

        if (iterations - 1) % int(check_every) == 0 or iterations == int(max_iterations):
            current_source = source_scale * (base @ target_scale)
            current_target = target_scale * (base.T @ source_scale)
            marginal_residual = float(
                max(
                    np.max(np.abs(current_source - source_marginal)),
                    np.max(np.abs(current_target - target_marginal)),
                )
            )
            if marginal_residual <= float(tolerance):
                converged = True
                break

    dual_fallback: dict[str, object] = {"used": False}
    if not converged:
        source_scale, target_scale, dual_fallback = _refine_sinkhorn_scaling_in_dual(
            base,
            source_marginal,
            target_marginal,
            source_scale,
            target_scale,
            tolerance=float(tolerance),
        )
        marginal_residual = float(dual_fallback["residual"])
        converged = marginal_residual <= float(tolerance)
    if not converged:
        raise RuntimeError(
            "Balanced Sinkhorn and dual refinement did not converge; "
            f"scaling_iterations={int(max_iterations)}, residual={marginal_residual:.6g}."
        )

    joint = (source_scale[:, None] * base) * target_scale[None, :]
    conditional = safe_row_normalize(joint)
    final_target_marginal = conditional.mean(axis=0)
    source_residual = float(np.max(np.abs(joint.sum(axis=1) - source_marginal)))
    target_residual = float(np.max(np.abs(final_target_marginal - target_marginal)))
    if not np.isfinite(joint).all() or not np.isfinite(conditional).all():
        raise RuntimeError("Balanced Sinkhorn produced non-finite output.")

    prebalanced_target_marginal = base.mean(axis=0)
    metadata: dict[str, object] = {
        "mode": "balanced_entropic_ot_uniform_marginals",
        "converged": True,
        "iterations": int(iterations),
        "scaling_iterations": int(iterations),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "check_every": int(check_every),
        "source_marginal_policy": "uniform",
        "target_marginal_policy": "uniform",
        "source_marginal_value": float(source_marginal[0]),
        "target_marginal_value": float(target_marginal[0]),
        "max_absolute_marginal_residual": float(max(source_residual, target_residual)),
        "source_marginal_residual": source_residual,
        "target_marginal_residual": target_residual,
        "prebalanced_target_entropy_bits": _entropy_bits(prebalanced_target_marginal),
        "balanced_target_entropy_bits": _entropy_bits(final_target_marginal),
        "maximum_target_entropy_bits": float(np.log2(target_count)),
        "mean_absolute_target_marginal_shift": float(
            np.mean(np.abs(final_target_marginal - prebalanced_target_marginal))
        ),
        "dual_fallback": dual_fallback,
        "uses_ei_for_fitting": False,
        "uses_layer_identity": False,
        "uses_labels": False,
        "uses_third_timepoint": False,
    }
    return joint, conditional, metadata


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


class CompareNGKlSinkhornGRNAnchorV7PijMethod:
    """Frozen V5 cost followed by balanced entropic optimal transport."""

    name = "compare_NG_kl_sinkhorn_grnanchor_v7"
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
                "algorithm_version": "lightcci_grn_sinkhorn_grnanchor_v7",
                "cost_is_exact_frozen_v5_formula": True,
                "uses_frozen_compare_N_kl_feature_path": True,
                "legacy_kl_block_weight_n_received_but_not_used": float(weight_n),
                "legacy_kl_block_weight_g_received_but_not_used": float(weight_g),
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
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
        joint, balanced_pij, sinkhorn_metadata = balance_kernel_sinkhorn(kernel)
        diagnostics = {
            "kind": "balanced_sinkhorn_grnanchored_block_feature_kl_kernel",
            "beta": FIXED_FEATURE_BETA,
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

        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "frozen_v5_cost_balanced_sinkhorn",
                "transition_construction": "balanced_sinkhorn_grnanchored_block_kl",
                "cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
                "fixed_feature_beta": FIXED_FEATURE_BETA,
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                "sinkhorn_max_iterations": SINKHORN_MAX_ITERATIONS,
                "sinkhorn_tolerance": SINKHORN_TOLERANCE,
                "source_marginal_policy": "uniform",
                "target_marginal_policy": "uniform",
                "uses_frozen_compare_N_kl_feature_path": True,
                "cost_is_exact_frozen_v5_formula": True,
                "feature_metadata": feature_set.metadata,
                "row_stochastic": True,
                "balanced_target_marginal": True,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
                "uses_ei_for_fitting": False,
                "uses_layer_identity": False,
                "uses_labels": False,
                "uses_third_timepoint": False,
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
                kernels.kernel_metadata[pair_label][side] = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    "fusion_mode": "frozen_v5_cost_balanced_sinkhorn",
                    "transition_construction": "balanced_sinkhorn_grnanchored_block_kl",
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
                    "cost_is_exact_frozen_v5_formula": True,
                    "sinkhorn": sinkhorn_metadata,
                    "raw_matrix_semantics": "balanced_joint_coupling",
                    "row_normalized_matrix_semantics": "conditional_transition_probability",
                    "uses_only_current_pair_timepoints": True,
                    "uses_developmental_features": False,
                    "uses_ei_for_fitting": False,
                    "uses_layer_identity": False,
                    "uses_labels": False,
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
                            "fusion_mode": "frozen_v5_cost_balanced_sinkhorn",
                            "transition_construction": "balanced_sinkhorn_grnanchored_block_kl",
                            "cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
                            "fixed_feature_beta": FIXED_FEATURE_BETA,
                            "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                            "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                            "cost_is_exact_frozen_v5_formula": True,
                            "source_marginal_policy": "uniform",
                            "target_marginal_policy": "uniform",
                            "raw_matrix_semantics": "balanced_joint_coupling",
                            "row_normalized_matrix_semantics": "conditional_transition_probability",
                            "final_cost_clipped_to_unit_interval": False,
                            "uses_only_current_pair_timepoints": True,
                            "uses_third_timepoint": False,
                            "uses_developmental_features": False,
                            "uses_ei_for_fitting": False,
                            "uses_layer_identity": False,
                            "uses_labels": False,
                            "uses_lower_to_upper_projection": False,
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
                "representation": "lightcci_grn_sinkhorn_grnanchor_v7",
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "frozen_v5_cost_balanced_sinkhorn",
                "transition_construction": "balanced_sinkhorn_grnanchored_block_kl",
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
                "uses_frozen_compare_N_kl_feature_path": True,
                "cost_is_exact_frozen_v5_formula": True,
                "source_marginal_policy": "uniform",
                "target_marginal_policy": "uniform",
                "uses_third_timepoint": False,
                "uses_developmental_features": False,
                "uses_ei_for_fitting": False,
                "uses_layer_identity": False,
                "uses_labels": False,
                "heldout_split_observed": False,
            },
        )
        return result, kernels
