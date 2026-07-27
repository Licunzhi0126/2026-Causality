from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary
from mignet_ce.pij.compare._shared.features import build_compare_feature_set
from mignet_ce.pij.compare._shared.lowrank_fgw import (
    FGW_FACTORIZATION_SEED,
    FGW_STRUCTURE_RANK,
)
from mignet_ce.pij.compare._shared.multiscale_fgw import (
    MULTISCALE_DIFFUSION_STEPS,
    MULTISCALE_STRUCTURE_WEIGHT,
    MULTISCALE_TEMPERATURE_SCHEDULE,
    solve_multiscale_directed_fgw,
)
from mignet_ce.pij.compare.common import export_compare_pair_artifacts
from mignet_ce.pij.compare.compare_NG_fgw_grnanchor_v9 import (
    _select_pair_adjacencies,
    _select_pair_features,
)
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    FIXED_KERNEL_TEMPERATURE,
    N_CORRECTION_WEIGHT,
    build_grnanchored_kl_cost,
)


class CompareNGMultiscaleFGWAnnealedV10PijMethod:
    """Frozen V5 node cost with fixed multi-scale directed FGW continuation."""

    name = "compare_NG_multiscale_fgw_annealed_v10"
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
                "algorithm_version": "lightcci_grn_multiscale_directed_fgw_annealed_v10",
                "node_cost_is_exact_frozen_v5_formula": True,
                "uses_frozen_compare_N_kl_feature_path": True,
                "legacy_kl_block_weight_n_received_but_not_used": float(weight_n),
                "legacy_kl_block_weight_g_received_but_not_used": float(weight_g),
                "fixed_entry_temperature": FIXED_KERNEL_TEMPERATURE,
                "fixed_internal_temperature_schedule": list(
                    MULTISCALE_TEMPERATURE_SCHEDULE
                ),
            }
        )
        return cost, metadata

    def _build_pair_kernel(
        self,
        *,
        source,
        target,
        cfg: TemporalRunConfig,
        source_adjacency,
        target_adjacency,
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
        joint, pij, multiscale_metadata = solve_multiscale_directed_fgw(
            cost,
            source_adjacency,
            target_adjacency,
        )
        diagnostics = {
            "kind": "multiscale_lowrank_directed_fgw_annealed_grnanchored_block_kl",
            "beta": FIXED_FEATURE_BETA,
            "entry_temperature": FIXED_KERNEL_TEMPERATURE,
            "final_temperature": MULTISCALE_TEMPERATURE_SCHEDULE[-1],
            "node_cost": matrix_summary(cost),
            "balanced_joint": matrix_summary(joint),
            "balanced_pij": matrix_summary(pij),
            "multiscale_fgw": multiscale_metadata,
            "main_cost_dense": cost,
            "block_kl": block_metadata,
        }
        return sp.csr_matrix(joint), sp.csr_matrix(pij), pij, diagnostics

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
            "fusion_mode": "frozen_v5_node_cost_plus_multiscale_directed_fgw_annealing",
            "transition_construction": "fixed_schedule_multiscale_lowrank_directed_fgw",
            "node_cost_source": "raw_GRN_KL_plus_0.25_robust_normalized_N_KL",
            "graph_cost_source": "current_pair_LightCCI_directed_diffusion_steps_1_2_4",
            "fixed_feature_beta": FIXED_FEATURE_BETA,
            "fixed_entry_temperature": FIXED_KERNEL_TEMPERATURE,
            "fixed_internal_temperature_schedule": list(MULTISCALE_TEMPERATURE_SCHEDULE),
            "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
            "multiscale_diffusion_steps": list(MULTISCALE_DIFFUSION_STEPS),
            "multiscale_outer_iterations": len(MULTISCALE_TEMPERATURE_SCHEDULE),
            "multiscale_structure_rank": FGW_STRUCTURE_RANK,
            "multiscale_structure_weight": MULTISCALE_STRUCTURE_WEIGHT,
            "factorization_seed": FGW_FACTORIZATION_SEED,
            "source_marginal_policy": "uniform",
            "target_marginal_policy": "uniform",
            "uses_frozen_compare_N_kl_feature_path": True,
            "node_cost_is_exact_frozen_v5_formula": True,
            "row_stochastic": True,
            "balanced_target_marginal": True,
            "uses_ei_for_fitting": False,
            "uses_layer_identity": False,
            "uses_labels": False,
            "uses_third_timepoint": False,
            "uses_developmental_features": False,
            "parameter_selection_split": "fixed_before_v10_evaluation",
            "heldout_split_observed": False,
        }
        kernels = TransitionKernels(
            kernel_metadata={
                **common_metadata,
                "feature_metadata": feature_set.metadata,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        should_export = bool(
            cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics
        )

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_dict in (("lower", kernels.p_lower), ("upper", kernels.p_upper)):
                source, target, pairwise_used = _select_pair_features(feature_set, side, pair)
                source_adjacency, target_adjacency = _select_pair_adjacencies(
                    context, side, pair
                )
                grn_pairwise = (
                    feature_set.pairwise_lower_grn_features
                    if side == "lower"
                    else feature_set.pairwise_upper_grn_features
                )
                if grn_pairwise is None or pair not in grn_pairwise:
                    raise ValueError(
                        f"{self.name} is missing {side} GRN features for time pair {pair_label}."
                    )
                grn_source, grn_target = grn_pairwise[pair]
                raw_sparse, pij_sparse, dense_pij, diagnostics = self._build_pair_kernel(
                    source=source,
                    target=target,
                    cfg=cfg,
                    source_adjacency=source_adjacency,
                    target_adjacency=target_adjacency,
                    grn_source=grn_source,
                    grn_target=grn_target,
                )
                target_dict[pair] = dense_pij
                block_metadata = diagnostics["block_kl"]
                multiscale_metadata = diagnostics["multiscale_fgw"]
                pair_metadata = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    **common_metadata,
                    "feature_source": (
                        "pairwise_compare_features" if pairwise_used else "timewise_compare_features"
                    ),
                    "pairwise_features_used": bool(pairwise_used),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "grn_block_used": True,
                    "grn_source_shape": list(np.asarray(grn_source).shape),
                    "grn_target_shape": list(np.asarray(grn_target).shape),
                    "combined_cost": block_metadata["combined_cost"],
                    "multiscale_fgw": multiscale_metadata,
                    "raw_matrix_semantics": "balanced_joint_coupling",
                    "row_normalized_matrix_semantics": "conditional_transition_probability",
                    "uses_only_current_pair_timepoints": True,
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
                            "row_normalized_matrix_semantics": "conditional_transition_probability",
                            "uses_only_current_pair_timepoints": True,
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
                "representation": "lightcci_grn_multiscale_directed_fgw_annealed_v10",
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
            },
        )
        return result, kernels
