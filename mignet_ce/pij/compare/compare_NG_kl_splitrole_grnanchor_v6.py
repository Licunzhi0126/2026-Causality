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
from mignet_ce.pij.feature_versions.grn_features import build_split_grn_features
from mignet_ce.pij.feature_versions.sources import (
    align_feature_to_context,
    load_raw_grn_inputs,
    standardize_pair,
)
from mignet_ce.pij.feature_versions.spec import FeatureRecipe


FIXED_FEATURE_BETA = 0.05
FIXED_KERNEL_TEMPERATURE = 1.0
N_CORRECTION_WEIGHT = 0.25
REGULATOR_ROLE_WEIGHT = 0.5
TARGET_ROLE_WEIGHT = 0.5


def _split_role_extraction_recipe(cfg: TemporalRunConfig) -> FeatureRecipe:
    """Resolve only the split-role extractor settings; V2 cost fusion is not reused."""
    return FeatureRecipe(
        recipe_id="splitrole_grnanchor_v6_extractor",
        entry_method="compare_NG_kl_splitrole_grnanchor_v6",
        algorithm_version="lightcci_grn_splitrole_grnanchor_v6",
        cci_mode="frozen_compare_N_pairwise_nmf",
        grn_mode="split_reg_tar_recomputed",
        block_distances={"g_reg": "kl", "g_tar": "kl"},
        distance_parameters={
            "g_reg": {"beta": FIXED_FEATURE_BETA},
            "g_tar": {"beta": FIXED_FEATURE_BETA},
        },
        fusion_weights={
            "g_reg": REGULATOR_ROLE_WEIGHT,
            "g_tar": TARGET_ROLE_WEIGHT,
        },
        nmf_rank=int(cfg.nmf_components),
        nmf_max_iter=int(cfg.nmf_max_iter),
        nmf_seeds=(int(cfg.nmf_seed),),
        grn_topk_targets=int(cfg.grn_topk_targets),
        projection_dim=int(cfg.grn_state_dim),
        projection_seed=int(cfg.grn_projection_seed),
        kernel_temperature=FIXED_KERNEL_TEMPERATURE,
        pseudocount=1.0e-8,
    )


def build_splitrole_grnanchored_kl_cost(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_reg_source: np.ndarray,
    g_reg_target: np.ndarray,
    g_tar_source: np.ndarray,
    g_tar_target: np.ndarray,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fuse raw regulator/target GRN KL with a bounded frozen-N correction."""
    beta = float(beta)
    n_correction_weight = float(n_correction_weight)
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if n_correction_weight < 0.0:
        raise ValueError("n_correction_weight must be nonnegative.")

    n_cost = pairwise_feature_kl(n_source, n_target, beta=beta)
    g_reg_cost = pairwise_feature_kl(g_reg_source, g_reg_target, beta=beta)
    g_tar_cost = pairwise_feature_kl(g_tar_source, g_tar_target, beta=beta)
    if n_cost.shape != g_reg_cost.shape or n_cost.shape != g_tar_cost.shape:
        raise ValueError(
            "N, regulator-GRN, and target-GRN KL cost shapes differ: "
            f"{n_cost.shape}, {g_reg_cost.shape}, {g_tar_cost.shape}."
        )

    normalized_n, n_normalization = robust_normalize_cost(n_cost, copy=True)
    raw_split_grn = REGULATOR_ROLE_WEIGHT * g_reg_cost + TARGET_ROLE_WEIGHT * g_tar_cost
    combined = raw_split_grn + n_correction_weight * normalized_n
    if not np.isfinite(combined).all() or np.any(combined < 0.0):
        raise ValueError("Split-role GRN-anchored KL cost must be finite and nonnegative.")

    return combined, {
        "mode": "raw_split_role_grn_kl_plus_bounded_n_correction",
        "beta_n": beta,
        "beta_g_reg": beta,
        "beta_g_tar": beta,
        "regulator_role_weight": REGULATOR_ROLE_WEIGHT,
        "target_role_weight": TARGET_ROLE_WEIGHT,
        "n_correction_weight": n_correction_weight,
        "n_cost": summarize_dense_cost(n_cost),
        "g_reg_cost": summarize_dense_cost(g_reg_cost),
        "g_tar_cost": summarize_dense_cost(g_tar_cost),
        "raw_split_grn_cost": summarize_dense_cost(raw_split_grn),
        "n_normalization": n_normalization,
        "combined_cost": summarize_dense_cost(combined),
        "grn_cost_scale": "raw_role_separated_kl_nats",
        "n_correction_scale": f"robust_5_95_times_{n_correction_weight:g}",
        "final_cost_clipped_to_unit_interval": False,
        "removes_unit_interval_gibbs_ei_bound": True,
        "regulator_target_summed_before_distance": False,
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


def _build_split_role_cache(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
) -> dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, object]]]:
    recipe = _split_role_extraction_recipe(cfg)
    needed_indices = sorted({index for pair in pairs for index in pair})
    cache: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, object]]] = {}
    for side in ("lower", "upper"):
        for time_index in needed_indices:
            raw = load_raw_grn_inputs(context, cfg, side, time_index)
            role_blocks, metadata, _ = build_split_grn_features(raw, recipe)
            aligned_blocks: dict[str, np.ndarray] = {}
            alignments: dict[str, object] = {}
            for name, values in role_blocks.items():
                aligned, alignment = align_feature_to_context(values, context, side, time_index)
                aligned_blocks[name] = np.asarray(aligned, dtype=float)
                alignments[name] = alignment
            cache[(side, time_index)] = (
                aligned_blocks,
                {
                    **metadata,
                    "alignments": alignments,
                    "uses_only_native_stage_input": True,
                    "uses_developmental_features": False,
                    "uses_labels": False,
                },
            )
    return cache


class CompareNGKlSplitRoleGRNAnchorV6PijMethod:
    """Frozen N features plus role-separated raw GRN KL in an isolated V6 method."""

    name = "compare_NG_kl_splitrole_grnanchor_v6"
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
        g_reg_source: np.ndarray | None = None,
        g_reg_target: np.ndarray | None = None,
        g_tar_source: np.ndarray | None = None,
        g_tar_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        role_values = (g_reg_source, g_reg_target, g_tar_source, g_tar_target)
        if any(value is None for value in role_values):
            raise ValueError(f"{self.name} requires separate regulator and target GRN feature blocks.")
        if not np.isclose(float(beta), FIXED_FEATURE_BETA, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{self.name} fixes pij_entropy_epsilon={FIXED_FEATURE_BETA}; got {float(beta)}."
            )
        cost, metadata = build_splitrole_grnanchored_kl_cost(
            source,
            target,
            np.asarray(g_reg_source, dtype=float),
            np.asarray(g_reg_target, dtype=float),
            np.asarray(g_tar_source, dtype=float),
            np.asarray(g_tar_target, dtype=float),
            beta=FIXED_FEATURE_BETA,
            n_correction_weight=N_CORRECTION_WEIGHT,
        )
        metadata.update(
            {
                "entry_method": self.name,
                "algorithm_version": "lightcci_grn_splitrole_grnanchor_v6",
                "uses_frozen_compare_N_feature_path": True,
                "legacy_kl_block_weight_n_received_but_not_used": float(weight_n),
                "legacy_kl_block_weight_g_received_but_not_used": float(weight_g),
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                "uses_v2_cost_fusion": False,
            }
        )
        return cost, metadata

    def _build_pair_kernel(
        self,
        *,
        source: np.ndarray,
        target: np.ndarray,
        cfg: TemporalRunConfig,
        g_reg_source: np.ndarray,
        g_reg_target: np.ndarray,
        g_tar_source: np.ndarray,
        g_tar_target: np.ndarray,
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
            g_reg_source=g_reg_source,
            g_reg_target=g_reg_target,
            g_tar_source=g_tar_source,
            g_tar_target=g_tar_target,
        )
        kernel, pij = row_normalized_kernel_from_cost(cost, tau=FIXED_KERNEL_TEMPERATURE)
        diagnostics = {
            "kind": "splitrole_grnanchored_block_feature_kl_kernel",
            "beta": FIXED_FEATURE_BETA,
            "tau": FIXED_KERNEL_TEMPERATURE,
            "cost": matrix_summary(cost),
            "kernel": matrix_summary(kernel),
            "main_cost_dense": cost,
            "block_kl": block_metadata,
        }
        return sp.csr_matrix(kernel), sp.csr_matrix(pij), pij, diagnostics

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        if context.network_method != "light_cci_grn":
            raise ValueError(f"{self.name} requires network_method='light_cci_grn'.")
        if context.feature_alignment_space != "native_units":
            raise ValueError(f"{self.name} requires native_units feature alignment to prevent layer projection leakage.")
        if context.pair.lower_layer == "gene" or context.pair.upper_layer == "gene":
            raise ValueError(f"{self.name} currently supports only non-gene layer pairs.")

        feature_set = build_compare_feature_set(context, cfg, self.feature_keys)
        split_role_cache = _build_split_role_cache(context, cfg, pairs)
        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "raw_split_role_grn_kl_plus_bounded_n_correction",
                "transition_construction": "splitrole_grnanchored_block_kl",
                "cost_source": "0.5_raw_Greg_KL_plus_0.5_raw_Gtar_KL_plus_0.25_robust_normalized_N_KL",
                "fixed_feature_beta": FIXED_FEATURE_BETA,
                "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                "uses_frozen_compare_N_feature_path": True,
                "feature_metadata": feature_set.metadata,
                "row_stochastic": True,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
                "transductive_pairwise_nmf": True,
                "uses_target_for_current_pair_nmf_fit": True,
                "uses_third_timepoint": False,
                "uses_developmental_features": False,
                "uses_labels": False,
                "uses_lower_to_upper_projection": False,
                "heldout_split_observed": False,
            }
        )
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_dict in (("lower", kernels.p_lower), ("upper", kernels.p_upper)):
                source, target, pairwise_used = _select_pair_features(feature_set, side, pair)
                source_roles, source_role_metadata = split_role_cache[(side, pair[0])]
                target_roles, target_role_metadata = split_role_cache[(side, pair[1])]
                g_reg_source, g_reg_target, g_reg_standardization = standardize_pair(
                    source_roles["g_reg"], target_roles["g_reg"]
                )
                g_tar_source, g_tar_target, g_tar_standardization = standardize_pair(
                    source_roles["g_tar"], target_roles["g_tar"]
                )
                raw_sparse, pij_sparse, dense_pij, diagnostics = self._build_pair_kernel(
                    source=source,
                    target=target,
                    cfg=cfg,
                    g_reg_source=g_reg_source,
                    g_reg_target=g_reg_target,
                    g_tar_source=g_tar_source,
                    g_tar_target=g_tar_target,
                )
                target_dict[pair] = dense_pij
                block_metadata = diagnostics["block_kl"]
                role_metadata = {
                    "regulator": {
                        "source_shape": list(g_reg_source.shape),
                        "target_shape": list(g_reg_target.shape),
                        "standardization": g_reg_standardization,
                    },
                    "target": {
                        "source_shape": list(g_tar_source.shape),
                        "target_shape": list(g_tar_target.shape),
                        "standardization": g_tar_standardization,
                    },
                    "source_extraction": source_role_metadata,
                    "target_extraction": target_role_metadata,
                }
                kernels.kernel_metadata[pair_label][side] = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    "fusion_mode": "raw_split_role_grn_kl_plus_bounded_n_correction",
                    "transition_construction": "splitrole_grnanchored_block_kl",
                    "cost_source": "0.5_raw_Greg_KL_plus_0.5_raw_Gtar_KL_plus_0.25_robust_normalized_N_KL",
                    "feature_source": "pairwise_compare_features" if pairwise_used else "timewise_compare_features",
                    "pairwise_features_used": bool(pairwise_used),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "grn_roles": role_metadata,
                    "final_cost_clipped_to_unit_interval": False,
                    "combined_cost": block_metadata["combined_cost"],
                    "uses_only_current_pair_timepoints": True,
                    "uses_developmental_features": False,
                    "uses_labels": False,
                }

                if should_export:
                    combined_grn_source = np.hstack([g_reg_source, g_tar_source])
                    combined_grn_target = np.hstack([g_reg_target, g_tar_target])
                    artifact_directory = export_compare_pair_artifacts(
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
                            "fusion_mode": "raw_split_role_grn_kl_plus_bounded_n_correction",
                            "transition_construction": "splitrole_grnanchored_block_kl",
                            "cost_source": "0.5_raw_Greg_KL_plus_0.5_raw_Gtar_KL_plus_0.25_robust_normalized_N_KL",
                            "fixed_feature_beta": FIXED_FEATURE_BETA,
                            "fixed_kernel_temperature": FIXED_KERNEL_TEMPERATURE,
                            "fixed_n_correction_weight": N_CORRECTION_WEIGHT,
                            "regulator_target_summed_before_distance": False,
                            "final_cost_clipped_to_unit_interval": False,
                            "uses_only_current_pair_timepoints": True,
                            "uses_third_timepoint": False,
                            "uses_developmental_features": False,
                            "uses_labels": False,
                            "uses_lower_to_upper_projection": False,
                            "heldout_split_observed": False,
                        },
                        grn_source_features=combined_grn_source,
                        grn_target_features=combined_grn_target,
                    )
                    np.save(artifact_directory / "grn_reg_features_source.npy", g_reg_source)
                    np.save(artifact_directory / "grn_reg_features_target.npy", g_reg_target)
                    np.save(artifact_directory / "grn_tar_features_source.npy", g_tar_source)
                    np.save(artifact_directory / "grn_tar_features_target.npy", g_tar_target)

        result = MethodResult(
            lower_features=feature_set.lower_features,
            upper_features=feature_set.upper_features,
            lower_coords=context.lower_coords_by_time,
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=feature_set.pairwise_lower_features,
            pairwise_upper_features=feature_set.pairwise_upper_features,
            method_metadata={
                "pij_method": self.name,
                "representation": "lightcci_grn_splitrole_grnanchor_v6",
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "fusion_mode": "raw_split_role_grn_kl_plus_bounded_n_correction",
                "transition_construction": "splitrole_grnanchored_block_kl",
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
                "uses_frozen_compare_N_feature_path": True,
                "regulator_target_summed_before_distance": False,
                "transductive_pairwise_nmf": True,
                "uses_third_timepoint": False,
                "uses_developmental_features": False,
                "uses_labels": False,
                "heldout_split_observed": False,
            },
        )
        return result, kernels
