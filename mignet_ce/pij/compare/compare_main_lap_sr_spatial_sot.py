from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare.common import export_compare_pair_artifacts
from mignet_ce.pij.compare.cosine import matrix_summary, pairwise_cosine_distance
from mignet_ce.pij.compare.distances import pairwise_euclidean_distance, robust_normalize_cost
from mignet_ce.pij.compare.features import CompareFeatureSet, build_compare_feature_set
from mignet_ce.pij.compare.sparse_ot import run_sparse_semi_relaxed_ot_from_cost


def _coords_for_side(context: NetworkContext, side: str) -> list[np.ndarray]:
    coords = context.lower_coords_by_time if side == "lower" else context.upper_coords_by_time
    units_by_time = context.lower_units_by_time if side == "lower" else context.upper_units_by_time
    if coords and len(coords) >= len(units_by_time):
        return [np.asarray(coords[idx], dtype=float) for idx in range(len(units_by_time))]
    return [np.zeros((len(units), 2), dtype=float) for units in units_by_time]


def _standardize_feature_lists(*groups: Sequence[np.ndarray]) -> tuple[list[np.ndarray], ...]:
    matrices = [np.asarray(matrix, dtype=float) for group in groups for matrix in group]
    if not matrices:
        return tuple([] for _ in groups)
    all_values = np.vstack(matrices)
    means = np.nanmean(all_values, axis=0, keepdims=True)
    stds = np.nanstd(all_values, axis=0, keepdims=True)
    out_groups: list[list[np.ndarray]] = []
    for group in groups:
        out_groups.append(
            [
                np.divide(
                    np.nan_to_num(matrix, nan=0.0) - means,
                    stds,
                    out=np.zeros_like(matrix, dtype=float),
                    where=stds > 0,
                )
                for matrix in group
            ]
        )
    return tuple(out_groups)


def _unweight(features: Sequence[np.ndarray], weight: float) -> list[np.ndarray]:
    if weight <= 0:
        return [np.zeros_like(np.asarray(matrix, dtype=float)) for matrix in features]
    scale = np.sqrt(float(weight))
    return [np.asarray(matrix, dtype=float) / scale for matrix in features]


class CompareMainLapSrSpatialSotPijMethod:
    name = "compare_main_lap_sr_spatial_sot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        if context.network_method != "light_cci":
            raise ValueError("compare_main_lap_sr_spatial_sot requires network_method='light_cci'.")

        l_features = build_compare_feature_set(context, cfg, ("L",))
        sr_weight = float(cfg.pij_sr_weight)
        sr_features = build_compare_feature_set(context, cfg, ("Sr",))
        sr_lower = _unweight(sr_features.lower_features, sr_weight)
        sr_upper = _unweight(sr_features.upper_features, sr_weight)
        lower_coords, upper_coords = _standardize_feature_lists(_coords_for_side(context, "lower"), _coords_for_side(context, "upper"))

        lambda_l = float(cfg.pij_graph_energy_weight)
        lambda_sr = float(cfg.pij_sr_weight)
        lambda_spatial = float(cfg.pij_spatial_weight)
        if lambda_l < 0 or lambda_sr < 0 or lambda_spatial < 0:
            raise ValueError("Main LightCCI cost weights must be nonnegative.")
        if lambda_l + lambda_sr + lambda_spatial <= 0:
            raise ValueError("At least one of pij_graph_energy_weight, pij_sr_weight, or pij_spatial_weight must be positive.")

        lower_method_features = self._method_features(l_features.lower_features, sr_lower, lower_coords, lambda_l, lambda_sr, lambda_spatial)
        upper_method_features = self._method_features(l_features.upper_features, sr_upper, upper_coords, lambda_l, lambda_sr, lambda_spatial)
        feature_names = (
            [f"main_L:{idx + 1}" for idx in range(l_features.lower_features[0].shape[1])]
            + [f"main_Sr:{idx + 1}" for idx in range(sr_lower[0].shape[1])]
            + [f"main_spatial:{idx + 1}" for idx in range(lower_coords[0].shape[1])]
        )
        feature_set = CompareFeatureSet(
            lower_features=lower_method_features,
            upper_features=upper_method_features,
            feature_names=feature_names,
            metadata={
                "pij_method": self.name,
                "method_role": "lightcci_main_method",
                "not_part_of_30_cell_compare_matrix": True,
                "definition": "LightCCI graph -> Laplacian-HKS + SR + spatial pre-cost -> sparse semi-relaxed OT.",
                "contrast_with_compare_L_Sr_sot": (
                    "compare_L_Sr_sot is a matrix cell using only L+Sr feature cosine distance; "
                    "this main method explicitly combines L, SR, and spatial cost components before OT."
                ),
                "network_method": context.network_method,
                "feature_alignment_space": context.feature_alignment_space,
                "weights": {
                    "laplacian_hks": lambda_l,
                    "sr": lambda_sr,
                    "spatial": lambda_spatial,
                },
                "laplacian_feature_metadata": l_features.metadata,
                "sr_feature_metadata": sr_features.metadata,
            },
        )

        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "method_role": "lightcci_main_method",
                "row_stochastic": True,
                "cost_components": ["laplacian_hks", "sr", "spatial"],
                "cost_weights": {
                    "laplacian_hks": lambda_l,
                    "sr": lambda_sr,
                    "spatial": lambda_spatial,
                },
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side in ("lower", "upper"):
                if side == "lower":
                    l_lists, sr_lists, coord_lists, method_lists, target_dict = (
                        l_features.lower_features,
                        sr_lower,
                        lower_coords,
                        lower_method_features,
                        kernels.p_lower,
                    )
                else:
                    l_lists, sr_lists, coord_lists, method_lists, target_dict = (
                        l_features.upper_features,
                        sr_upper,
                        upper_coords,
                        upper_method_features,
                        kernels.p_upper,
                    )
                pre_cost, component_costs, component_summaries = self._pre_cost(
                    l_source=l_lists[pair[0]],
                    l_target=l_lists[pair[1]],
                    sr_source=sr_lists[pair[0]],
                    sr_target=sr_lists[pair[1]],
                    coord_source=coord_lists[pair[0]],
                    coord_target=coord_lists[pair[1]],
                    weights={
                        "laplacian_hks": lambda_l,
                        "sr": lambda_sr,
                        "spatial": lambda_spatial,
                    },
                )
                ot_result = run_sparse_semi_relaxed_ot_from_cost(
                    pre_cost,
                    epsilon=cfg.ot_epsilon,
                    gamma=cfg.ot_gamma,
                    max_iter=cfg.ot_max_iter,
                    source_k=cfg.ot_dist_k,
                    target_k=cfg.ot_sim_k,
                    raw_cost_column="raw_main_pre_cost",
                    cost_source="lightcci_main_laplacian_hks_sr_spatial_pre_cost",
                )
                target_dict[pair] = ot_result.pij_row_normalized_sparse.toarray()
                kernels.kernel_metadata[pair_label][side] = {
                    "method_role": "lightcci_main_method",
                    "cost_components": ["laplacian_hks", "sr", "spatial"],
                    "cost_source": "Laplacian-HKS + SR + spatial pre-cost",
                    "cost_summary": {
                        "weights": {
                            "laplacian_hks": lambda_l,
                            "sr": lambda_sr,
                            "spatial": lambda_spatial,
                        },
                        "components": component_summaries,
                        "pre_cost": matrix_summary(pre_cost),
                    },
                    "sparse_ot": ot_result.convergence,
                }
                if cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0:
                    kernels.kernel_diagnostics[side][pair] = {
                        "main_cost": pre_cost,
                        "laplacian_hks_cost": component_costs["laplacian_hks"],
                        "sr_cost": component_costs["sr"],
                        "spatial_cost": component_costs["spatial"],
                    }
                if should_export:
                    export_compare_pair_artifacts(
                        cfg=cfg,
                        context=context,
                        method_name=self.name,
                        feature_keys=("L", "Sr", "spatial"),
                        pij_key="sot",
                        feature_set=feature_set,
                        pair=pair,
                        side=side,
                        source_features=method_lists[pair[0]],
                        target_features=method_lists[pair[1]],
                        raw_sparse=ot_result.transport_sparse,
                        pij_sparse=ot_result.pij_row_normalized_sparse,
                        diagnostics={
                            "kind": "lightcci_main_lap_sr_spatial_sparse_ot",
                            "method_role": "lightcci_main_method",
                            "cost_components": ["laplacian_hks", "sr", "spatial"],
                            "cost_summary": {
                                "weights": {
                                    "laplacian_hks": lambda_l,
                                    "sr": lambda_sr,
                                    "spatial": lambda_spatial,
                                },
                                "components": component_summaries,
                                "pre_cost": matrix_summary(pre_cost),
                            },
                            "ot_convergence": ot_result.convergence,
                        },
                        sparse_ot_result=ot_result,
                    )

        return (
            MethodResult(
                lower_features=lower_method_features,
                upper_features=upper_method_features,
                lower_coords=context.lower_coords_by_time,
                upper_coords=context.upper_coords_by_time,
                method_metadata={
                    "pij_method": self.name,
                    "representation": "lightcci_main_laplacian_sr_spatial_sot",
                    "method_role": "lightcci_main_method",
                    "not_part_of_30_cell_compare_matrix": True,
                    "feature_names": feature_names,
                    "feature_metadata": feature_set.metadata,
                },
            ),
            kernels,
        )

    @staticmethod
    def _method_features(
        l_features: Sequence[np.ndarray],
        sr_features: Sequence[np.ndarray],
        spatial_features: Sequence[np.ndarray],
        lambda_l: float,
        lambda_sr: float,
        lambda_spatial: float,
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for l_mat, sr_mat, spatial_mat in zip(l_features, sr_features, spatial_features):
            out.append(
                np.hstack(
                    [
                        np.asarray(l_mat, dtype=float) * np.sqrt(max(lambda_l, 0.0)),
                        np.asarray(sr_mat, dtype=float) * np.sqrt(max(lambda_sr, 0.0)),
                        np.asarray(spatial_mat, dtype=float) * np.sqrt(max(lambda_spatial, 0.0)),
                    ]
                )
            )
        return out

    @staticmethod
    def _pre_cost(
        *,
        l_source: np.ndarray,
        l_target: np.ndarray,
        sr_source: np.ndarray,
        sr_target: np.ndarray,
        coord_source: np.ndarray,
        coord_target: np.ndarray,
        weights: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
        l_cost, _ = robust_normalize_cost(pairwise_cosine_distance(l_source, l_target))
        sr_cost, _ = robust_normalize_cost(pairwise_euclidean_distance(sr_source, sr_target))
        spatial_cost, _ = robust_normalize_cost(pairwise_euclidean_distance(coord_source, coord_target))
        total_weight = sum(float(value) for value in weights.values())
        pre_cost = (
            float(weights["laplacian_hks"]) * l_cost
            + float(weights["sr"]) * sr_cost
            + float(weights["spatial"]) * spatial_cost
        ) / total_weight
        component_costs = {
            "laplacian_hks": l_cost,
            "sr": sr_cost,
            "spatial": spatial_cost,
        }
        component_summaries = {
            name: {
                "weight": float(weights[name]),
                "summary": matrix_summary(cost),
            }
            for name, cost in component_costs.items()
        }
        return pre_cost, component_costs, component_summaries
