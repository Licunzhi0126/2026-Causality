from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pij.base import MethodResult, TransitionKernels
from mignet_ce.pij.sinkhorn_3dot import build_3dot_transition_kernel
from mignet_ce.pij.slat_adapter import build_slat_transition_kernel
from mignet_ce.pipelines.vertical_context import VerticalPairContext
from mignet_ce.representations.graph_features import build_graph_feature_result
from mignet_ce.representations.joint_nmf import build_joint_nmf_result
from mignet_ce.representations.laplacian import build_laplacian_result


def build_method_result_and_kernels(
    context: VerticalPairContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[Tuple[int, int]],
) -> tuple[MethodResult, TransitionKernels | None]:
    method = cfg.effective_pij_method()
    if method == "joint_nmf":
        return (
            build_joint_nmf_result(
                context=context,
                n_components=cfg.nmf_components,
                max_iter=cfg.nmf_max_iter,
                seed=cfg.nmf_seed,
            ),
            None,
        )
    if method == "laplacian":
        return (
            build_laplacian_result(
                context=context,
                n_components=cfg.laplacian_components,
                normalized=cfg.laplacian_normalized,
            ),
            None,
        )
    if method == "3dot":
        result = build_graph_feature_result(
            context=context,
            n_components=cfg.pij_feature_components,
            seed=cfg.nmf_seed,
        )
        kernels = TransitionKernels(kernel_metadata={"pij_method": "3dot"})
        for t0, t1 in pairs:
            p_lower, sim_lower = build_3dot_transition_kernel(
                result.lower_features[t0],
                result.lower_features[t1],
                result.lower_coords[t0],
                result.lower_coords[t1],
                epsilon=cfg.ot_epsilon,
                gamma=cfg.ot_gamma,
                max_iter=cfg.ot_max_iter,
                sim_k=cfg.ot_sim_k,
                dist_k=cfg.ot_dist_k,
                return_similarity=True,
            )
            p_upper, sim_upper = build_3dot_transition_kernel(
                result.upper_features[t0],
                result.upper_features[t1],
                result.upper_coords[t0],
                result.upper_coords[t1],
                epsilon=cfg.ot_epsilon,
                gamma=cfg.ot_gamma,
                max_iter=cfg.ot_max_iter,
                sim_k=cfg.ot_sim_k,
                dist_k=cfg.ot_dist_k,
                return_similarity=True,
            )
            kernels.p_lower[(t0, t1)] = p_lower
            kernels.p_upper[(t0, t1)] = p_upper
            kernels.kernel_metadata[f"{context.time_points[t0]}->{context.time_points[t1]}"] = {
                "lower_similarity_shape": list(sim_lower.shape),
                "upper_similarity_shape": list(sim_upper.shape),
                "ot_epsilon": float(cfg.ot_epsilon),
                "ot_gamma": float(cfg.ot_gamma),
                "ot_max_iter": int(cfg.ot_max_iter),
                "ot_sim_k": int(cfg.ot_sim_k),
                "ot_dist_k": int(cfg.ot_dist_k),
            }
        return result, kernels
    if method == "slat":
        result = build_graph_feature_result(
            context=context,
            n_components=cfg.pij_feature_components,
            seed=cfg.slat_seed,
        )
        kernels = TransitionKernels(kernel_metadata={"pij_method": "slat"})
        pairwise_lower: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        pairwise_upper: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for t0, t1 in pairs:
            p_lower, lower_meta = build_slat_transition_kernel(
                result.lower_features[t0],
                result.lower_features[t1],
                result.lower_coords[t0],
                result.lower_coords[t1],
                k_neighbors=cfg.slat_k_neighbors,
                hidden_dim=cfg.slat_hidden_dim,
                n_layers=cfg.slat_layers,
                epochs=cfg.slat_epochs,
                mlp_hidden=cfg.slat_mlp_hidden,
                alpha=cfg.slat_alpha,
                temperature=cfg.slat_temperature,
                seed=cfg.slat_seed,
            )
            p_upper, upper_meta = build_slat_transition_kernel(
                result.upper_features[t0],
                result.upper_features[t1],
                result.upper_coords[t0],
                result.upper_coords[t1],
                k_neighbors=cfg.slat_k_neighbors,
                hidden_dim=cfg.slat_hidden_dim,
                n_layers=cfg.slat_layers,
                epochs=cfg.slat_epochs,
                mlp_hidden=cfg.slat_mlp_hidden,
                alpha=cfg.slat_alpha,
                temperature=cfg.slat_temperature,
                seed=cfg.slat_seed,
            )
            kernels.p_lower[(t0, t1)] = p_lower
            kernels.p_upper[(t0, t1)] = p_upper
            pairwise_lower[(t0, t1)] = (
                lower_meta["source_embedding"],
                lower_meta["target_embedding"],
            )
            pairwise_upper[(t0, t1)] = (
                upper_meta["source_embedding"],
                upper_meta["target_embedding"],
            )
            kernels.kernel_metadata[f"{context.time_points[t0]}->{context.time_points[t1]}"] = {
                "lower": {k: v for k, v in lower_meta.items() if not isinstance(v, np.ndarray)},
                "upper": {k: v for k, v in upper_meta.items() if not isinstance(v, np.ndarray)},
            }
        result.pairwise_lower_features = pairwise_lower
        result.pairwise_upper_features = pairwise_upper
        result.method_metadata["representation"] = "slat"
        return result, kernels
    raise ValueError(f"Unsupported pij_method {method!r}.")
