from __future__ import annotations

from typing import Sequence

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class ThreeDotPijMethod:
    name = "3dot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        from mignet_ce.pij.sinkhorn_3dot import build_3dot_transition_kernel
        from mignet_ce.representations.graph_features import build_graph_feature_result

        result = build_graph_feature_result(
            context=context,
            n_components=cfg.pij_feature_components,
            seed=cfg.nmf_seed,
        )
        kernels = TransitionKernels(kernel_metadata={"pij_method": self.name})
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
