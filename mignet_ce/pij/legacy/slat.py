from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class SLATPijMethod:
    name = "slat"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        from mignet_ce.transition.slat_adapter import build_slat_transition_kernel
        from mignet_ce.representations.graph_features import build_graph_feature_result

        result = build_graph_feature_result(
            context=context,
            n_components=cfg.pij_feature_components,
            seed=cfg.slat_seed,
        )
        kernels = TransitionKernels(kernel_metadata={"pij_method": self.name})
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
                "lower": {key: value for key, value in lower_meta.items() if not isinstance(value, np.ndarray)},
                "upper": {key: value for key, value in upper_meta.items() if not isinstance(value, np.ndarray)},
            }
        result.pairwise_lower_features = pairwise_lower
        result.pairwise_upper_features = pairwise_upper
        result.method_metadata["representation"] = self.name
        return result, kernels
