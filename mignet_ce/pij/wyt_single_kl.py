from __future__ import annotations

"""WYT network80 + single-direction KL transition method.

Migration sources:
  - reference/network_only_coarse_grain/scripts/build_network_only_features_v57.py
  - reference/network_only_coarse_grain/scripts/train_feature_align_deltaei_v40.py
"""

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as torch_f

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PairFeatures, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary, row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.representations.wyt_network80 import Network80Pair, build_network80_pair


EPS = 1e-12


def single_kl_pij_numpy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if float(temperature) <= 0.0:
        raise ValueError("WYT single-KL temperature must be positive.")
    cost = pairwise_feature_kl(source, target, beta=1.0)
    _, pij = row_normalized_kernel_from_cost(cost, tau=float(temperature))
    return cost, pij


def single_kl_pij_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("WYT single-KL temperature must be positive.")
    source_prob = torch_f.softmax(source, dim=1)
    target_prob = torch_f.softmax(target, dim=1)
    source_safe = torch.clamp(source_prob, min=EPS)
    target_safe = torch.clamp(target_prob, min=EPS)
    cost = (
        source_safe[:, None, :]
        * (
            torch.log(source_safe)[:, None, :]
            - torch.log(target_safe)[None, :, :]
        )
    ).sum(dim=2)
    return torch_f.softmax(-cost / float(temperature), dim=1)


def _graph_adjacency(context: NetworkContext, side: str, time_index: int) -> sp.csr_matrix:
    if side == "lower":
        graph = context.lower_graphs[time_index]
        expected_units = context.lower_units_by_time[time_index]
    elif side == "upper":
        graph = context.upper_graphs[time_index]
        expected_units = context.upper_units_by_time[time_index]
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    if list(map(str, graph.units)) != list(map(str, expected_units)):
        raise ValueError(f"WYT single-KL {side} graph units are not aligned.")
    stored = graph.metadata.get("adjacency_csr")
    if stored is None:
        raise ValueError(f"WYT single-KL requires adjacency_csr for {side} time {time_index}.")
    matrix = stored.tocsr() if sp.issparse(stored) else sp.csr_matrix(stored, dtype=float)
    if matrix.shape != (len(expected_units), len(expected_units)):
        raise ValueError(
            f"WYT single-KL adjacency shape {matrix.shape} does not match {len(expected_units)} units."
        )
    return matrix


class WYTSingleKLPijMethod:
    name = "wyt_single_kl"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "representation": "wyt_network80_joint_zscore_fixed_pca",
                "transition_construction": "single_direction_feature_kl_row_softmax",
                "network80_svd_dim": int(cfg.wyt_network_svd_dim),
                "network80_pca_dim": int(cfg.wyt_network_pca_dim),
                "temperature": float(cfg.pij_temperature),
                "row_stochastic": True,
                "target_marginal_constrained": False,
                "uses_fgw": False,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        pairwise_lower: PairFeatures = {}
        pairwise_upper: PairFeatures = {}
        first_feature_names: list[str] = []
        pair_metadata: dict[str, object] = {}

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            pair_metadata[pair_label] = {}
            for side, target_dict, feature_dict in (
                ("lower", kernels.p_lower, pairwise_lower),
                ("upper", kernels.p_upper, pairwise_upper),
            ):
                network_pair: Network80Pair = build_network80_pair(
                    _graph_adjacency(context, side, pair[0]),
                    _graph_adjacency(context, side, pair[1]),
                    svd_dim=cfg.wyt_network_svd_dim,
                    pca_dim=cfg.wyt_network_pca_dim,
                    random_state=cfg.nmf_seed,
                )
                cost, pij = single_kl_pij_numpy(
                    network_pair.latent_t,
                    network_pair.latent_tp,
                    temperature=cfg.pij_temperature,
                )
                target_dict[pair] = pij
                feature_dict[pair] = (network_pair.latent_t, network_pair.latent_tp)
                if not first_feature_names:
                    first_feature_names = list(network_pair.feature_names)
                side_metadata = {
                    **network_pair.metadata,
                    "cost": matrix_summary(cost),
                    "pij": matrix_summary(pij),
                    "source_shape": list(network_pair.latent_t.shape),
                    "target_shape": list(network_pair.latent_tp.shape),
                    "temperature": float(cfg.pij_temperature),
                    "row_stochastic": True,
                    "target_marginal_constrained": False,
                }
                kernels.kernel_metadata[pair_label][side] = side_metadata
                pair_metadata[pair_label][side] = side_metadata

        lower_empty = [
            np.zeros((len(units), 0), dtype=float)
            for units in context.lower_units_by_time
        ]
        upper_empty = [
            np.zeros((len(units), 0), dtype=float)
            for units in context.upper_units_by_time
        ]
        result = MethodResult(
            lower_features=lower_empty,
            upper_features=upper_empty,
            lower_coords=(
                context.lower_coords_by_time
                if context.feature_alignment_space == "native_units"
                else context.upper_coords_by_time
            ),
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=pairwise_lower,
            pairwise_upper_features=pairwise_upper,
            method_metadata={
                "pij_method": self.name,
                "representation": "wyt_network80_joint_zscore_fixed_pca",
                "feature_names": first_feature_names,
                "transition_construction": "single_direction_feature_kl_row_softmax",
                "temperature": float(cfg.pij_temperature),
                "pair_metadata": pair_metadata,
            },
        )
        return result, kernels
