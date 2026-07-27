from __future__ import annotations

"""RegSim-V9: RegSim-V7 node cost plus directed low-rank FGW structure.

Adaptation source:
  mignet_ce/pij/compare/compare_NG_fgw_grnanchor_v9.py

The protected V9 implementation remains unchanged. Its NumPy FGW solver is
reused for vertical/micro PIJ, while a Torch mirror supplies differentiable
macro PIJ inside the WYT DeltaEI learner.
"""

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PairFeatures, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary
from mignet_ce.pij.compare._shared.lowrank_fgw import (
    FGW_OUTER_ITERATIONS,
    FGW_STRUCTURE_RANK,
    FGW_STRUCTURE_WEIGHT,
    solve_lowrank_directed_fgw,
)
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    FIXED_KERNEL_TEMPERATURE,
    N_CORRECTION_WEIGHT,
)
from mignet_ce.pij.compare.compare_NR_kl_sinkhorn_regsim_v7 import (
    TORCH_SINKHORN_ITERATIONS,
    _select_n_pair,
    _torch_balanced_sinkhorn_from_cost,
    _torch_pairwise_kl,
    _torch_robust_normalize,
    build_pairwise_cci_n_feature_set,
    build_pairwise_regsim_features,
    build_regsimanchored_kl_cost,
)


EPS = 1e-12


def _select_pair_adjacencies(
    context: NetworkContext,
    side: str,
    pair: TimePair,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if context.feature_alignment_space != "native_units":
        raise ValueError("wyt_regsim_v9 requires native-unit graph alignment.")
    if side == "lower":
        graphs = context.lower_graphs
        units_by_time = context.lower_units_by_time
    elif side == "upper":
        graphs = context.upper_graphs
        units_by_time = context.upper_units_by_time
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    matrices: list[sp.csr_matrix] = []
    for time_index in pair:
        graph = graphs[time_index]
        expected = list(map(str, units_by_time[time_index]))
        if list(map(str, graph.units)) != expected:
            raise ValueError(f"RegSim-V9 {side} graph units are not aligned.")
        stored = graph.metadata.get("adjacency_csr")
        if stored is None:
            raise ValueError(f"RegSim-V9 requires adjacency_csr for {side} {time_index}.")
        matrix = stored.tocsr() if sp.issparse(stored) else sp.csr_matrix(stored)
        if matrix.shape != (len(expected), len(expected)):
            raise ValueError(
                f"RegSim-V9 adjacency shape {matrix.shape} does not match {len(expected)} units."
            )
        matrices.append(matrix)
    return matrices[0], matrices[1]


def regsim_v9_pij_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    r_source: np.ndarray,
    r_target: np.ndarray,
    source_adjacency: sp.spmatrix | np.ndarray,
    target_adjacency: sp.spmatrix | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cost, cost_metadata = build_regsimanchored_kl_cost(
        n_source,
        n_target,
        r_source,
        r_target,
    )
    joint, conditional, fgw = solve_lowrank_directed_fgw(
        cost,
        source_adjacency,
        target_adjacency,
    )
    return joint, conditional, cost, {"cost": cost_metadata, "fgw": fgw}


def _torch_row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    values = matrix.clamp_min(0.0)
    return values / values.sum(dim=1, keepdim=True).clamp_min(EPS)


def _torch_directed_factors(
    matrix: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = _torch_row_normalize(matrix)
    left, singular, right_h = torch.linalg.svd(normalized, full_matrices=False)
    effective = min(int(rank), singular.shape[0])
    root = torch.sqrt(singular[:effective].clamp_min(0.0))
    return left[:, :effective] * root[None, :], right_h[:effective].T * root[None, :]


def _torch_directed_structural_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    source_left: torch.Tensor,
    source_right: torch.Tensor,
    target_left: torch.Tensor,
    target_right: torch.Tensor,
    coupling: torch.Tensor,
) -> torch.Tensor:
    source_count, target_count = coupling.shape
    source_mass = torch.full(
        (source_count,),
        1.0 / source_count,
        dtype=source.dtype,
        device=source.device,
    )
    target_mass = torch.full(
        (target_count,),
        1.0 / target_count,
        dtype=target.dtype,
        device=target.device,
    )
    source_out = source.square() @ source_mass
    target_out = target.square() @ target_mass
    source_in = source.T.square() @ source_mass
    target_in = target.T.square() @ target_mass
    outgoing_middle = source_right.T @ coupling @ target_right
    incoming_middle = source_left.T @ coupling @ target_left
    outgoing_cross = source_left @ outgoing_middle @ target_left.T
    incoming_cross = source_right @ incoming_middle @ target_right.T
    outgoing = source_out[:, None] + target_out[None, :] - 2.0 * outgoing_cross
    incoming = source_in[:, None] + target_in[None, :] - 2.0 * incoming_cross
    return (0.5 * (outgoing + incoming)).clamp_min(0.0)


def regsim_v9_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    r_source: torch.Tensor,
    r_target: torch.Tensor,
    source_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
    outer_iterations: int = FGW_OUTER_ITERATIONS,
    structure_rank: int = FGW_STRUCTURE_RANK,
    structure_weight: float = FGW_STRUCTURE_WEIGHT,
    sinkhorn_iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    n_cost = _torch_pairwise_kl(n_source, n_target, beta=beta)
    r_cost = _torch_pairwise_kl(r_source, r_target, beta=beta)
    node_cost = r_cost + float(n_correction_weight) * _torch_robust_normalize(n_cost)
    source = _torch_row_normalize(source_adjacency)
    target = _torch_row_normalize(target_adjacency)
    source_left, source_right = _torch_directed_factors(source, structure_rank)
    target_left, target_right = _torch_directed_factors(target, structure_rank)
    joint, conditional = _torch_balanced_sinkhorn_from_cost(
        node_cost,
        iterations=sinkhorn_iterations,
    )
    for _ in range(int(outer_iterations)):
        structural = _torch_directed_structural_cost(
            source,
            target,
            source_left,
            source_right,
            target_left,
            target_right,
            joint,
        )
        combined = node_cost + float(structure_weight) * _torch_robust_normalize(structural)
        joint, conditional = _torch_balanced_sinkhorn_from_cost(
            combined,
            iterations=sinkhorn_iterations,
        )
    return conditional


class CompareNRFGWRegSimV9PijMethod:
    name = "wyt_regsim_v9"
    feature_keys = ("N",)
    pij_key = "kl"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        if context.network_method != "wyt_cci_regsim":
            raise ValueError(f"{self.name} requires network_method='wyt_cci_regsim'.")
        if not np.isclose(cfg.pij_entropy_epsilon, FIXED_FEATURE_BETA):
            raise ValueError(
                f"{self.name} fixes pij_entropy_epsilon={FIXED_FEATURE_BETA}; "
                f"got {cfg.pij_entropy_epsilon}."
            )
        if not np.isclose(cfg.pij_temperature, FIXED_KERNEL_TEMPERATURE):
            raise ValueError(
                f"{self.name} fixes pij_temperature={FIXED_KERNEL_TEMPERATURE}; "
                f"got {cfg.pij_temperature}."
            )
        feature_set = build_pairwise_cci_n_feature_set(context, cfg, pairs)
        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "transition_construction": "regsim_v9_lowrank_directed_fgw",
                "node_cost_source": "raw_RegSim_R_KL_plus_0.25_robust_normalized_N_KL",
                "graph_cost_source": "wyt_cci_regsim_integrated_adjacency",
                "fgw_outer_iterations": FGW_OUTER_ITERATIONS,
                "fgw_structure_rank": FGW_STRUCTURE_RANK,
                "fgw_structure_weight": FGW_STRUCTURE_WEIGHT,
                "uses_true_grn": False,
                "feature_metadata": feature_set.metadata,
                "row_stochastic": True,
                "balanced_target_marginal": True,
            }
        )
        regsim_lower: PairFeatures = {}
        regsim_upper: PairFeatures = {}
        for pair in pairs:
            label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[label] = {}
            for side, target_dict, regsim_dict in (
                ("lower", kernels.p_lower, regsim_lower),
                ("upper", kernels.p_upper, regsim_upper),
            ):
                n_source, n_target, pairwise_used = _select_n_pair(feature_set, side, pair)
                r_source, r_target, r_metadata = build_pairwise_regsim_features(
                    context,
                    side,
                    pair,
                )
                source_adj, target_adj = _select_pair_adjacencies(context, side, pair)
                joint, pij, cost, details = regsim_v9_pij_numpy(
                    n_source,
                    n_target,
                    r_source,
                    r_target,
                    source_adj,
                    target_adj,
                )
                target_dict[pair] = pij
                regsim_dict[pair] = (r_source, r_target)
                kernels.kernel_metadata[label][side] = {
                    "feature_source": "pairwise_N_and_RegSim_R",
                    "pairwise_n_used": bool(pairwise_used),
                    "n_source_shape": list(n_source.shape),
                    "n_target_shape": list(n_target.shape),
                    "r_source_shape": list(r_source.shape),
                    "r_target_shape": list(r_target.shape),
                    "regsim": r_metadata,
                    "cost": matrix_summary(cost),
                    "joint": matrix_summary(joint),
                    "pij": matrix_summary(pij),
                    "fgw": details["fgw"],
                    "uses_true_grn": False,
                }
                if cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0:
                    kernels.kernel_diagnostics[side][pair] = {"main_cost": cost}
        return (
            MethodResult(
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
                    "representation": "pairwise_joint_NMF_plus_H5AD_RegSim_R",
                    "transition_construction": "regsim_v9_lowrank_directed_fgw",
                    "uses_true_grn": False,
                    "feature_metadata": feature_set.metadata,
                    "regsim_pairwise_lower_dims": {
                        f"{left}->{right}": int(pair[0].shape[1])
                        for (left, right), pair in regsim_lower.items()
                    },
                    "regsim_pairwise_upper_dims": {
                        f"{left}->{right}": int(pair[0].shape[1])
                        for (left, right), pair in regsim_upper.items()
                    },
                },
            ),
            kernels,
        )
