from __future__ import annotations

"""RegSim-V7: N/R block KL followed by balanced Sinkhorn.

Adaptation sources:
  - mignet_ce/pij/compare/compare_NG_kl_grnanchor_v5.py
  - mignet_ce/pij/compare/compare_NG_kl_sinkhorn_grnanchor_v7.py

The protected V5/V7 files are not modified. This adapter replaces their true
GRN G block with the H5AD regulatory-activity R block stored by
``wyt_cci_regsim``. The NumPy path reuses the frozen V7 solver; the Torch path
mirrors the same equations for differentiable macro-PIJ training.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.config import TemporalRunConfig
from mignet_ce.metrics import pairwise_shared_core_directed_nmf
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PairFeatures, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import matrix_summary, row_normalized_kernel_from_cost
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import (
    FIXED_FEATURE_BETA,
    FIXED_KERNEL_TEMPERATURE,
    N_CORRECTION_WEIGHT,
    build_grnanchored_kl_cost,
)
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import (
    balance_kernel_sinkhorn,
)


TORCH_SINKHORN_ITERATIONS = 96
EPS = 1e-12


@dataclass(frozen=True)
class CCINFeatureSet:
    lower_features: list[np.ndarray]
    upper_features: list[np.ndarray]
    pairwise_lower_features: PairFeatures
    pairwise_upper_features: PairFeatures
    metadata: dict[str, object]


def _cci_graph_payload(
    context: NetworkContext,
    side: str,
    time_index: int,
) -> sp.csr_matrix:
    if side == "lower":
        graph = context.lower_graphs[time_index]
        expected_units = context.lower_units_by_time[time_index]
    elif side == "upper":
        graph = context.upper_graphs[time_index]
        expected_units = context.upper_units_by_time[time_index]
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    if list(map(str, graph.units)) != list(map(str, expected_units)):
        raise ValueError(f"CCI N-feature units are not aligned for {side} time {time_index}.")
    stored = graph.metadata.get("cci_adjacency_csr")
    if stored is None:
        raise ValueError(
            f"RegSim N features require the raw cci_adjacency_csr for {side} time {time_index}."
        )
    matrix = stored.tocsr() if sp.issparse(stored) else sp.csr_matrix(stored)
    if matrix.shape != (len(expected_units), len(expected_units)):
        raise ValueError(
            f"Raw CCI shape {matrix.shape} does not match {len(expected_units)} units."
        )
    return matrix


def _pairwise_zscore(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([source, target]).astype(float)
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < EPS] = 1.0
    split = source.shape[0]
    standardized = (combined - mean) / std
    return standardized[:split], standardized[split:]


def build_pairwise_cci_n_feature_set(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
) -> CCINFeatureSet:
    pairwise_lower: PairFeatures = {}
    pairwise_upper: PairFeatures = {}
    summaries: list[dict[str, object]] = []
    for side, output in (
        ("lower", pairwise_lower),
        ("upper", pairwise_upper),
    ):
        for source_index, target_index in pairs:
            source_matrix = _cci_graph_payload(context, side, source_index)
            target_matrix = _cci_graph_payload(context, side, target_index)
            u_source, v_source, u_target, v_target, core = (
                pairwise_shared_core_directed_nmf(
                    source_matrix.toarray(),
                    target_matrix.toarray(),
                    n_components=cfg.nmf_components,
                    max_iter=cfg.nmf_max_iter,
                    seed=cfg.nmf_seed + source_index * 1009 + target_index,
                )
            )
            source, target = _pairwise_zscore(
                np.hstack([u_source, v_source]),
                np.hstack([u_target, v_target]),
            )
            output[(source_index, target_index)] = (source, target)
            summaries.append(
                {
                    "side": side,
                    "time_pair": (
                        f"{context.time_points[source_index]}->"
                        f"{context.time_points[target_index]}"
                    ),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "core_shape": list(core.shape),
                    "source_adjacency_nnz": int(source_matrix.nnz),
                    "target_adjacency_nnz": int(target_matrix.nnz),
                }
            )
    return CCINFeatureSet(
        lower_features=[
            np.zeros((len(units), 0), dtype=float)
            for units in context.lower_units_by_time
        ],
        upper_features=[
            np.zeros((len(units), 0), dtype=float)
            for units in context.upper_units_by_time
        ],
        pairwise_lower_features=pairwise_lower,
        pairwise_upper_features=pairwise_upper,
        metadata={
            "N_definition": "raw_CCI_pairwise_shared_core_directed_NMF_concat_U_V",
            "pairwise_zscore": True,
            "nmf_components": int(cfg.nmf_components),
            "nmf_max_iter": int(cfg.nmf_max_iter),
            "pairs": summaries,
        },
    )


def _select_n_pair(
    feature_set: CCINFeatureSet,
    side: str,
    pair: TimePair,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if side == "lower":
        pairwise = feature_set.pairwise_lower_features
    elif side == "upper":
        pairwise = feature_set.pairwise_upper_features
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    if pair in pairwise:
        source, target = pairwise[pair]
        return np.asarray(source, dtype=float), np.asarray(target, dtype=float), True
    raise KeyError(f"Missing pairwise CCI N features for {side} pair {pair}.")


def _regsim_graph_payload(
    context: NetworkContext,
    side: str,
    time_index: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    if side == "lower":
        graph = context.lower_graphs[time_index]
        expected_units = context.lower_units_by_time[time_index]
    elif side == "upper":
        graph = context.upper_graphs[time_index]
        expected_units = context.upper_units_by_time[time_index]
    else:
        raise ValueError("side must be one of ['lower', 'upper'].")
    stored = graph.metadata.get("regsim_feature_csr")
    units = list(map(str, graph.metadata.get("regsim_feature_units", [])))
    names = list(map(str, graph.metadata.get("regsim_feature_names", [])))
    if stored is None:
        raise ValueError(
            f"wyt_regsim_v7 requires regsim_feature_csr for {side} time index {time_index}."
        )
    if units != list(map(str, expected_units)):
        raise ValueError(
            f"RegSim feature units are not aligned for {side} time index {time_index}."
        )
    values = sp.csr_matrix(stored).toarray().astype(float)
    if values.shape != (len(units), len(names)):
        raise ValueError(
            f"RegSim feature shape {values.shape} does not match units/names "
            f"{len(units)}x{len(names)}."
        )
    return values, units, names


def build_pairwise_regsim_features(
    context: NetworkContext,
    side: str,
    pair: TimePair,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source, _, source_names = _regsim_graph_payload(context, side, pair[0])
    target, _, target_names = _regsim_graph_payload(context, side, pair[1])
    if source_names != target_names:
        raise ValueError(f"RegSim canonical columns differ across time pair {pair} for {side}.")
    combined = np.vstack([source, target])
    mean = np.nanmean(combined, axis=0, keepdims=True)
    std = np.nanstd(combined, axis=0, keepdims=True)
    standardized_source = np.divide(
        np.nan_to_num(source, nan=0.0) - mean,
        std,
        out=np.zeros_like(source, dtype=float),
        where=std > 0.0,
    )
    standardized_target = np.divide(
        np.nan_to_num(target, nan=0.0) - mean,
        std,
        out=np.zeros_like(target, dtype=float),
        where=std > 0.0,
    )
    return standardized_source, standardized_target, {
        "feature_source": "h5ad_regulatory_activity",
        "standardization": "pairwise_zscore_on_concat_source_target",
        "feature_names": source_names,
        "source_shape": list(standardized_source.shape),
        "target_shape": list(standardized_target.shape),
        "zero_variance_columns": int(np.count_nonzero(np.squeeze(std, axis=0) <= 0.0)),
    }


def build_regsimanchored_kl_cost(
    n_source: np.ndarray,
    n_target: np.ndarray,
    r_source: np.ndarray,
    r_target: np.ndarray,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
) -> tuple[np.ndarray, dict[str, object]]:
    cost, metadata = build_grnanchored_kl_cost(
        n_source,
        n_target,
        r_source,
        r_target,
        beta=beta,
        n_correction_weight=n_correction_weight,
    )
    metadata.update(
        {
            "mode": "raw_regsim_kl_plus_bounded_n_correction",
            "regulatory_block": "H5AD_RegSim_R_not_true_GRN_G",
            "uses_true_grn": False,
            "r_cost": metadata.pop("g_cost"),
            "r_cost_scale": metadata.pop("grn_cost_scale"),
        }
    )
    return cost, metadata


def regsim_v7_pij_numpy(
    n_source: np.ndarray,
    n_target: np.ndarray,
    r_source: np.ndarray,
    r_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cost, cost_metadata = build_regsimanchored_kl_cost(
        n_source,
        n_target,
        r_source,
        r_target,
    )
    kernel, prebalanced = row_normalized_kernel_from_cost(
        cost,
        tau=FIXED_KERNEL_TEMPERATURE,
    )
    joint, conditional, sinkhorn = balance_kernel_sinkhorn(kernel)
    return joint, conditional, cost, {
        "cost": cost_metadata,
        "sinkhorn": sinkhorn,
        "prebalanced_pij": matrix_summary(prebalanced),
    }


def _torch_pairwise_kl(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    source_prob = torch.softmax(source / float(beta), dim=1).clamp_min(EPS)
    target_prob = torch.softmax(target / float(beta), dim=1).clamp_min(EPS)
    source_entropy = (source_prob * torch.log(source_prob)).sum(dim=1, keepdim=True)
    return (source_entropy - source_prob @ torch.log(target_prob).T).clamp_min(0.0)


def _torch_robust_normalize(cost: torch.Tensor) -> torch.Tensor:
    flat = cost.reshape(-1)
    lower = torch.quantile(flat, 0.05)
    upper = torch.quantile(flat, 0.95)
    scale = (upper - lower).clamp_min(EPS)
    return ((cost - lower) / scale).clamp(0.0, 1.0)


def _torch_balanced_sinkhorn_from_cost(
    cost: torch.Tensor,
    *,
    iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError(f"Torch Sinkhorn cost must be non-empty 2D; got {tuple(cost.shape)}.")
    source_count, target_count = cost.shape
    log_kernel = -cost
    log_source = torch.full(
        (source_count,),
        -float(np.log(source_count)),
        dtype=cost.dtype,
        device=cost.device,
    )
    log_target = torch.full(
        (target_count,),
        -float(np.log(target_count)),
        dtype=cost.dtype,
        device=cost.device,
    )
    source_potential = torch.zeros_like(log_source)
    target_potential = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        source_potential = log_source - torch.logsumexp(
            log_kernel + target_potential[None, :],
            dim=1,
        )
        target_potential = log_target - torch.logsumexp(
            log_kernel + source_potential[:, None],
            dim=0,
        )
    joint = torch.exp(
        log_kernel + source_potential[:, None] + target_potential[None, :]
    )
    conditional = joint / joint.sum(dim=1, keepdim=True).clamp_min(EPS)
    return joint, conditional


def regsim_v7_pij_torch(
    n_source: torch.Tensor,
    n_target: torch.Tensor,
    r_source: torch.Tensor,
    r_target: torch.Tensor,
    *,
    beta: float = FIXED_FEATURE_BETA,
    n_correction_weight: float = N_CORRECTION_WEIGHT,
    sinkhorn_iterations: int = TORCH_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    n_cost = _torch_pairwise_kl(n_source, n_target, beta=beta)
    r_cost = _torch_pairwise_kl(r_source, r_target, beta=beta)
    cost = r_cost + float(n_correction_weight) * _torch_robust_normalize(n_cost)
    _, conditional = _torch_balanced_sinkhorn_from_cost(
        cost,
        iterations=sinkhorn_iterations,
    )
    return conditional


class CompareNRKlSinkhornRegSimV7PijMethod:
    name = "wyt_regsim_v7"
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
                "transition_construction": "regsim_v7_balanced_sinkhorn",
                "cost_source": "raw_RegSim_R_KL_plus_0.25_robust_normalized_N_KL",
                "source_marginal_policy": "uniform",
                "target_marginal_policy": "uniform",
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
                joint, pij, cost, details = regsim_v7_pij_numpy(
                    n_source,
                    n_target,
                    r_source,
                    r_target,
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
                    "sinkhorn": details["sinkhorn"],
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
                    "transition_construction": "regsim_v7_balanced_sinkhorn",
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
