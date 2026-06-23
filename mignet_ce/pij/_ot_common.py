from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.representations.graph_features import build_graph_feature_result
from mignet_ce.transition.cost_components import combine_costs
from mignet_ce.transition.ot import build_entropic_ot_kernel


CostBuilder = Callable[
    [np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str, int, int],
    tuple[Mapping[str, np.ndarray], Mapping[str, float]]
    | tuple[Mapping[str, np.ndarray], Mapping[str, float], Mapping[str, object]],
]


def _coords_at(coords: list[np.ndarray] | None, index: int) -> np.ndarray | None:
    if coords is None:
        return None
    return coords[index]


def _build_kernel(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_coords: np.ndarray | None,
    target_coords: np.ndarray | None,
    space: str,
    t0: int,
    t1: int,
    cfg: TemporalRunConfig,
    component_builder: CostBuilder,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    built = component_builder(
        source_features,
        target_features,
        source_coords,
        target_coords,
        space,
        t0,
        t1,
    )
    if len(built) == 2:
        components, weights = built
        extra_metadata: Mapping[str, object] = {}
    else:
        components, weights, extra_metadata = built
    cost, cost_summary = combine_costs(components, weights)
    kernel = build_entropic_ot_kernel(
        cost,
        epsilon=cfg.pij_entropy_epsilon,
        use_pot=True,
        unbalanced=cfg.pij_use_unbalanced_ot,
        mass_reg=cfg.pij_unbalanced_mass,
        max_iter=cfg.ot_max_iter,
    )
    metadata = {
        "space": space,
        "cost_shape": list(cost.shape),
        "kernel_shape": list(kernel.shape),
        "cost_components": list(components),
        "cost_summary": cost_summary,
        "row_stochastic": bool(kernel.shape[1] == 0 or np.allclose(kernel.sum(axis=1), 1.0)),
    }
    metadata.update(dict(extra_metadata))
    diagnostics = {f"{name}_cost": np.asarray(cost, dtype=float) for name, cost in components.items()}
    diagnostics["main_cost"] = np.asarray(cost, dtype=float)
    return kernel, metadata, diagnostics


def run_ot_pij_method(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
    method_name: str,
    component_builder: CostBuilder,
    seed: int | None = None,
) -> tuple[MethodResult, TransitionKernels]:
    result = build_graph_feature_result(
        context=context,
        n_components=cfg.pij_feature_components,
        seed=cfg.nmf_seed if seed is None else seed,
    )
    return run_ot_pij_method_with_result(
        context=context,
        cfg=cfg,
        pairs=pairs,
        method_name=method_name,
        component_builder=component_builder,
        result=result,
    )


def run_ot_pij_method_with_result(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
    method_name: str,
    component_builder: CostBuilder,
    result: MethodResult,
) -> tuple[MethodResult, TransitionKernels]:
    result.method_metadata["pij_method"] = method_name
    kernels = TransitionKernels(
        kernel_metadata={
            "pij_method": method_name,
            "ot_solver": "pot.sinkhorn_or_internal_sinkhorn",
            "epsilon": float(cfg.pij_entropy_epsilon),
            "unbalanced": bool(cfg.pij_use_unbalanced_ot),
            "unbalanced_mass": float(cfg.pij_unbalanced_mass),
            "cost_metric": cfg.pij_cost_metric,
            "row_stochastic": True,
        }
    )
    for t0, t1 in pairs:
        lower_kernel, lower_meta, lower_diagnostics = _build_kernel(
            result.lower_features[t0],
            result.lower_features[t1],
            _coords_at(result.lower_coords, t0),
            _coords_at(result.lower_coords, t1),
            "lower",
            t0,
            t1,
            cfg,
            component_builder,
        )
        upper_kernel, upper_meta, upper_diagnostics = _build_kernel(
            result.upper_features[t0],
            result.upper_features[t1],
            _coords_at(result.upper_coords, t0),
            _coords_at(result.upper_coords, t1),
            "upper",
            t0,
            t1,
            cfg,
            component_builder,
        )
        kernels.p_lower[(t0, t1)] = lower_kernel
        kernels.p_upper[(t0, t1)] = upper_kernel
        kernels.kernel_diagnostics.setdefault("lower", {})[(t0, t1)] = lower_diagnostics
        kernels.kernel_diagnostics.setdefault("upper", {})[(t0, t1)] = upper_diagnostics
        pair_label = f"{context.time_points[t0]}->{context.time_points[t1]}"
        kernels.kernel_metadata[pair_label] = {
            "lower": lower_meta,
            "upper": upper_meta,
        }
    return result, kernels
