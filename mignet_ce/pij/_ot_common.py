from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Mapping, Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.representations.graph_features import build_graph_feature_result
from mignet_ce.transition.cost_components import combine_costs
from mignet_ce.transition.ot import build_entropic_ot_kernel


DEFAULT_PIJ_MEMORY_FRACTION = 0.5

CostBuilder = Callable[
    [np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str, int, int],
    tuple[Mapping[str, np.ndarray], Mapping[str, float]]
    | tuple[Mapping[str, np.ndarray], Mapping[str, float], Mapping[str, object]],
]


def estimate_ot_task_bytes(n_source: int, n_target: int, multiplier: float = 6.0) -> int:
    return int(max(0, n_source) * max(0, n_target) * 8 * float(multiplier))


def _available_memory_bytes() -> int | None:
    try:
        import os

        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * available_pages)
    except (OSError, ValueError, AttributeError):
        pass

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except Exception:
        return None
    return None


def memory_limited_workers(
    candidate_workers: int,
    estimated_task_bytes: Sequence[int],
    memory_fraction: float,
) -> int:
    candidate = max(1, int(candidate_workers))
    if candidate <= 1 or not estimated_task_bytes:
        return 1
    largest_task = max(int(value) for value in estimated_task_bytes)
    if largest_task <= 0:
        return candidate
    available = _available_memory_bytes()
    if available is None:
        return candidate
    memory_budget = int(max(0.0, float(memory_fraction)) * available)
    if memory_budget <= 0:
        return 1
    return max(1, min(candidate, memory_budget // largest_task))


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
    keep_diagnostics: bool = True,
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
    diagnostics: dict[str, np.ndarray] = {}
    if keep_diagnostics:
        diagnostics = {f"{name}_cost": np.asarray(cost, dtype=float) for name, cost in components.items()}
        diagnostics["main_cost"] = np.asarray(cost, dtype=float)
    return kernel, metadata, diagnostics


def _run_with_single_blas_thread(fn):
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return fn()
    with threadpool_limits(limits=1):
        return fn()


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
    keep_diagnostics = bool(cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0)
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

    tasks = []
    for t0, t1 in pairs:
        lower_source = result.lower_features[t0]
        lower_target = result.lower_features[t1]
        upper_source = result.upper_features[t0]
        upper_target = result.upper_features[t1]
        tasks.append(
            {
                "order": len(tasks),
                "space": "lower",
                "t0": t0,
                "t1": t1,
                "source_features": lower_source,
                "target_features": lower_target,
                "source_coords": _coords_at(result.lower_coords, t0),
                "target_coords": _coords_at(result.lower_coords, t1),
                "estimated_bytes": estimate_ot_task_bytes(lower_source.shape[0], lower_target.shape[0]),
            }
        )
        tasks.append(
            {
                "order": len(tasks),
                "space": "upper",
                "t0": t0,
                "t1": t1,
                "source_features": upper_source,
                "target_features": upper_target,
                "source_coords": _coords_at(result.upper_coords, t0),
                "target_coords": _coords_at(result.upper_coords, t1),
                "estimated_bytes": estimate_ot_task_bytes(upper_source.shape[0], upper_target.shape[0]),
            }
        )

    def build_task(task):
        def _build():
            kernel, metadata, diagnostics = _build_kernel(
                task["source_features"],
                task["target_features"],
                task["source_coords"],
                task["target_coords"],
                task["space"],
                task["t0"],
                task["t1"],
                cfg,
                component_builder,
                keep_diagnostics=keep_diagnostics,
            )
            return task["order"], task["space"], task["t0"], task["t1"], kernel, metadata, diagnostics

        return _run_with_single_blas_thread(_build)

    candidate_workers = min(int(cfg.max_workers), len(tasks)) if tasks else 1
    actual_workers = memory_limited_workers(
        candidate_workers,
        [int(task["estimated_bytes"]) for task in tasks],
        getattr(cfg, "pij_memory_fraction", DEFAULT_PIJ_MEMORY_FRACTION),
    )
    if actual_workers <= 1:
        built_results = [build_task(task) for task in tasks]
    else:
        built_results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_order = {
                pool.submit(build_task, task): int(task["order"])
                for task in tasks
            }
            for future in as_completed(future_to_order):
                built_results[future_to_order[future]] = future.result()
        built_results = [result for result in built_results if result is not None]

    by_key = {
        (space, t0, t1): (kernel, metadata, diagnostics)
        for _order, space, t0, t1, kernel, metadata, diagnostics in built_results
    }

    for t0, t1 in pairs:
        lower_kernel, lower_meta, lower_diagnostics = by_key[("lower", t0, t1)]
        upper_kernel, upper_meta, upper_diagnostics = by_key[("upper", t0, t1)]
        kernels.p_lower[(t0, t1)] = lower_kernel
        kernels.p_upper[(t0, t1)] = upper_kernel
        if lower_diagnostics:
            kernels.kernel_diagnostics.setdefault("lower", {})[(t0, t1)] = lower_diagnostics
        if upper_diagnostics:
            kernels.kernel_diagnostics.setdefault("upper", {})[(t0, t1)] = upper_diagnostics
        pair_label = f"{context.time_points[t0]}->{context.time_points[t1]}"
        kernels.kernel_metadata[pair_label] = {
            "lower": lower_meta,
            "upper": upper_meta,
        }
    return result, kernels
