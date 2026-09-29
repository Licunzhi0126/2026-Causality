from __future__ import annotations

from typing import Mapping

import numpy as np

from mignet_ce.utils.progress import emit_progress


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the PIJ CUDA backend.") from exc
    return torch


def _torch_dtype(dtype: str):
    torch = _import_torch()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be one of: float32, float64")


def estimate_gpu_kernel_bytes(n_source: int, n_target: int, dtype: str = "float32", multiplier: int = 5) -> int:
    bytes_per = 4 if dtype == "float32" else 8
    return int(n_source) * int(n_target) * bytes_per * int(multiplier)


def cuda_memory_info() -> tuple[int | None, int | None]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        return None, None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def cuda_has_enough_memory(estimated_bytes: int, safety_ratio: float = 0.80) -> bool:
    free_bytes, _total_bytes = cuda_memory_info()
    if free_bytes is None:
        return False
    return int(estimated_bytes) < int(free_bytes * float(safety_ratio))


def _cuda_available() -> bool:
    try:
        torch = _import_torch()
    except RuntimeError:
        return False
    return bool(torch.cuda.is_available())


def select_pij_backend(cfg, n_source: int, n_target: int) -> str:
    requested = getattr(cfg, "pij_device", "cpu")
    if requested == "cpu":
        return "cpu"

    entries = int(n_source) * int(n_target)
    dtype = getattr(cfg, "pij_gpu_dtype", "float32")
    fallback_cpu = bool(getattr(cfg, "pij_gpu_fallback_cpu", True))

    if requested == "auto" and entries < int(getattr(cfg, "pij_gpu_min_entries", 0)):
        return "cpu"

    if not _cuda_available():
        emit_progress(
            "pij_gpu_memory_fallback",
            phase="pij",
            status="fallback",
            backend="cuda",
            cuda_available=False,
            fallback_reason="cuda_unavailable",
        )
        if fallback_cpu or requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested for PIJ but is not available.")

    estimated = estimate_gpu_kernel_bytes(n_source, n_target, dtype=dtype)
    free_bytes, total_bytes = cuda_memory_info()
    emit_progress(
        "pij_gpu_memory_check",
        phase="pij",
        backend="cuda",
        estimated_bytes=estimated,
        cuda_available=True,
        extra={"free_bytes": free_bytes, "total_bytes": total_bytes},
    )
    if not cuda_has_enough_memory(estimated):
        emit_progress(
            "pij_gpu_memory_fallback",
            phase="pij",
            status="fallback",
            backend="cuda",
            estimated_bytes=estimated,
            cuda_available=True,
            fallback_reason="estimated_bytes_exceeds_free_memory",
            extra={"free_bytes": free_bytes, "total_bytes": total_bytes},
        )
        if fallback_cpu or requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested for PIJ but estimated kernel memory exceeds available VRAM.")
    return "cuda"


def _as_cuda_tensor(value, *, dtype: str, device: str):
    torch = _import_torch()
    torch_dtype = _torch_dtype(dtype)
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch_dtype)
    return torch.as_tensor(value, dtype=torch_dtype, device=device)


def _normalize_cost_tensor(cost):
    torch = _import_torch()
    if cost.ndim != 2:
        raise ValueError(f"Expected a 2D cost matrix, got shape {tuple(cost.shape)}.")
    if cost.numel() == 0:
        return cost.clone()
    finite = torch.isfinite(cost)
    if not bool(finite.any().item()):
        return torch.zeros_like(cost)
    finite_values = cost[finite]
    min_value = torch.min(finite_values)
    max_value = torch.max(finite_values)
    denom = max_value - min_value
    if bool((denom <= 1e-12).item()):
        return torch.zeros_like(cost)
    filled = torch.where(finite, cost, max_value)
    return torch.clamp((filled - min_value) / denom, 0.0, 1.0)


def _clean_cost_tensor(cost):
    torch = _import_torch()
    if cost.ndim != 2:
        raise ValueError(f"Expected a 2D cost matrix, got shape {tuple(cost.shape)}.")
    if cost.numel() == 0:
        return cost.clone()
    finite = torch.isfinite(cost)
    if not bool(finite.any().item()):
        return torch.zeros_like(cost)
    finite_values = cost[finite]
    min_value = torch.min(finite_values)
    max_value = torch.max(finite_values)
    filled = torch.where(finite, cost, max_value)
    return torch.clamp(filled - min_value, min=0.0)


def _component_summary_tensor(name: str, tensor, weight: float, enabled: bool) -> dict[str, object]:
    torch = _import_torch()
    finite = torch.isfinite(tensor)
    warnings: list[str] = []
    if tensor.numel() == 0 or not bool(finite.any().item()):
        warnings.append("all_non_finite")
        min_value = max_value = mean_value = None
    else:
        values = tensor[finite]
        min_value = float(torch.min(values).detach().cpu().item())
        max_value = float(torch.max(values).detach().cpu().item())
        mean_value = float(torch.mean(values).detach().cpu().item())
        if max_value - min_value <= 1e-12:
            warnings.append("constant_cost")
    return {
        "component": name,
        "weight": float(weight),
        "enabled": bool(enabled),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "warnings": warnings,
    }


def pairwise_expression_cosine_cost_gpu(
    source: np.ndarray,
    target: np.ndarray,
    *,
    dtype: str = "float32",
    device: str = "cuda",
    eps: float = 1e-12,
):
    torch = _import_torch()
    source_tensor = _as_cuda_tensor(source, dtype=dtype, device=device)
    target_tensor = _as_cuda_tensor(target, dtype=dtype, device=device)
    if source_tensor.ndim != 2 or target_tensor.ndim != 2:
        raise ValueError(f"source and target must be 2D arrays, got {tuple(source_tensor.shape)} and {tuple(target_tensor.shape)}.")
    if source_tensor.shape[1] != target_tensor.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_tensor.shape[1]} != {target_tensor.shape[1]}.")
    if source_tensor.shape[0] == 0 or target_tensor.shape[0] == 0:
        return torch.zeros((source_tensor.shape[0], target_tensor.shape[0]), dtype=source_tensor.dtype, device=source_tensor.device)

    source_norm = torch.linalg.norm(source_tensor, dim=1, keepdim=True)
    target_norm = torch.linalg.norm(target_tensor, dim=1, keepdim=True)
    source_unit = torch.where(source_norm > eps, source_tensor / source_norm.clamp_min(eps), torch.zeros_like(source_tensor))
    target_unit = torch.where(target_norm > eps, target_tensor / target_norm.clamp_min(eps), torch.zeros_like(target_tensor))
    cost = 1.0 - source_unit @ target_unit.T
    return _normalize_cost_tensor(cost)


def combine_components_on_gpu(
    components: Mapping[str, object],
    weights: Mapping[str, float],
    *,
    dtype: str,
    device: str,
):
    if not components:
        raise ValueError("At least one cost component is required.")
    shape = None
    combined = None
    summaries: dict[str, object] = {}
    total_weight = 0.0
    for name, value in components.items():
        tensor = _as_cuda_tensor(value, dtype=dtype, device=device)
        if tensor.ndim != 2:
            raise ValueError(f"Cost component {name!r} must be 2D, got shape {tuple(tensor.shape)}.")
        if shape is None:
            shape = tuple(tensor.shape)
            combined = tensor.new_zeros(shape)
        elif tuple(tensor.shape) != shape:
            raise ValueError(f"Cost component {name!r} has shape {tuple(tensor.shape)}; expected {shape}.")
        weight = float(weights.get(name, 0.0))
        if weight < 0:
            raise ValueError(f"Cost weight for {name!r} must be nonnegative.")
        enabled = weight > 0
        summaries[name] = _component_summary_tensor(name, tensor, weight, enabled)
        if enabled:
            combined = combined + weight * _normalize_cost_tensor(tensor)
            total_weight += weight
    if combined is None or shape is None:
        raise ValueError("At least one cost component is required.")
    if total_weight <= 0:
        raise ValueError("At least one cost component must have positive weight.")
    combined = combined / total_weight
    summary = {
        "components": summaries,
        "total_weight": float(total_weight),
        "combined": _component_summary_tensor("combined", combined, 1.0, True),
    }
    return combined, summary


def build_entropic_ot_kernel_gpu(
    cost,
    *,
    epsilon: float,
    max_iter: int,
    tol: float = 1e-9,
    unbalanced: bool = False,
    mass_reg: float = 1.0,
) -> np.ndarray:
    torch = _import_torch()
    if not torch.is_tensor(cost) or not cost.is_cuda:
        raise ValueError("cost must be a CUDA tensor")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if mass_reg <= 0:
        raise ValueError("mass_reg must be positive.")

    with torch.no_grad():
        clean_cost = _clean_cost_tensor(cost)
        n_source, n_target = clean_cost.shape
        if n_source == 0 or n_target == 0:
            return np.zeros((int(n_source), int(n_target)), dtype=float)

        dtype = clean_cost.dtype
        device = clean_cost.device
        a = torch.full((n_source,), 1.0 / int(n_source), dtype=dtype, device=device)
        b = torch.full((n_target,), 1.0 / int(n_target), dtype=dtype, device=device)
        tiny = torch.finfo(dtype).tiny
        denom_eps = max(float(tiny), 1e-30)
        kernel = torch.exp(-clean_cost / float(epsilon))
        kernel = torch.clamp(kernel, min=tiny)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        if unbalanced:
            power = float(mass_reg) / (float(mass_reg) + float(epsilon))
            for _ in range(max(0, int(max_iter))):
                v = torch.pow(b / (kernel.T @ u + denom_eps), power)
                u = torch.pow(a / (kernel @ v + denom_eps), power)
        else:
            for iteration in range(max(0, int(max_iter))):
                u_prev = u
                u = a / (kernel @ v + denom_eps)
                v = b / (kernel.T @ u + denom_eps)
                if iteration % 25 == 0:
                    err = torch.max(torch.abs(u - u_prev)).detach().cpu().item()
                    if err < tol:
                        break

        transport = (u[:, None] * kernel) * v[None, :]
        row_sums = transport.sum(dim=1, keepdim=True)
        normalized = torch.divide(
            transport,
            row_sums.clamp_min(denom_eps),
        )
        zero_rows = torch.squeeze(row_sums <= denom_eps, dim=1)
        if bool(zero_rows.any().item()):
            normalized[zero_rows] = 1.0 / int(n_target)
        out = normalized.detach().cpu().numpy()
        del clean_cost, kernel, transport, normalized
        return out


def component_to_numpy(value) -> np.ndarray:
    try:
        torch = _import_torch()
    except RuntimeError:
        return np.asarray(value, dtype=float)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)
