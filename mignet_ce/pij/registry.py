from __future__ import annotations

from typing import Dict, Sequence, Type

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.joint_nmf import JointNMFPijMethod
from mignet_ce.pij.laplacian import LaplacianPijMethod
from mignet_ce.pij.slat import SLATPijMethod
from mignet_ce.pij.three_dot import ThreeDotPijMethod
from mignet_ce.pij.base import MethodResult, PijMethod, TimePair, TransitionKernels


PIJ_METHOD_REGISTRY: Dict[str, Type[PijMethod]] = {
    "joint_nmf": JointNMFPijMethod,
    "laplacian": LaplacianPijMethod,
    "3dot": ThreeDotPijMethod,
    "slat": SLATPijMethod,
}


def get_pij_method(pij_method: str) -> PijMethod:
    try:
        method_cls = PIJ_METHOD_REGISTRY[pij_method]
    except KeyError as exc:
        raise ValueError(f"Unsupported pij_method {pij_method!r}. Expected one of {sorted(PIJ_METHOD_REGISTRY)}.") from exc
    return method_cls()


def build_method_result_and_kernels(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
) -> tuple[MethodResult, TransitionKernels | None]:
    method = get_pij_method(cfg.effective_pij_method())
    return method.run(context, cfg, pairs)
