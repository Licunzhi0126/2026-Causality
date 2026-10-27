from __future__ import annotations

from typing import Dict, Sequence, Type

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PijMethod, TimePair, TransitionKernels
from mignet_ce.pij.legacy.joint_nmf import JointNMFPijMethod
from mignet_ce.pij.legacy.laplacian import LaplacianPijMethod
from mignet_ce.pij.legacy.slat import SLATPijMethod
from mignet_ce.pij.legacy.three_dot import ThreeDotPijMethod
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import CompareNGKlGRNAnchorV5PijMethod
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import CompareNGKlSinkhornGRNAnchorV7PijMethod
from mignet_ce.pij.compare.NG_KLot import NGKLotPijMethod
from mignet_ce.pij.wyt_single_kl import WYTSingleKLPijMethod


# Frozen PIJ registry: retained core baselines and methods still required by the
# current coarse-graining / Formal Core Analysis code. Historical wrappers are
# no longer discoverable through the official registry.
PIJ_METHOD_REGISTRY: Dict[str, Type[PijMethod]] = {
    "joint_nmf": JointNMFPijMethod,
    "laplacian": LaplacianPijMethod,
    "3dot": ThreeDotPijMethod,
    "slat": SLATPijMethod,
    "compare_N_kl": CompareNKlPijMethod,
    "compare_NG_kl_grnanchor_v5": CompareNGKlGRNAnchorV5PijMethod,
    "compare_NG_kl_sinkhorn_grnanchor_v7": CompareNGKlSinkhornGRNAnchorV7PijMethod,
    "NG_KLot": NGKLotPijMethod,
    "wyt_single_kl": WYTSingleKLPijMethod,
}


def get_pij_method(pij_method: str) -> PijMethod:
    try:
        method_cls = PIJ_METHOD_REGISTRY[pij_method]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported pij_method {pij_method!r}. Expected one of {sorted(PIJ_METHOD_REGISTRY)}."
        ) from exc
    return method_cls()


def build_method_result_and_kernels(
    context: NetworkContext,
    cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
) -> tuple[MethodResult, TransitionKernels | None]:
    method = get_pij_method(cfg.effective_pij_method())
    return method.run(context, cfg, pairs)
