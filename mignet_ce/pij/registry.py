from __future__ import annotations

from typing import Dict, Sequence, Type

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.development_ot import DevelopmentOTPijMethod
from mignet_ce.pij.energy_ot import EnergyOTPijMethod
from mignet_ce.pij.energy_entropy_ot import EnergyEntropyOTPijMethod
from mignet_ce.pij.expr_ot import ExprOTPijMethod
from mignet_ce.pij.expr_pseudotime_sr_energy_spatial_ot import ExprPseudotimeSREnergySpatialOTPijMethod
from mignet_ce.pij.expr_pseudotime_sr_energy_ot import ExprPseudotimeSREnergyOTPijMethod
from mignet_ce.pij.expr_pseudotime_sr_ot import ExprPseudotimeSROTPijMethod
from mignet_ce.pij.expr_pseudotime_sr_spatial_ot import ExprPseudotimeSRSpatialOTPijMethod
from mignet_ce.pij.joint_nmf import JointNMFPijMethod
from mignet_ce.pij.laplacian import LaplacianPijMethod
from mignet_ce.pij.pseudotime_expression_ot import PseudotimeExpressionOTPijMethod
from mignet_ce.pij.pseudotime_ot import PseudotimeOTPijMethod
from mignet_ce.pij.pseudotime_spatial_ot import PseudotimeSpatialOTPijMethod
from mignet_ce.pij.pure_expression_ot import PureExpressionOTPijMethod
from mignet_ce.pij.slat import SLATPijMethod
from mignet_ce.pij.spatial_ot import SpatialOTPijMethod
from mignet_ce.pij.sr_expression_ot import SRExpressionOTPijMethod
from mignet_ce.pij.sr_ot import SROTPijMethod
from mignet_ce.pij.sr_spatial_ot import SRSpatialOTPijMethod
from mignet_ce.pij.three_dot import ThreeDotPijMethod
from mignet_ce.pij.velocity_ot import VelocityOTPijMethod
from mignet_ce.pij.base import MethodResult, PijMethod, TimePair, TransitionKernels


PIJ_METHOD_REGISTRY: Dict[str, Type[PijMethod]] = {
    "joint_nmf": JointNMFPijMethod,
    "laplacian": LaplacianPijMethod,
    "3dot": ThreeDotPijMethod,
    "slat": SLATPijMethod,
    "expr_ot": ExprOTPijMethod,
    "pure_expression_ot": PureExpressionOTPijMethod,
    "energy_ot": EnergyOTPijMethod,
    "energy_entropy_ot": EnergyEntropyOTPijMethod,
    "pseudotime_ot": PseudotimeOTPijMethod,
    "sr_ot": SROTPijMethod,
    "spatial_ot": SpatialOTPijMethod,
    "sr_spatial_ot": SRSpatialOTPijMethod,
    "pseudotime_spatial_ot": PseudotimeSpatialOTPijMethod,
    "sr_expression_ot": SRExpressionOTPijMethod,
    "pseudotime_expression_ot": PseudotimeExpressionOTPijMethod,
    "expr_pseudotime_sr_ot": ExprPseudotimeSROTPijMethod,
    "expr_pseudotime_sr_spatial_ot": ExprPseudotimeSRSpatialOTPijMethod,
    "expr_pseudotime_sr_energy_ot": ExprPseudotimeSREnergyOTPijMethod,
    "expr_pseudotime_sr_energy_spatial_ot": ExprPseudotimeSREnergySpatialOTPijMethod,
    "velocity_ot": VelocityOTPijMethod,
    "development_ot": DevelopmentOTPijMethod,
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
