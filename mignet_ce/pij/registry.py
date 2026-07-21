from __future__ import annotations

from typing import Dict, Sequence, Type

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PijMethod, TimePair, TransitionKernels
from mignet_ce.pij.compare.compare_E_cos import CompareECosPijMethod
from mignet_ce.pij.compare.compare_E_kl import CompareEKlPijMethod
from mignet_ce.pij.compare.compare_E_L_cos import CompareELCosPijMethod
from mignet_ce.pij.compare.compare_E_L_kl import CompareELKlPijMethod
from mignet_ce.pij.compare.compare_E_L_sot import CompareELSotPijMethod
from mignet_ce.pij.compare.compare_E_N_cos import CompareENCosPijMethod
from mignet_ce.pij.compare.compare_E_N_kl import CompareENKlPijMethod
from mignet_ce.pij.compare.compare_E_N_sot import CompareENSotPijMethod
from mignet_ce.pij.compare.compare_E_Sr_cos import CompareESrCosPijMethod
from mignet_ce.pij.compare.compare_E_Sr_kl import CompareESrKlPijMethod
from mignet_ce.pij.compare.compare_E_Sr_sot import CompareESrSotPijMethod
from mignet_ce.pij.compare.compare_E_sot import CompareESotPijMethod
from mignet_ce.pij.compare.compare_E_euc_sot import CompareEEucSotPijMethod
from mignet_ce.pij.compare.compare_E_Sr_costmix_cos_sot import CompareESrCostMixCosSotPijMethod
from mignet_ce.pij.compare.compare_E_Sr_costmix_euc_sot import CompareESrCostMixEucSotPijMethod
from mignet_ce.pij.compare.compare_L_cos import CompareLCosPijMethod
from mignet_ce.pij.compare.compare_L_kl import CompareLKlPijMethod
from mignet_ce.pij.compare.compare_L_Sr_cos import CompareLSrCosPijMethod
from mignet_ce.pij.compare.compare_L_Sr_kl import CompareLSrKlPijMethod
from mignet_ce.pij.compare.compare_L_Sr_sot import CompareLSrSotPijMethod
from mignet_ce.pij.compare.compare_L_sot import CompareLSotPijMethod
from mignet_ce.pij.compare.compare_L_euc_sot import CompareLEucSotPijMethod
from mignet_ce.pij.compare.compare_L_E_costmix_cos_sot import CompareLECostMixCosSotPijMethod
from mignet_ce.pij.compare.compare_L_E_costmix_euc_sot import CompareLECostMixEucSotPijMethod
from mignet_ce.pij.compare.compare_L_E_Sr_costmix_cos_sot import CompareLESrCostMixCosSotPijMethod
from mignet_ce.pij.compare.compare_L_E_Sr_costmix_euc_sot import CompareLESrCostMixEucSotPijMethod
from mignet_ce.pij.compare.compare_L_Sr_costmix_cos_sot import CompareLSrCostMixCosSotPijMethod
from mignet_ce.pij.compare.compare_L_Sr_costmix_euc_sot import CompareLSrCostMixEucSotPijMethod
from mignet_ce.pij.compare.compare_main_lap_sr_spatial_sot import CompareMainLapSrSpatialSotPijMethod
from mignet_ce.pij.compare.compare_N_cos import CompareNCosPijMethod
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod
from mignet_ce.pij.compare.compare_NG_kl_splitbeta_v1 import CompareNGKlSplitBetaV1PijMethod
from mignet_ce.pij.compare.compare_Ncomp_Gcos_v2 import CompareNCompGCosV2PijMethod
from mignet_ce.pij.compare.compare_Nshape_Gcos_v3 import CompareNShapeGCosV3PijMethod
from mignet_ce.pij.compare.compare_N_L_cos import CompareNLCosPijMethod
from mignet_ce.pij.compare.compare_N_L_kl import CompareNLKlPijMethod
from mignet_ce.pij.compare.compare_N_L_sot import CompareNLSotPijMethod
from mignet_ce.pij.compare.compare_N_Sr_cos import CompareNSrCosPijMethod
from mignet_ce.pij.compare.compare_N_Sr_kl import CompareNSrKlPijMethod
from mignet_ce.pij.compare.compare_N_Sr_sot import CompareNSrSotPijMethod
from mignet_ce.pij.compare.compare_N_sot import CompareNSotPijMethod
from mignet_ce.pij.compare.compare_Sr_cos import CompareSrCosPijMethod
from mignet_ce.pij.compare.compare_Sr_kl import CompareSrKlPijMethod
from mignet_ce.pij.compare.compare_Sr_sot import CompareSrSotPijMethod
from mignet_ce.pij.legacy.development_ot import DevelopmentOTPijMethod
from mignet_ce.pij.legacy.energy_entropy_ot import EnergyEntropyOTPijMethod
from mignet_ce.pij.legacy.energy_ot import EnergyOTPijMethod
from mignet_ce.pij.legacy.expr_ot import ExprOTPijMethod
from mignet_ce.pij.legacy.expr_pseudotime_sr_energy_ot import ExprPseudotimeSREnergyOTPijMethod
from mignet_ce.pij.legacy.expr_pseudotime_sr_energy_spatial_ot import ExprPseudotimeSREnergySpatialOTPijMethod
from mignet_ce.pij.legacy.expr_pseudotime_sr_ot import ExprPseudotimeSROTPijMethod
from mignet_ce.pij.legacy.expr_pseudotime_sr_spatial_ot import ExprPseudotimeSRSpatialOTPijMethod
from mignet_ce.pij.legacy.joint_nmf import JointNMFPijMethod
from mignet_ce.pij.legacy.laplacian import LaplacianPijMethod
from mignet_ce.pij.legacy.pseudotime_expression_ot import PseudotimeExpressionOTPijMethod
from mignet_ce.pij.legacy.pseudotime_ot import PseudotimeOTPijMethod
from mignet_ce.pij.legacy.pseudotime_spatial_ot import PseudotimeSpatialOTPijMethod
from mignet_ce.pij.legacy.pure_expression_ot import PureExpressionOTPijMethod
from mignet_ce.pij.legacy.slat import SLATPijMethod
from mignet_ce.pij.legacy.spatial_ot import SpatialOTPijMethod
from mignet_ce.pij.legacy.sr_expression_ot import SRExpressionOTPijMethod
from mignet_ce.pij.legacy.sr_ot import SROTPijMethod
from mignet_ce.pij.legacy.sr_spatial_ot import SRSpatialOTPijMethod
from mignet_ce.pij.legacy.three_dot import ThreeDotPijMethod
from mignet_ce.pij.legacy.velocity_ot import VelocityOTPijMethod


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
    "compare_E_cos": CompareECosPijMethod,
    "compare_E_kl": CompareEKlPijMethod,
    "compare_E_sot": CompareESotPijMethod,
    "compare_E_euc_sot": CompareEEucSotPijMethod,
    "compare_N_cos": CompareNCosPijMethod,
    "compare_N_kl": CompareNKlPijMethod,
    "compare_NG_kl_splitbeta_v1": CompareNGKlSplitBetaV1PijMethod,
    "compare_Ncomp_Gcos_v2": CompareNCompGCosV2PijMethod,
    "compare_Nshape_Gcos_v3": CompareNShapeGCosV3PijMethod,
    "compare_N_sot": CompareNSotPijMethod,
    "compare_L_cos": CompareLCosPijMethod,
    "compare_L_kl": CompareLKlPijMethod,
    "compare_L_sot": CompareLSotPijMethod,
    "compare_L_euc_sot": CompareLEucSotPijMethod,
    "compare_Sr_cos": CompareSrCosPijMethod,
    "compare_Sr_kl": CompareSrKlPijMethod,
    "compare_Sr_sot": CompareSrSotPijMethod,
    "compare_E_N_cos": CompareENCosPijMethod,
    "compare_E_N_kl": CompareENKlPijMethod,
    "compare_E_N_sot": CompareENSotPijMethod,
    "compare_E_L_cos": CompareELCosPijMethod,
    "compare_E_L_kl": CompareELKlPijMethod,
    "compare_E_L_sot": CompareELSotPijMethod,
    "compare_E_Sr_cos": CompareESrCosPijMethod,
    "compare_E_Sr_kl": CompareESrKlPijMethod,
    "compare_E_Sr_sot": CompareESrSotPijMethod,
    "compare_N_L_cos": CompareNLCosPijMethod,
    "compare_N_L_kl": CompareNLKlPijMethod,
    "compare_N_L_sot": CompareNLSotPijMethod,
    "compare_N_Sr_cos": CompareNSrCosPijMethod,
    "compare_N_Sr_kl": CompareNSrKlPijMethod,
    "compare_N_Sr_sot": CompareNSrSotPijMethod,
    "compare_L_Sr_cos": CompareLSrCosPijMethod,
    "compare_L_Sr_kl": CompareLSrKlPijMethod,
    "compare_L_Sr_sot": CompareLSrSotPijMethod,
    "compare_L_E_costmix_cos_sot": CompareLECostMixCosSotPijMethod,
    "compare_L_E_costmix_euc_sot": CompareLECostMixEucSotPijMethod,
    "compare_L_Sr_costmix_cos_sot": CompareLSrCostMixCosSotPijMethod,
    "compare_L_Sr_costmix_euc_sot": CompareLSrCostMixEucSotPijMethod,
    "compare_L_E_Sr_costmix_cos_sot": CompareLESrCostMixCosSotPijMethod,
    "compare_L_E_Sr_costmix_euc_sot": CompareLESrCostMixEucSotPijMethod,
    "compare_E_Sr_costmix_cos_sot": CompareESrCostMixCosSotPijMethod,
    "compare_E_Sr_costmix_euc_sot": CompareESrCostMixEucSotPijMethod,
    "compare_main_lap_sr_spatial_sot": CompareMainLapSrSpatialSotPijMethod,
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
