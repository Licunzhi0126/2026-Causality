from __future__ import annotations

from typing import Callable

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse import (
    prepare as prepare_complete_combined_coarse,
)
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import (
    prepare as prepare_complete_combined_coarse_maturity_cci_grn,
)
from mignet_ce.coarse_frontends.maturity_cci_grn_two_stage import (
    prepare as prepare_maturity_cci_grn_two_stage,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from mignet_ce.coarse_frontends.two_stage_no_keff import prepare as prepare_no_keff
from mignet_ce.coarse_frontends.two_stage_no_keff_hard_ei import prepare as prepare_hard_ei
from mignet_ce.coarse_frontends.single_stage_cq_only import prepare as prepare_cq_only
from mignet_ce.coarse_frontends.formal_two_stage import prepare as prepare_formal_two_stage

MATURITY_GRN_METHODS = frozenset({
    "formal_two_stage",
    "complete_combined_coarse_maturity_cci_grn", "maturity_cci_grn_two_stage",
    "two_stage_noKeff", "two_stage_noKeff_HardEI", "single_stage_cq_only",
})


COARSE_FRONTEND_REGISTRY: dict[
    str,
    Callable[[CoarseFrontendRequest], PreparedCoarseInput],
] = {
    "formal_two_stage": prepare_formal_two_stage,
    "complete_combined_coarse": prepare_complete_combined_coarse,
    "complete_combined_coarse_maturity_cci_grn": prepare_complete_combined_coarse_maturity_cci_grn,
    "maturity_cci_grn_two_stage": prepare_maturity_cci_grn_two_stage,
    "two_stage_noKeff": prepare_no_keff,
    "two_stage_noKeff_HardEI": prepare_hard_ei,
    "single_stage_cq_only": prepare_cq_only,
}


def prepare_coarse_input(
    method: str,
    request: CoarseFrontendRequest,
) -> PreparedCoarseInput:
    if request.grn_feature_method != "legacy" and method not in MATURITY_GRN_METHODS:
        raise ValueError(
            f"GRN structural method {request.grn_feature_method} is not integrated into coarse frontend {method}; "
            "choose legacy or a supported GRN+maturity frontend."
        )
    if method not in MATURITY_GRN_METHODS and (
        request.cci_feature_method != "shared_nmf" or abs(request.alpha_cci - 0.01) > 1e-12
    ):
        raise ValueError(
            "Custom CCI feature and alpha_cci are only integrated into the maturity+GRN "
            "single-stage and two-stage coarse frontends."
        )
    try:
        frontend = COARSE_FRONTEND_REGISTRY[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown coarse frontend {method!r}; expected one of "
            f"{sorted(COARSE_FRONTEND_REGISTRY)}."
        ) from exc
    return frontend(request)
