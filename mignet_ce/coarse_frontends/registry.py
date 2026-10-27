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
from mignet_ce.coarse_frontends.shr_method_8 import prepare as prepare_shr_method_8
from mignet_ce.coarse_frontends.shr_method_10 import prepare as prepare_shr_method_10

MATURITY_GRN_METHODS = frozenset({
    "complete_combined_coarse_maturity_cci_grn", "maturity_cci_grn_two_stage",
    "shr_method_8", "shr_method_10",
})


COARSE_FRONTEND_REGISTRY: dict[
    str,
    Callable[[CoarseFrontendRequest], PreparedCoarseInput],
] = {
    "complete_combined_coarse": prepare_complete_combined_coarse,
    "complete_combined_coarse_maturity_cci_grn": prepare_complete_combined_coarse_maturity_cci_grn,
    "maturity_cci_grn_two_stage": prepare_maturity_cci_grn_two_stage,
    "shr_method_8": prepare_shr_method_8,
    "shr_method_10": prepare_shr_method_10,
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
