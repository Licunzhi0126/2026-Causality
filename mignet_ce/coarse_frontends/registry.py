from __future__ import annotations

from typing import Callable

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse import (
    prepare as prepare_complete_combined_coarse,
)
from mignet_ce.coarse_frontends.wyt_cg_cci import prepare as prepare_wyt_cg_cci
from mignet_ce.coarse_frontends.wyt_cg_cci_regsim import prepare as prepare_wyt_cg_cci_regsim
from mignet_ce.coarse_frontends.wyt_cg_regsim_v7 import prepare as prepare_wyt_cg_regsim_v7
from mignet_ce.coarse_frontends.wyt_cg_regsim_v9 import prepare as prepare_wyt_cg_regsim_v9
from mignet_ce.representations.coarse_input import PreparedCoarseInput


COARSE_FRONTEND_REGISTRY: dict[
    str,
    Callable[[CoarseFrontendRequest], PreparedCoarseInput],
] = {
    "complete_combined_coarse": prepare_complete_combined_coarse,
    "wyt_cg_cci": prepare_wyt_cg_cci,
    "wyt_cg_cci_regsim": prepare_wyt_cg_cci_regsim,
    "wyt_cg_regsim_v7": prepare_wyt_cg_regsim_v7,
    "wyt_cg_regsim_v9": prepare_wyt_cg_regsim_v9,
}


def prepare_coarse_input(
    method: str,
    request: CoarseFrontendRequest,
) -> PreparedCoarseInput:
    try:
        frontend = COARSE_FRONTEND_REGISTRY[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown coarse frontend {method!r}; expected one of "
            f"{sorted(COARSE_FRONTEND_REGISTRY)}."
        ) from exc
    return frontend(request)
