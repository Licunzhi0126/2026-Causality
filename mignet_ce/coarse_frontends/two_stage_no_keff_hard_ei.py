"""No-Keff EI-first protocol with a 100% EI retention target and native fallback."""
from dataclasses import replace

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.two_stage_no_keff import prepare as prepare_no_keff
from mignet_ce.representations.coarse_input import PreparedCoarseInput

METHOD = "two_stage_noKeff_HardEI"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    prepared = prepare_no_keff(request)
    result = replace(prepared, method=METHOD, provenance={
        **dict(prepared.provenance), "coarse_method": METHOD,
        "default_ei_retain_ratio": 1.0,
    })
    result.validate()
    return result
