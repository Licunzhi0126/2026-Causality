"""SHR Method 10 uses the existing maturity/CCI/GRN scientific input protocol."""
from dataclasses import replace

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import (
    METHOD as DELEGATED_FRONTEND,
    prepare as prepare_maturity_cci_grn,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput

METHOD = "shr_method_10"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    prepared = prepare_maturity_cci_grn(request)
    result = replace(prepared, method=METHOD, provenance={
        **dict(prepared.provenance), "coarse_method": METHOD,
        "frontend_delegate": DELEGATED_FRONTEND,
        "frontend_identity_contract": "value_identical_delegate",
    })
    result.validate()
    return result
