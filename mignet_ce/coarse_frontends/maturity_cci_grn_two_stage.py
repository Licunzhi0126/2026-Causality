from __future__ import annotations

"""Two-stage method with the exact maturity+CCI+GRN frontend."""

from dataclasses import replace

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import (
    METHOD as DELEGATED_FRONTEND,
    prepare as prepare_maturity_cci_grn,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput


METHOD = "maturity_cci_grn_two_stage"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    """Delegate all mathematical preparation to the existing maturity frontend."""

    prepared = prepare_maturity_cci_grn(request)
    result = replace(
        prepared,
        method=METHOD,
        provenance={
            **dict(prepared.provenance),
            "coarse_method": METHOD,
            "frontend_delegate": DELEGATED_FRONTEND,
            "frontend_identity_contract": "value_identical_delegate",
        },
    )
    result.validate()
    return result

