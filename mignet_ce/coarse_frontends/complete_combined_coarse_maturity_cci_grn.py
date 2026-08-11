from __future__ import annotations

"""Complete-combined coarse graining with registered LightCCI-GRN and maturity."""

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.complete_combined_maturity import (
    prepare_complete_combined_maturity,
)

METHOD = "complete_combined_coarse_maturity_cci_grn"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    return prepare_complete_combined_maturity(
        request,
        method=METHOD,
        network_method="light_cci_grn",
    )
