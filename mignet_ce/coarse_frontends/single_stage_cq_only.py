"""Single-stage CQ protocol using the shared maturity/CCI/GRN input preparation."""
from dataclasses import replace

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import (
    METHOD as DELEGATED_FRONTEND, prepare as prepare_maturity_cci_grn,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput

METHOD = "single_stage_cq_only"


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    prepared = prepare_maturity_cci_grn(request)
    result = replace(prepared, method=METHOD, provenance={
        **dict(prepared.provenance), "coarse_method": METHOD,
        "frontend_delegate": DELEGATED_FRONTEND,
        "frontend_identity_contract": "value_identical_delegate",
        "keff_policy": "iterative_no_floor", "method_semantics": "cq_only",
    })
    result.validate()
    return result
