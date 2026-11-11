"""Formal two-stage frontend using the native maturity, CCI and GRN inputs."""

from dataclasses import replace

from ._common import CoarseFrontendRequest
from .method_specs import FORMAL_TWO_STAGE_PROTOCOL
from .maturity_cci_grn_two_stage import prepare as prepare_two_stage
from mignet_ce.representations.coarse_input import PreparedCoarseInput


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    prepared = prepare_two_stage(request)
    result = replace(prepared, method="formal_two_stage", provenance={
        **dict(prepared.provenance),
        "coarse_method": "formal_two_stage",
        "loss_protocol": FORMAL_TWO_STAGE_PROTOCOL,
    })
    result.validate()
    return result
