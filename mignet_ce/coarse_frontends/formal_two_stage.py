"""Formal two-stage frontend using the native maturity, CCI and GRN inputs."""

from dataclasses import replace

from ._common import CoarseFrontendRequest
from .maturity_cci_grn_two_stage import prepare as prepare_two_stage
from mignet_ce.representations.coarse_input import PreparedCoarseInput


def prepare(request: CoarseFrontendRequest) -> PreparedCoarseInput:
    prepared = prepare_two_stage(request)
    result = replace(prepared, method="formal_two_stage", provenance={
        **dict(prepared.provenance),
        "coarse_method": "formal_two_stage",
        "loss_protocol": "formal_two_stage_v1",
    })
    result.validate()
    return result
