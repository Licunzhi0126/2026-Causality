from __future__ import annotations

"""Method-level coarse-graining metadata shared by CLIs and trainers."""

from dataclasses import dataclass


LEGACY_SINGLE_STAGE = "legacy_single_stage"
DYNAMIC_CLOSURE_TWO_STAGE = "dynamic_closure_two_stage"


@dataclass(frozen=True)
class CoarseMethodSpec:
    name: str
    training_mode: str = LEGACY_SINGLE_STAGE
    frontend: str | None = None
    requires_maturity: bool = False
    requires_cci: bool = True
    requires_grn: bool = False
    default_lambda_dev: float = 0.0
    objective_version: str = "legacy_feature_align_deltaei_v40"


_LEGACY_METHODS = (
    "complete_combined_coarse",
    "complete_combined_coarse_maturity_cci",
    "complete_combined_coarse_maturity_cci_grn",
    "wyt_cg_cci",
    "wyt_cg_cci_regsim",
    "wyt_cg_regsim_v7",
    "wyt_cg_regsim_v9",
)


COARSE_METHOD_SPECS: dict[str, CoarseMethodSpec] = {
    name: CoarseMethodSpec(name=name)
    for name in _LEGACY_METHODS
}
COARSE_METHOD_SPECS.update(
    {
        "complete_combined_coarse": CoarseMethodSpec(
            name="complete_combined_coarse",
            requires_grn=True,
        ),
        "complete_combined_coarse_maturity_cci": CoarseMethodSpec(
            name="complete_combined_coarse_maturity_cci",
            requires_maturity=True,
            default_lambda_dev=0.05,
        ),
        "complete_combined_coarse_maturity_cci_grn": CoarseMethodSpec(
            name="complete_combined_coarse_maturity_cci_grn",
            requires_maturity=True,
            requires_grn=True,
            default_lambda_dev=0.05,
        ),
        "maturity_cci_grn_two_stage": CoarseMethodSpec(
            name="maturity_cci_grn_two_stage",
            training_mode=DYNAMIC_CLOSURE_TWO_STAGE,
            frontend="complete_combined_coarse_maturity_cci_grn",
            requires_maturity=True,
            requires_grn=True,
            default_lambda_dev=0.05,
            objective_version="wyt_dynamic_closure_two_stage_v2",
        ),
    }
)


def get_coarse_method_spec(method: str) -> CoarseMethodSpec:
    """Return a known method spec, defaulting unknown programmatic fixtures to legacy."""

    return COARSE_METHOD_SPECS.get(method, CoarseMethodSpec(name=method))


def training_mode_for_method(method: str) -> str:
    return get_coarse_method_spec(method).training_mode

