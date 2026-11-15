"""Shared method identity and ordering for formal Core; no model imports."""

LEGACY_METHODS = (
    "complete_combined_coarse_maturity_cci_grn",
    "maturity_cci_grn_two_stage",
    "two_stage_noKeff", "two_stage_noKeff_HardEI", "single_stage_cq_only",
)
METHOD_LABELS = dict(zip(LEGACY_METHODS, (
    "Single-stage", "Two-stage",
    "two_stage_noKeff", "two_stage_noKeff_HardEI", "single_stage_cq_only",
)))
METHODS = ("formal_two_stage",)
METHOD_LABELS["formal_two_stage"] = "Formal two-stage"
ORGANS = ("lung", "brain", "heart")
ORGAN_LABELS = {"lung": "Lung", "brain": "Brain", "heart": "Heart"}


def expected_conditions(scales, pairs, organs=ORGANS) -> int:
    return len(METHODS) * len(scales) * len(pairs) * len(organs)
