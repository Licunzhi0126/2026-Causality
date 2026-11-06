"""Shared method identity and ordering for formal Core; no model imports."""

METHODS = (
    "complete_combined_coarse_maturity_cci_grn",
    "maturity_cci_grn_two_stage",
)
METHOD_LABELS = dict(zip(METHODS, (
    "Single-stage", "Two-stage",
)))


def expected_conditions(scales, pairs) -> int:
    return len(METHODS) * len(scales) * len(pairs)
