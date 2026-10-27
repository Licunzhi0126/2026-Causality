"""Shared method identity and ordering for formal Core; no model imports."""

METHODS = (
    "complete_combined_coarse_maturity_cci_grn",
    "maturity_cci_grn_two_stage",
    "shr_method_8",
    "shr_method_10",
)
METHOD_LABELS = dict(zip(METHODS, (
    "Single-stage", "Two-stage", "SHR Method 8", "SHR Method 10",
)))


def expected_conditions(scales, pairs) -> int:
    return len(METHODS) * len(scales) * len(pairs)
