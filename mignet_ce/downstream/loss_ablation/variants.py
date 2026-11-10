"""Ordered, cumulative changes to the Formal Core two-stage objective."""

from mignet_ce.downstream.formal_core_analysis.run_pipeline import K_BY_SCALE, PAIRS, SCALES


BASELINE_METHOD = "maturity_cci_grn_two_stage"
TIME_PAIRS = tuple(tuple(pair.split(":")) for pair in PAIRS)
TARGET_K = {scale: int(k) for scale, k in (item.split("=") for item in K_BY_SCALE)}
STAGES = ("stage1", "stage2")

VARIANTS = (
    ("baseline", "Baseline", {}),
    ("step0", "Step 0", {
        "lambda-var": 0.0, "lambda-local": 0.0,
        "lambda-proto": 0.0, "lambda-dead-usage": 0.0,
    }),
    ("step1", "Step 1", {"lambda-sharp": 0.0}),
    ("step2d", "Step 2D", {"lambda-align": 0.0, "lambda-stage1-closure": 0.10}),
    ("step3", "Step 3", {"lambda-within": 0.0, "lambda-inter": 0.0}),
    ("step4a", "Step 4A", {
        "lambda-retain-norm": 0.0, "lambda-retain-floor": 0.0, "lambda-stage2-cq": 1.0,
    }),
)
METHOD_LABELS = tuple(label for _, label, _ in VARIANTS)


def variant_flags() -> dict[str, list[str]]:
    weights: dict[str, float] = {}
    flags = {}
    for name, _, increment in VARIANTS:
        weights.update(increment)
        flags[name] = ["--report-hard-occupancy"]
        for option, value in weights.items():
            flags[name].extend((f"--{option}", str(value)))
    return flags
