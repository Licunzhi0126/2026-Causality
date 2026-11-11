"""Independent branches of the registered formal_two_stage baseline."""

from dataclasses import dataclass

from ..variants import TIME_PAIRS


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stage1_group: str
    flags: tuple[str, ...] = ()


LEGACY_ROUND1_VARIANTS = (
    # Explicit original settings preserve the first round's definitions.
    Variant("baseline", "Formal baseline", "baseline", (
        "--lambda-align", "1", "--lambda-stage1-closure", "0",
    )),
    Variant("branch1a", "Branch 1A", "branch1a", (
        "--lambda-align", "0", "--lambda-stage1-closure", "0",
    )),
    Variant("branch1b", "Branch 1B", "branch1b", (
        "--lambda-align", "0", "--lambda-stage1-closure", "0.5",
    )),
)
VARIANTS = (
    Variant("baseline", "Baseline (1B)", "baseline"),
    Variant("branch2a", "Branch 2A", "baseline", ("--lambda-retain-floor", "0")),
    Variant("branch2b", "Branch 2B", "baseline", (
        "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1",
    )),
    Variant("branch3", "Branch 3", "branch3", ("--lambda-keff", "0", "--lambda-gap", "1")),
    Variant("branch4", "Branch 4", "branch4", ("--lambda-dev", "1")),
    Variant("branch5", "Branch 5", "baseline", ("--ei-loss-retain-ratio", "1")),
)
METHOD_LABELS = tuple(variant.label for variant in VARIANTS)
STAGE1_VARIANTS = tuple(variant for variant in VARIANTS if variant.name == variant.stage1_group)
DEFAULT_BRANCHES = ("branch3",)


def select_variants(branches=None) -> tuple[Variant, ...]:
    requested = set(DEFAULT_BRANCHES if branches is None else branches)
    unknown = requested - {variant.name for variant in VARIANTS}
    if unknown:
        raise ValueError(f"Unknown formal branches: {sorted(unknown)}")
    requested.add("baseline")
    return tuple(variant for variant in VARIANTS if variant.name in requested)


def required_stage1_variants(selected: tuple[Variant, ...]) -> tuple[Variant, ...]:
    groups = {variant.stage1_group for variant in selected}
    return tuple(variant for variant in STAGE1_VARIANTS if variant.name in groups)


def report_name(selected: tuple[Variant, ...]) -> str:
    return "_".join(variant.name for variant in selected if variant.name != "baseline") or "baseline"
