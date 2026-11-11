"""Independent branches of the registered formal_two_stage baseline."""

from dataclasses import dataclass, replace

from mignet_ce.coarse_frontends.method_specs import (
    FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL,
    FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL,
)

from ..variants import TIME_PAIRS


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stage1_group: str
    flags: tuple[str, ...] = ()
    predecessor: str = "baseline"


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
LEGACY_1B_DEFINITIONS = (
    Variant("baseline", "Baseline (1B)", "baseline"),
    Variant("branch2a", "Branch 2A", "baseline", ("--lambda-retain-floor", "0")),
    Variant("branch2b", "Branch 2B", "baseline", (
        "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1",
    )),
    Variant("branch3", "Branch 3", "branch3", ("--lambda-keff", "0", "--lambda-gap", "1")),
    Variant("branch4", "Branch 4", "branch4", ("--lambda-dev", "1")),
    Variant("branch5", "Branch 5", "baseline", ("--ei-loss-retain-ratio", "1")),
)


def _legacy_variant(variant: Variant, align: str, closure: str, *, gap: bool = False) -> Variant:
    # Freeze promoted defaults so historical runs keep their original objectives.
    weights = {"--lambda-align": align, "--lambda-stage1-closure": closure,
               "--lambda-keff": "0" if gap else "1", "--lambda-gap": "1" if gap else "0",
               "--lambda-dev": "0.05", "--ei-loss-retain-ratio": "0.85"}
    weights.update(zip(variant.flags[::2], variant.flags[1::2]))
    return replace(variant, flags=tuple(item for pair in weights.items() for item in pair))


LEGACY_1B_VARIANTS = tuple(_legacy_variant(variant, "0", "0.5") for variant in LEGACY_1B_DEFINITIONS)
LEGACY_V1_VARIANTS = tuple(_legacy_variant(variant, "1", "0") for variant in (
    *LEGACY_ROUND1_VARIANTS, *LEGACY_1B_DEFINITIONS[1:],
))
LEGACY_GAP_DEFINITIONS = (
    Variant("baseline", "Baseline (1B+3)", "baseline"),
    Variant("branch2a", "Branch 2A", "baseline", ("--lambda-retain-floor", "0")),
    Variant("branch2b", "Branch 2B", "baseline", (
        "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1",
    )),
    Variant("branch4", "Branch 4", "branch4", ("--lambda-dev", "1")),
    Variant("branch5", "Branch 4+5", "branch4", (
        "--lambda-dev", "1", "--ei-loss-retain-ratio", "1",
    ), predecessor="branch4"),
)
LEGACY_GAP_VARIANTS = tuple(_legacy_variant(variant, "0", "0.5", gap=True) for variant in LEGACY_GAP_DEFINITIONS)
VARIANTS = (
    Variant("baseline", "Baseline (4+5)", "baseline"),
    Variant("branch2a", "Branch 2A", "baseline", ("--lambda-retain-floor", "0")),
    Variant("branch2b", "Branch 2B", "baseline", (
        "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1",
    ), predecessor="branch2a"),
)
METHOD_LABELS = tuple(variant.label for variant in VARIANTS)
STAGE1_VARIANTS = tuple(variant for variant in VARIANTS if variant.name == variant.stage1_group)
DEFAULT_BRANCHES = ("branch2a", "branch2b")


def variants_for_protocol(protocol: str = FORMAL_TWO_STAGE_PROTOCOL) -> tuple[Variant, ...]:
    profiles = {FORMAL_TWO_STAGE_PROTOCOL: VARIANTS, FORMAL_TWO_STAGE_GAP_PROTOCOL: LEGACY_GAP_VARIANTS,
                FORMAL_TWO_STAGE_1B_PROTOCOL: LEGACY_1B_VARIANTS,
                FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL: LEGACY_V1_VARIANTS}
    if protocol not in profiles:
        raise ValueError(f"Unknown formal loss protocol: {protocol}")
    return profiles[protocol]


def select_variants(branches=None, *, protocol: str = FORMAL_TWO_STAGE_PROTOCOL) -> tuple[Variant, ...]:
    variants = variants_for_protocol(protocol)
    defaults = (DEFAULT_BRANCHES if protocol == FORMAL_TWO_STAGE_PROTOCOL else
                ("branch4", "branch5") if protocol == FORMAL_TWO_STAGE_GAP_PROTOCOL else
                ("branch3",) if protocol == FORMAL_TWO_STAGE_1B_PROTOCOL else ("branch1a", "branch1b"))
    requested = set(defaults if branches is None else branches)
    unknown = requested - {variant.name for variant in variants}
    if unknown:
        raise ValueError(f"Unknown formal branches: {sorted(unknown)}")
    requested.add("baseline")
    by_name = {variant.name: variant for variant in variants}
    for name in tuple(requested):
        predecessor = by_name[name].predecessor
        while predecessor not in requested:
            requested.add(predecessor)
            predecessor = by_name[predecessor].predecessor
    return tuple(variant for variant in variants if variant.name in requested)


def required_stage1_variants(selected: tuple[Variant, ...], *, protocol: str = FORMAL_TWO_STAGE_PROTOCOL) -> tuple[Variant, ...]:
    groups = {variant.stage1_group for variant in selected}
    return tuple(variant for variant in variants_for_protocol(protocol) if variant.name in groups)


def comparison_predecessors(selected: tuple[Variant, ...]) -> dict[str, str]:
    labels = {variant.name: variant.label for variant in selected}
    return {variant.label: labels[variant.predecessor] for variant in selected}


def shared_stage1_preview(rows: list[dict], selected: tuple[Variant, ...]) -> None:
    """Copy each method's actual Stage 1 source, including shared baseline state."""
    labels = {variant.name: variant.label for variant in selected}
    for variant in selected:
        if variant.stage1_group == variant.name:
            continue
        for row in rows:
            if row["method"] == variant.label and row["stage"] == "stage1":
                source = next(reference for reference in rows if
                    reference["method"] == labels[variant.stage1_group] and reference["stage"] == "stage1"
                    and all(reference[key] == row[key] for key in ("input_scale", "source_time", "target_time", "k")))
                row.update({key: value for key, value in source.items() if key != "method"})


def report_name(selected: tuple[Variant, ...]) -> str:
    return "_".join(variant.name for variant in selected if variant.name != "baseline") or "baseline"
