"""Retain/CQ ablations of the sole formal_two_stage_main protocol."""

from dataclasses import dataclass

from mignet_ce.coarse_frontends.method_specs import (
    FORMAL_TWO_STAGE_PROTOCOL, validate_formal_protocol,
)

from ..variants import TIME_PAIRS


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stage1_group: str
    flags: tuple[str, ...] = ()
    predecessor: str = "baseline"


VARIANTS = (
    Variant("baseline", "Formal main", "baseline"),
    Variant("branch2a", "Branch 2A", "baseline", ("--lambda-retain-floor", "0")),
    Variant("branch2b", "Branch 2B", "baseline", (
        "--lambda-retain-floor", "0", "--lambda-stage2-cq", "1",
    ), predecessor="branch2a"),
)
METHOD_LABELS = tuple(variant.label for variant in VARIANTS)
STAGE1_VARIANTS = tuple(variant for variant in VARIANTS if variant.name == variant.stage1_group)
DEFAULT_BRANCHES = ("branch2a", "branch2b")


def variants_for_protocol(protocol: str = FORMAL_TWO_STAGE_PROTOCOL) -> tuple[Variant, ...]:
    validate_formal_protocol(protocol)
    return VARIANTS


def select_variants(branches=None, *, protocol: str = FORMAL_TWO_STAGE_PROTOCOL) -> tuple[Variant, ...]:
    variants = variants_for_protocol(protocol)
    requested = set(DEFAULT_BRANCHES if branches is None else branches)
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
