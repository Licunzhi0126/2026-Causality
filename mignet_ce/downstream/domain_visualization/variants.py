"""Domain experiment designs; the training method and loss protocol stay unchanged."""

from dataclasses import dataclass


TIME_PAIRS = (("11.5", "12.5"), ("12.5", "13.5"), ("11.5", "13.5"))
BASELINE_WEIGHTS = {
    "lambda_stage1_closure": 0.5, "lambda_closure": 0.1, "lambda_dev": 1.0,
    "lambda_ei_constraint": 5.0, "lambda_retain_norm": 0.5, "lambda_retain_floor": 2.0,
}
# The six-version design above is frozen for reading historical experiments.
FORMAL_WEIGHTS = {
    "lambda_stage1_closure": 1.0, "lambda_closure": 1.0, "lambda_dev": 2.0,
    "lambda_ei_constraint": 1.0, "lambda_retain_norm": 1.0, "lambda_retain_floor": 1.0,
}


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stage1_group: str
    overrides: tuple[tuple[str, float], ...] = ()
    base_weights: tuple[tuple[str, float], ...] = tuple(BASELINE_WEIGHTS.items())

    @property
    def weights(self) -> dict[str, float]:
        return {**dict(self.base_weights), **dict(self.overrides)}

    @property
    def flags(self) -> list[str]:
        return [item for key, value in self.weights.items()
                for item in ("--" + key.replace("_", "-"), str(value))]


VARIANTS = (
    Variant("baseline", "Baseline", "baseline"),
    Variant("closure", "Closure = 1", "closure", (
        ("lambda_stage1_closure", 1.0), ("lambda_closure", 1.0),
    )),
    Variant("development", "Development = 2", "development", (("lambda_dev", 2.0),)),
    Variant("ei_constraint", "EI constraint = 1", "baseline", (("lambda_ei_constraint", 1.0),)),
    Variant("retain_norm", "Retain norm = 1", "baseline", (("lambda_retain_norm", 1.0),)),
    Variant("retain_floor", "Retain floor = 1", "baseline", (("lambda_retain_floor", 1.0),)),
)
STAGE1_VARIANTS = tuple(variant for variant in VARIANTS if variant.name == variant.stage1_group)

FACTORIAL_VARIANTS = tuple(
    Variant(
        f"s{index}", f"S{index} · C={closure:g}, D={dev:g}, R={retain:g}",
        f"c{closure:g}_d{dev:g}".replace(".", "p"),
        (("lambda_stage1_closure", closure), ("lambda_closure", closure),
         ("lambda_dev", dev), ("lambda_retain_norm", retain)),
        tuple(FORMAL_WEIGHTS.items()),
    )
    for index, (closure, dev, retain) in enumerate(
        ((c, d, r) for c in (0.5, 1.0) for d in (1.0, 2.0) for r in (0.5, 1.0)), 1
    )
)


def variants_for_experiment(experiment: str = "factorial") -> tuple[Variant, ...]:
    if experiment == "factorial":
        return FACTORIAL_VARIANTS
    if experiment == "single_factor":
        return VARIANTS
    raise ValueError(f"Unknown domain experiment: {experiment}")


def reference_for_experiment(experiment: str) -> str:
    variants_for_experiment(experiment)
    return "s8" if experiment == "factorial" else "baseline"


def selected_variants(args) -> tuple[Variant, ...]:
    return variants_for_experiment(getattr(args, "experiment", None) or "factorial")
