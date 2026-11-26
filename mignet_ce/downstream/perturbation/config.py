"""Configuration for direct, mirrored GRN weight perturbations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..formal_protocol import FORMAL_SEED, resolve_options


@dataclass(frozen=True)
class PerturbationConfig:
    strengths: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    mirror_fraction: float = 1.0
    region_node_fraction: float = 0.40
    region_edge_fraction: float = 0.40
    random_seed: int = FORMAL_SEED
    mirror_tolerance: float = 1e-10
    k: int = 40
    device: str = "cpu"
    method: str = "formal_two_stage"
    local_graph_mode: str = "all_features"
    frontend_options: dict = field(default_factory=dict)
    training_options: dict = field(default_factory=dict)

    def formal(self):
        if self.random_seed != FORMAL_SEED:
            raise ValueError(f"Formal Core requires perturbation seed {FORMAL_SEED}")
        frontend, training = resolve_options(self.frontend_options, self.training_options, strict=True)
        return replace(self, frontend_options=frontend, training_options=training)

    def __post_init__(self) -> None:
        if self.method != "formal_two_stage":
            raise ValueError("method must be formal_two_stage.")
        if self.local_graph_mode not in {"legacy_features", "coords", "all_features"}:
            raise ValueError("Unsupported local_graph_mode.")
        if not isinstance(self.frontend_options, dict) or not isinstance(self.training_options, dict):
            raise ValueError("frontend_options and training_options must be dicts.")
        options = dict(self.training_options)
        if "local_graph_mode" in options and options["local_graph_mode"] != self.local_graph_mode:
            raise ValueError("Conflicting local_graph_mode in config and training_options.")
        options["local_graph_mode"] = self.local_graph_mode
        object.__setattr__(self, "training_options", options)
        object.__setattr__(self, "frontend_options", dict(self.frontend_options))
        if not self.strengths or any(not 0.0 < value < 1.0 for value in self.strengths):
            raise ValueError("strengths must be non-empty values strictly between zero and one.")
        if not 0.0 < self.mirror_fraction <= 1.0:
            raise ValueError("mirror_fraction must be strictly between zero and one.")
        if not 0.0 < self.region_node_fraction <= 1.0 or not 0.0 < self.region_edge_fraction <= 1.0:
            raise ValueError("region fractions must be strictly between zero and one.")
        if self.mirror_tolerance < 0.0:
            raise ValueError("mirror_tolerance must be non-negative.")
