"""Configuration for direct, mirrored GRN weight perturbations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationConfig:
    strengths: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    mirror_fraction: float = 1.0
    region_node_fraction: float = 0.40
    region_edge_fraction: float = 0.40
    random_seed: int = 20260930
    mirror_tolerance: float = 1e-10
    k: int = 40
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.strengths or any(not 0.0 < value < 1.0 for value in self.strengths):
            raise ValueError("strengths must be non-empty values strictly between zero and one.")
        if not 0.0 < self.mirror_fraction <= 1.0:
            raise ValueError("mirror_fraction must be strictly between zero and one.")
        if not 0.0 < self.region_node_fraction <= 1.0 or not 0.0 < self.region_edge_fraction <= 1.0:
            raise ValueError("region fractions must be strictly between zero and one.")
        if self.mirror_tolerance < 0.0:
            raise ValueError("mirror_tolerance must be non-negative.")
