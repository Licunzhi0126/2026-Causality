"""Stable condition identifiers and deterministic seeds."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib


PAIR_IDS = (
    "P1_degree_heterogeneity", "P2_modularity", "P3_clustering", "P4_assortativity", "P5_reciprocity",
    "P6_triangle_intensity", "P7_feedforward_loop", "P8_feedback_cycle",
)
DIRECTIONS = ("positive", "negative")


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


@dataclass(frozen=True)
class PerturbationCondition:
    time_pair: str
    pair_id: str
    strength: float
    direction: str
    seed: int

    def __post_init__(self) -> None:
        if self.pair_id not in PAIR_IDS:
            raise ValueError(f"Unknown topology pair: {self.pair_id}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {self.direction}")
        if not 0.0 < self.strength < 1.0:
            raise ValueError("strength must be strictly between zero and one.")

    @property
    def run_id(self) -> str:
        return f"{self.time_pair}:{self.pair_id}:{self.strength:g}:{self.direction}"
