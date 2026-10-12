"""Non-overwriting output paths for perturbation conditions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .condition import PerturbationCondition


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    time_pair: str

    @property
    def pair_root(self) -> Path:
        return self.root / self.time_pair

    @property
    def baseline(self) -> Path:
        return self.pair_root / "baseline"

    def condition_root(self, condition: PerturbationCondition) -> Path:
        return self.pair_root / condition.pair_id / f"strength_{condition.strength:g}" / condition.direction

    def pair_qc(self, condition: PerturbationCondition) -> Path:
        return self.pair_root / condition.pair_id / f"strength_{condition.strength:g}" / "symmetry_qc.json"

    def strength_root(self, pair_id: str, strength: float) -> Path:
        return self.pair_root / pair_id / f"strength_{strength:g}"

    def region_root(self, pair_id: str) -> Path:
        return self.pair_root / pair_id

    def region_file(self, pair_id: str) -> Path:
        return self.region_root(pair_id) / "region.json"
