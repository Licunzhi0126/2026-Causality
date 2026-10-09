from __future__ import annotations
from dataclasses import dataclass

METHODS = ("legacy", "legacy_nmf", "legacy_lap_hks")

@dataclass(frozen=True)
class GRNFeatureConfig:
    method: str = "legacy"
    residual_lambda: float = 0.15
    nmf_rank: int = 6
    nmf_steps: int = 65
    hks_rank: int = 8
    nmf_seed: int = 42

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported GRN feature method {self.method!r}; choose {METHODS}.")
        if not 0 <= self.residual_lambda <= 2:
            raise ValueError("GRN residual_lambda must lie in [0, 2].")
        if self.nmf_rank < 1 or self.nmf_steps < 1 or self.hks_rank < 1:
            raise ValueError("GRN factor dimensions and iteration counts must be positive.")
