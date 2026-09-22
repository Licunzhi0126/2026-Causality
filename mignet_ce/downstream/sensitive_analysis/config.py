from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mignet_ce.downstream.analysis.config import FullDeltaEIBenchmarkProfile
from mignet_ce.pij.compare._shared.ng_kl_ot import (
    CANONICAL_ALPHA_CCI,
    CANONICAL_FEATURE_BETA_G,
    CANONICAL_FEATURE_BETA_N,
    CANONICAL_TEMPERATURE,
)

_PRODUCTION_PROFILE = FullDeltaEIBenchmarkProfile()

DEFAULT_TIME_PAIRS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
DEFAULT_LAYERS = ("spot", "seurat_k150", "seurat_k40")
DEFAULT_COMPARISONS = (
    ("spot", "seurat_k150", "spot:seuratK150"),
    ("spot", "seurat_k40", "spot:seuratK40"),
    ("seurat_k150", "seurat_k40", "seuratK150:seuratK40"),
)


def default_alphas() -> tuple[float, ...]:
    return (
        0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075,
        0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00,
    )


@dataclass(frozen=True)
class SensitivityConfig:
    data_root: Path
    output_dir: Path
    organ: str = "heart"
    time_pairs: tuple[str, ...] = DEFAULT_TIME_PAIRS
    layers: tuple[str, ...] = DEFAULT_LAYERS
    alphas: tuple[float, ...] = default_alphas()
    beta_n: float = CANONICAL_FEATURE_BETA_N
    beta_g: float = CANONICAL_FEATURE_BETA_G
    tau: float = CANONICAL_TEMPERATURE
    canonical_alpha: float = CANONICAL_ALPHA_CCI
    nmf_components: int = _PRODUCTION_PROFILE.nmf_components
    nmf_max_iter: int = _PRODUCTION_PROFILE.nmf_max_iter
    random_seed: int = _PRODUCTION_PROFILE.random_seed

    def normalized(self) -> "SensitivityConfig":
        return SensitivityConfig(
            data_root=Path(self.data_root).resolve(),
            output_dir=Path(self.output_dir).resolve(),
            organ=str(self.organ),
            time_pairs=tuple(map(str, self.time_pairs)),
            layers=tuple(map(str, self.layers)),
            alphas=tuple(float(x) for x in self.alphas),
            beta_n=float(self.beta_n),
            beta_g=float(self.beta_g),
            tau=float(self.tau),
            canonical_alpha=float(self.canonical_alpha),
            nmf_components=int(self.nmf_components),
            nmf_max_iter=int(self.nmf_max_iter),
            random_seed=int(self.random_seed),
        )

    def validate(self) -> None:
        if not Path(self.data_root).exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")
        if not self.time_pairs:
            raise ValueError("At least one time pair is required.")
        if tuple(self.layers) != DEFAULT_LAYERS:
            raise ValueError(f"This analysis expects layers {DEFAULT_LAYERS}; got {self.layers}")
        if not self.alphas:
            raise ValueError("At least one alpha is required.")
        if any(alpha < 0.0 or alpha > 1.0 for alpha in self.alphas):
            raise ValueError("Every alpha must lie in [0, 1].")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("Alpha sweep values must be unique.")
        if not 0.0 <= self.canonical_alpha <= 1.0:
            raise ValueError("canonical_alpha must lie in [0, 1].")
        if not any(abs(alpha - self.canonical_alpha) <= 1e-12 for alpha in self.alphas):
            raise ValueError("The alpha sweep must include canonical_alpha for parity auditing.")
        if self.beta_n <= 0.0 or self.beta_g <= 0.0 or self.tau <= 0.0:
            raise ValueError("beta_n, beta_g, and tau must be positive.")
        fixed = {
            "beta_n": (self.beta_n, CANONICAL_FEATURE_BETA_N),
            "beta_g": (self.beta_g, CANONICAL_FEATURE_BETA_G),
            "tau": (self.tau, CANONICAL_TEMPERATURE),
            "canonical_alpha": (self.canonical_alpha, CANONICAL_ALPHA_CCI),
        }
        mismatched = {
            name: {"received": received, "expected": expected}
            for name, (received, expected) in fixed.items()
            if abs(float(received) - float(expected)) > 1e-12
        }
        if mismatched:
            raise ValueError(
                "Sensitivity varies alpha only; production beta/tau/reference alpha "
                f"are fixed: {mismatched}"
            )
        if self.nmf_components < 1 or self.nmf_max_iter < 1:
            raise ValueError("NMF settings must be positive.")
        for pair in self.time_pairs:
            if pair.count("->") != 1 or any(not item for item in pair.split("->")):
                raise ValueError(f"Invalid time pair {pair!r}; expected e.g. 11.5->12.5")
        if len(set(self.time_pairs)) != len(self.time_pairs):
            raise ValueError("Time pairs must be unique.")
        if "12.5->13.5" not in self.time_pairs:
            raise ValueError("The requested sensitivity figure requires 12.5->13.5.")
