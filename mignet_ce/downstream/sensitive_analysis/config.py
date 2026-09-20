from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mignet_ce.downstream.analysis.config import FullDeltaEIBenchmarkProfile

_PRODUCTION_PROFILE = FullDeltaEIBenchmarkProfile()

DEFAULT_TIME_PAIRS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
DEFAULT_LAYERS = ("spot", "seurat_k150", "seurat_k40")
DEFAULT_COMPARISONS = (
    ("spot", "seurat_k150", "spot:seuratK150"),
    ("spot", "seurat_k40", "spot:seuratK40"),
    ("seurat_k150", "seurat_k40", "seuratK150:seuratK40"),
)


def default_alphas() -> tuple[float, ...]:
    return tuple(round(i * 0.05, 10) for i in range(21))


@dataclass(frozen=True)
class SensitivityConfig:
    data_root: Path
    output_dir: Path
    organ: str = "heart"
    time_pairs: tuple[str, ...] = DEFAULT_TIME_PAIRS
    layers: tuple[str, ...] = DEFAULT_LAYERS
    alphas: tuple[float, ...] = default_alphas()
    beta_n: float = 0.05
    beta_g: float = 0.05
    tau: float = 0.1
    canonical_alpha: float = 0.1
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
        if self.beta_n <= 0.0 or self.beta_g <= 0.0 or self.tau <= 0.0:
            raise ValueError("beta_n, beta_g, and tau must be positive.")
        if self.nmf_components < 1 or self.nmf_max_iter < 1:
            raise ValueError("NMF settings must be positive.")
        for pair in self.time_pairs:
            if "->" not in pair:
                raise ValueError(f"Invalid time pair {pair!r}; expected e.g. 11.5->12.5")
