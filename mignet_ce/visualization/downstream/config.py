from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


MAPPING_K150 = "Seurat K150"
MAPPING_K40 = "Seurat K40"
MAPPING_COMPLETE = "Optimized complete_combined_coarse"
MAPPING_MATURITY = "Optimized complete_combined_coarse_maturity_cci_grn"
UNIFIED_MAPPINGS = (
    MAPPING_K150,
    MAPPING_K40,
    MAPPING_COMPLETE,
    MAPPING_MATURITY,
)


@dataclass(frozen=True)
class DownstreamConfig:
    """Inputs and reproducibility controls for one downstream run."""

    data_root: Path
    metrics_csv: Path
    pair_archive: Path
    output_dir: Path
    organ: str = "heart"
    times: tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5")
    network_method: str = "light_cci_grn"
    pij_method: str = "NG_KLot"
    lower_layer: str = "seurat_k150"
    upper_layer: str = "seurat_k40"
    random_repeats: int = 500
    random_seed: int = 20260731
    spatial_knn: int = 6
    perturb_random_repeats: int = 200

    def normalized(self) -> "DownstreamConfig":
        return DownstreamConfig(
            data_root=Path(self.data_root).resolve(),
            metrics_csv=Path(self.metrics_csv).resolve(),
            pair_archive=Path(self.pair_archive).resolve(),
            output_dir=Path(self.output_dir).resolve(),
            organ=str(self.organ),
            times=tuple(map(str, self.times)),
            network_method=str(self.network_method),
            pij_method=str(self.pij_method),
            lower_layer=str(self.lower_layer),
            upper_layer=str(self.upper_layer),
            random_repeats=int(self.random_repeats),
            random_seed=int(self.random_seed),
            spatial_knn=int(self.spatial_knn),
            perturb_random_repeats=int(self.perturb_random_repeats),
        )

    def validate(self) -> None:
        missing = [
            path
            for path in (self.data_root, self.metrics_csv, self.pair_archive)
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing downstream input(s): " + ", ".join(map(str, missing)))
        if len(self.times) != 4:
            raise ValueError("The fixed 2x3 figure suite requires exactly four ordered time points.")
        if len(set(self.times)) != len(self.times):
            raise ValueError(f"time points must be unique, got {self.times}")
        if self.random_repeats < 1 or self.perturb_random_repeats < 1:
            raise ValueError("random repeat counts must be positive")
        if self.spatial_knn < 1:
            raise ValueError("spatial_knn must be positive")

    @property
    def adjacent_pairs(self) -> tuple[str, ...]:
        return tuple(f"{left}->{right}" for left, right in zip(self.times[:-1], self.times[1:]))

    @property
    def all_pairs(self) -> tuple[str, ...]:
        return tuple(
            f"{self.times[i]}->{self.times[j]}"
            for i in range(len(self.times))
            for j in range(i + 1, len(self.times))
        )


@dataclass(frozen=True)
class FullDeltaEIBenchmarkProfile:
    """Locked production profile for every downstream-triggered DeltaEI run.

    The unified production workflow accepts this exact profile only.  Small smoke
    settings belong in isolated tests and can never satisfy the full-cache audit.
    """

    optimized_k: int = 40
    optimized_epochs: int = 1500
    nmf_components: int = 5
    nmf_max_iter: int = 300
    matched_null_repeats: int = 200
    perturb_random_repeats: int = 200
    crossfit_folds: int = 5
    random_seed: int = 20260809
    maturity_lambda_dev: float = 0.05

    def validate(self) -> None:
        expected = FullDeltaEIBenchmarkProfile()
        if self != expected:
            raise ValueError(
                "The formal unified downstream entry requires the locked full "
                f"DeltaEI profile {expected}; received {self}."
            )

    @property
    def profile_id(self) -> str:
        return (
            f"fullv2_k{self.optimized_k}_e{self.optimized_epochs}_"
            f"nmf{self.nmf_components}_i{self.nmf_max_iter}_"
            f"seed{self.random_seed}"
        )


@dataclass(frozen=True)
class UnifiedDownstreamConfig:
    """Inputs for the four-representation, full-scale downstream workflow."""

    data_root: Path
    cache_root: Path
    output_dir: Path
    developmental_feature_root: Path | None = None
    organ: str = "heart"
    times: tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5")
    device: str = "auto"
    spatial_knn: int = 6
    profile: FullDeltaEIBenchmarkProfile = field(default_factory=FullDeltaEIBenchmarkProfile)

    def normalized(self) -> "UnifiedDownstreamConfig":
        return UnifiedDownstreamConfig(
            data_root=Path(self.data_root).resolve(),
            cache_root=Path(self.cache_root).resolve(),
            output_dir=Path(self.output_dir).resolve(),
            developmental_feature_root=(
                Path(self.developmental_feature_root).resolve()
                if self.developmental_feature_root is not None
                else None
            ),
            organ=str(self.organ),
            times=tuple(map(str, self.times)),
            device=str(self.device),
            spatial_knn=int(self.spatial_knn),
            profile=self.profile,
        )

    def validate(self) -> None:
        self.profile.validate()
        if not Path(self.data_root).exists():
            raise FileNotFoundError(f"Unified downstream data root does not exist: {self.data_root}")
        if len(self.times) != 4 or len(set(self.times)) != 4:
            raise ValueError(
                "The formal unified benchmark requires exactly four unique ordered time points."
            )
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of cpu, cuda, auto")
        if self.spatial_knn < 1:
            raise ValueError("spatial_knn must be positive")

    @property
    def adjacent_pairs(self) -> tuple[str, ...]:
        return tuple(f"{left}->{right}" for left, right in zip(self.times[:-1], self.times[1:]))

    @property
    def full_cache_root(self) -> Path:
        return Path(self.cache_root) / "full" / self.profile.profile_id

    @property
    def mapping_names(self) -> tuple[str, ...]:
        return UNIFIED_MAPPINGS
