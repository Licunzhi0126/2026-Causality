from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FORMAL_CACHE_PROTOCOL = "full_model_space_v4_canonical_ng"
FORMAL_TRANSITION_PROTOCOL = "canonical_ng_v1"


MAPPING_K150 = "Seurat K150"
MAPPING_K40 = "Seurat K40"
MAPPING_COMPLETE = "Optimized complete_combined_coarse"
MAPPING_MATURITY = "Optimized complete_combined_coarse_maturity_cci_grn"
MAPPING_TWO_STAGE = "Optimized maturity_cci_grn_two_stage"
UNIFIED_MAPPINGS = (
    MAPPING_K150,
    MAPPING_K40,
    MAPPING_COMPLETE,
    MAPPING_MATURITY,
    MAPPING_TWO_STAGE,
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
            f"fullv4_ngcanon_k{self.optimized_k}_e{self.optimized_epochs}_"
            f"nmf{self.nmf_components}_i{self.nmf_max_iter}_"
            f"seed{self.random_seed}"
        )


@dataclass(frozen=True)
class UnifiedDownstreamConfig:
    """Inputs for the five-representation, full-scale downstream workflow."""

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
