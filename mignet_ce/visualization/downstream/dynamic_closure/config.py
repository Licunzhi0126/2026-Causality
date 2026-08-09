from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


DEFAULT_TIME_POINTS = ("11.5", "12.5", "13.5", "14.5")


@dataclass(frozen=True)
class DynamicClosureConfig:
    """Configuration shared by the full, deep, and ultra-deep closure stages."""

    data_root: Path
    output_root: Path
    organ: str = "heart"
    time_points: tuple[str, ...] = DEFAULT_TIME_POINTS
    k_optimal: int = 40
    optimal_epochs: int = 300
    nmf_components: int = 5
    nmf_max_iter: int = 300
    large_target_nmf_max_iter: int = 60
    seed: int = 42
    null_repeats: int = 200
    bootstrap_repeats: int = 200
    repair_max_splits: int = 8
    repair_random_repeats: int = 40
    device: str = "cpu"
    force: bool = False
    output_mode: str = "both"
    crossfit_folds: int = 5
    low_signal_threshold_bits: float | None = None

    def normalized(self) -> "DynamicClosureConfig":
        return replace(
            self,
            data_root=Path(self.data_root).resolve(),
            output_root=Path(self.output_root).resolve(),
            organ=str(self.organ),
            time_points=tuple(map(str, self.time_points)),
            k_optimal=int(self.k_optimal),
            optimal_epochs=int(self.optimal_epochs),
            nmf_components=int(self.nmf_components),
            nmf_max_iter=int(self.nmf_max_iter),
            large_target_nmf_max_iter=int(self.large_target_nmf_max_iter),
            seed=int(self.seed),
            null_repeats=int(self.null_repeats),
            bootstrap_repeats=int(self.bootstrap_repeats),
            repair_max_splits=int(self.repair_max_splits),
            repair_random_repeats=int(self.repair_random_repeats),
            device=str(self.device),
            force=bool(self.force),
            output_mode=str(self.output_mode),
            crossfit_folds=int(self.crossfit_folds),
            low_signal_threshold_bits=(None if self.low_signal_threshold_bits is None else float(self.low_signal_threshold_bits)),
        )

    def validate(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Dynamic-closure data root does not exist: {self.data_root}")
        if len(self.time_points) != 4 or len(set(self.time_points)) != 4:
            raise ValueError("The extended dynamic-closure suite requires four unique ordered time points.")
        if self.output_mode not in {"paper", "diagnostic", "both"}:
            raise ValueError("output_mode must be paper, diagnostic, or both")
        if self.crossfit_folds < 2:
            raise ValueError("crossfit_folds must be at least 2")
        if self.low_signal_threshold_bits is not None and self.low_signal_threshold_bits < 0:
            raise ValueError("low_signal_threshold_bits must be nonnegative")
        if self.k_optimal < 2:
            raise ValueError("k_optimal must be at least 2")
        if self.optimal_epochs < 1:
            raise ValueError("optimal_epochs must be positive")
        if min(
            self.nmf_components,
            self.nmf_max_iter,
            self.large_target_nmf_max_iter,
            self.null_repeats,
            self.bootstrap_repeats,
            self.repair_max_splits,
            self.repair_random_repeats,
        ) < 1:
            raise ValueError("NMF, repeat, and repair controls must be positive")

    @property
    def adjacent_pairs(self) -> tuple[str, ...]:
        return tuple(
            f"{left}->{right}"
            for left, right in zip(self.time_points[:-1], self.time_points[1:])
        )

    @property
    def full_root(self) -> Path:
        return self.output_root / "full"

    @property
    def deep_root(self) -> Path:
        return self.output_root / "deep"

    @property
    def ultradeep_root(self) -> Path:
        return self.output_root / "ultradeep"

    @property
    def paper_root(self) -> Path:
        return self.output_root / "paper"


# Backwards-compatible name used by the museum implementation.
ClosureFullConfig = DynamicClosureConfig


__all__ = ["ClosureFullConfig", "DEFAULT_TIME_POINTS", "DynamicClosureConfig"]
