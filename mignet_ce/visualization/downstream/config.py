from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
