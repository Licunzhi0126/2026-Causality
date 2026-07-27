from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            frozen[str(key)] = _freeze_mapping(value)
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class FeatureRecipe:
    recipe_id: str
    entry_method: str
    algorithm_version: str
    cci_mode: str
    grn_mode: str
    block_distances: Mapping[str, str]
    distance_parameters: Mapping[str, Mapping[str, float]]
    fusion_weights: Mapping[str, float]
    nmf_rank: int
    nmf_max_iter: int
    nmf_seeds: tuple[int, ...]
    grn_topk_targets: int
    projection_dim: int
    projection_seed: int
    kernel_temperature: float
    pseudocount: float
    normalization_quantiles: tuple[float, float] = (5.0, 95.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_distances", _freeze_mapping(self.block_distances))
        object.__setattr__(self, "distance_parameters", _freeze_mapping(self.distance_parameters))
        object.__setattr__(self, "fusion_weights", _freeze_mapping(self.fusion_weights))
        if not self.recipe_id or not self.entry_method:
            raise ValueError("recipe_id and entry_method must be non-empty.")
        if self.nmf_rank <= 0 or self.nmf_max_iter < 0:
            raise ValueError("nmf_rank must be positive and nmf_max_iter must be nonnegative.")
        if not self.nmf_seeds:
            raise ValueError("At least one NMF seed is required.")
        if self.grn_topk_targets <= 0 or self.projection_dim <= 0:
            raise ValueError("GRN top-k and projection dimension must be positive.")
        if self.kernel_temperature <= 0.0 or self.pseudocount <= 0.0:
            raise ValueError("Kernel temperature and pseudocount must be positive.")
        if set(self.block_distances) != set(self.fusion_weights):
            raise ValueError("block_distances and fusion_weights must contain identical block names.")
        weights = [float(value) for value in self.fusion_weights.values()]
        if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Fusion weights must be nonnegative and sum to 1.")
        q_low, q_high = map(float, self.normalization_quantiles)
        if not 0.0 <= q_low < q_high <= 100.0:
            raise ValueError("normalization_quantiles must satisfy 0 <= low < high <= 100.")


@dataclass
class PairFeatureBundle:
    source_blocks: dict[str, np.ndarray]
    target_blocks: dict[str, np.ndarray]
    metadata: dict[str, object]
    artifacts: dict[str, object]

    def validate(self) -> None:
        if set(self.source_blocks) != set(self.target_blocks):
            raise ValueError("Source and target block names differ.")
        for name in self.source_blocks:
            source = np.asarray(self.source_blocks[name], dtype=float)
            target = np.asarray(self.target_blocks[name], dtype=float)
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Feature block {name!r} must contain 2D matrices.")
            if source.shape[1] != target.shape[1]:
                raise ValueError(f"Feature block {name!r} dimensions differ: {source.shape} vs {target.shape}.")
            if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
                raise ValueError(f"Feature block {name!r} contains non-finite values.")
