"""Shared WYT FeatureAlign-DeltaEI coarse-graining core."""

from wyt_deltaei_coarse_grain.trainer import (
    WYTDeltaEIConfig,
    WYTDeltaEIResult,
    train_deltaei,
)
from wyt_deltaei_coarse_grain.two_stage_trainer import (
    WYTTwoStageDeltaEIConfig,
    WYTTwoStageDeltaEIResult,
)

__all__ = [
    "WYTDeltaEIConfig",
    "WYTDeltaEIResult",
    "WYTTwoStageDeltaEIConfig",
    "WYTTwoStageDeltaEIResult",
    "train_deltaei",
]
