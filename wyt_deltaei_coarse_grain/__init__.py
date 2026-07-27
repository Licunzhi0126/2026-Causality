"""Shared WYT FeatureAlign-DeltaEI coarse-graining core."""

from wyt_deltaei_coarse_grain.trainer import (
    WYTDeltaEIConfig,
    WYTDeltaEIResult,
    train_deltaei,
)

__all__ = ["WYTDeltaEIConfig", "WYTDeltaEIResult", "train_deltaei"]
