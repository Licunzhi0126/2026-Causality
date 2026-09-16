from __future__ import annotations

import pytest

from mignet_ce.downstream.analysis.config import (
    FullDeltaEIBenchmarkProfile,
    UnifiedDownstreamConfig,
)


def test_unified_config_rejects_non_four_time_point_benchmark(tmp_path) -> None:
    cfg = UnifiedDownstreamConfig(
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        times=("11.5", "12.5", "13.5"),
    )
    with pytest.raises(ValueError, match="exactly four"):
        cfg.validate()


def test_formal_profile_rejects_reduced_benchmark_settings() -> None:
    profile = FullDeltaEIBenchmarkProfile(optimized_epochs=2)
    with pytest.raises(ValueError, match="locked full DeltaEI profile"):
        profile.validate()
