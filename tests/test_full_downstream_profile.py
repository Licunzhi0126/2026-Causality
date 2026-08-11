from __future__ import annotations

import json
from dataclasses import replace

from mignet_ce.visualization.downstream.config import FullDeltaEIBenchmarkProfile
from mignet_ce.visualization.downstream.preparation import (
    REQUIRED_OPTIMIZED_OUTPUTS,
    _is_valid_optimized_cache,
)


def test_formal_profile_is_locked_to_full_deltaei() -> None:
    profile = FullDeltaEIBenchmarkProfile()
    profile.validate()
    assert profile.optimized_epochs == 1500
    assert profile.optimized_k == 40
    assert profile.nmf_max_iter == 300
    assert profile.large_target_nmf_max_iter == 60
    assert profile.matched_null_repeats == 200
    assert profile.perturb_random_repeats == 200


def test_reduced_preview_profile_is_rejected() -> None:
    preview = replace(FullDeltaEIBenchmarkProfile(), optimized_epochs=3)
    try:
        preview.validate()
    except ValueError as exc:
        assert "locked full DeltaEI profile" in str(exc)
    else:
        raise AssertionError("A reduced preview profile must not validate as formal full DeltaEI")


def test_preview_trainer_config_cannot_validate_as_full_cache(tmp_path) -> None:
    expected = {
        "optimized_k": 40,
        "optimized_epochs": 1500,
        "random_seed": 20260809,
        "lambda_dev": 0.0,
    }
    for name in REQUIRED_OPTIMIZED_OUTPUTS:
        path = tmp_path / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.touch()
    (tmp_path / "downstream_full_manifest.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"k": 40, "epochs": 3, "seed": 20260809, "lambda_dev": 0.0}),
        encoding="utf-8",
    )
    assert not _is_valid_optimized_cache(tmp_path, expected)
