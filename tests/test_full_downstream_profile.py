from __future__ import annotations

import json
from dataclasses import replace

from mignet_ce.downstream.analysis.config import (
    FORMAL_CACHE_PROTOCOL,
    FORMAL_TRANSITION_PROTOCOL,
    FullDeltaEIBenchmarkProfile,
)
from mignet_ce.downstream.analysis.preparation import (
    _is_valid_optimized_cache,
    required_optimized_outputs,
)
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_transition_contract


def test_formal_profile_is_locked_to_full_deltaei() -> None:
    profile = FullDeltaEIBenchmarkProfile()
    profile.validate()
    assert profile.optimized_epochs == 1500
    assert profile.optimized_k == 40
    assert profile.nmf_max_iter == 300
    assert not hasattr(profile, "large_target_nmf_max_iter")
    assert not hasattr(profile, "large_target_threshold")
    assert profile.profile_id == "fullv5_rawkl_ng_k40_e1500_nmf5_i300_seed20260809"
    assert FORMAL_CACHE_PROTOCOL == "full_model_space_v5_rawkl_ng"
    assert FORMAL_TRANSITION_PROTOCOL == "canonical_ng_rawkl_v2"
    contract = canonical_transition_contract()
    assert contract["alpha_cci"] == 0.01
    assert contract["tau"] == 0.8
    assert contract["component_normalization"] == "none"
    assert contract["combined_scale_control"] == "none"
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
        "cache_protocol": FORMAL_CACHE_PROTOCOL,
        "transition_protocol": FORMAL_TRANSITION_PROTOCOL,
        "transition_contract": canonical_transition_contract(),
        "method": "complete_combined_coarse",
        "nmf_max_iter_used": 300,
        "optimized_k": 40,
        "optimized_epochs": 1500,
        "random_seed": 20260809,
        "lambda_dev": 0.0,
        "training_hyperparameters": {
            "k": 40,
            "epochs": 1500,
            "seed": 20260809,
            "lambda_dev": 0.0,
        },
    }
    for name in required_optimized_outputs("complete_combined_coarse"):
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


def test_old_l60_cache_protocol_cannot_validate_as_full_v2(tmp_path) -> None:
    expected = {
        "cache_protocol": FORMAL_CACHE_PROTOCOL,
        "transition_protocol": FORMAL_TRANSITION_PROTOCOL,
        "transition_contract": canonical_transition_contract(),
        "method": "complete_combined_coarse",
        "nmf_max_iter_used": 300,
        "optimized_k": 40,
        "optimized_epochs": 1500,
        "random_seed": 20260809,
        "lambda_dev": 0.0,
        "training_hyperparameters": {
            "k": 40,
            "epochs": 1500,
            "seed": 20260809,
            "lambda_dev": 0.0,
        },
    }
    for name in required_optimized_outputs("complete_combined_coarse"):
        path = tmp_path / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.touch()
    old = dict(expected)
    old.update(cache_protocol="formal_full_deltaei", nmf_max_iter_used=60)
    (tmp_path / "downstream_full_manifest.json").write_text(
        json.dumps(old), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"k": 40, "epochs": 1500, "seed": 20260809, "lambda_dev": 0.0}),
        encoding="utf-8",
    )
    assert not _is_valid_optimized_cache(tmp_path, expected)


def test_v3_cache_protocol_cannot_validate_as_canonical_v4(tmp_path) -> None:
    expected = {
        "cache_protocol": FORMAL_CACHE_PROTOCOL,
        "transition_protocol": FORMAL_TRANSITION_PROTOCOL,
        "transition_contract": canonical_transition_contract(),
        "method": "complete_combined_coarse",
        "nmf_max_iter_used": 300,
        "optimized_k": 40,
        "optimized_epochs": 1500,
        "random_seed": 20260809,
        "lambda_dev": 0.0,
        "training_hyperparameters": {
            "k": 40,
            "epochs": 1500,
            "seed": 20260809,
            "lambda_dev": 0.0,
        },
    }
    for name in required_optimized_outputs("complete_combined_coarse"):
        path = tmp_path / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.touch()
    old = dict(expected)
    old.pop("transition_protocol")
    old.pop("transition_contract")
    old["cache_protocol"] = "full_model_space_v3"
    (tmp_path / "downstream_full_manifest.json").write_text(
        json.dumps(old), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(expected["training_hyperparameters"]), encoding="utf-8"
    )
    assert not _is_valid_optimized_cache(tmp_path, expected)
