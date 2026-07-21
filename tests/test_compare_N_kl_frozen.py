from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mignet_ce.metrics import effective_information
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare.compare_N_kl import CompareNKlPijMethod, build_block_kl_cost
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(__file__).resolve().parent / "baseline"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_protected_compare_n_kl_files_are_byte_identical() -> None:
    payload = json.loads((BASELINE_ROOT / "protected_file_sha256.json").read_text(encoding="utf-8"))
    for relative_path, expected in payload["protected_files"].items():
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        assert _sha256(path) == expected, relative_path


def test_protected_files_do_not_depend_on_feature_versions() -> None:
    payload = json.loads((BASELINE_ROOT / "protected_file_sha256.json").read_text(encoding="utf-8"))
    for relative_path in payload["protected_files"]:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "feature_versions" not in text, relative_path


def test_new_feature_version_dependency_direction_is_isolated() -> None:
    method_base = (REPO_ROOT / "mignet_ce/pij/feature_versions/method_base.py").read_text(encoding="utf-8")
    assert "monkeypatch" not in method_base
    assert "VerticalMIGNetPipeline" not in method_base
    assert "PIJ_METHOD_REGISTRY" not in method_base
    for filename in (
        "compare_NG_kl_splitbeta_v1.py",
        "compare_Ncomp_Gcos_v2.py",
        "compare_Nshape_Gcos_v3.py",
    ):
        text = (REPO_ROOT / "mignet_ce/pij/compare" / filename).read_text(encoding="utf-8")
        assert "build_block_kl_cost" not in text
        assert "FeatureVersionPijMethod" in text


def test_compare_n_kl_registry_binding_is_frozen() -> None:
    assert PIJ_METHOD_REGISTRY["compare_N_kl"] is CompareNKlPijMethod
    assert isinstance(get_pij_method("compare_N_kl"), CompareNKlPijMethod)


def test_compare_n_kl_numeric_fixture_is_frozen() -> None:
    fixture = np.load(BASELINE_ROOT / "compare_N_kl_fixture_pij.npz")
    weights = {"weight_n": float(fixture["weight_n"]), "weight_g": float(fixture["weight_g"])}
    betas = {"beta_n": float(fixture["beta_n"]), "beta_g": float(fixture["beta_g"])}

    actual_pij: dict[str, np.ndarray] = {}
    for side in ("lower", "upper"):
        cost, _ = build_block_kl_cost(
            fixture[f"{side}_n_source"],
            fixture[f"{side}_n_target"],
            fixture[f"{side}_g_source"],
            fixture[f"{side}_g_target"],
            **weights,
            **betas,
        )
        _, pij = row_normalized_kernel_from_cost(cost, tau=float(fixture["tau"]))
        np.testing.assert_allclose(cost, fixture[f"{side}_cost"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(pij, fixture[f"{side}_pij"], rtol=1e-10, atol=1e-12)
        actual_pij[side] = pij

    expected = pd.read_csv(BASELINE_ROOT / "compare_N_kl_fixture_metrics.csv").iloc[0]
    ei_lower = effective_information(actual_pij["lower"])
    ei_upper = effective_information(actual_pij["upper"])
    np.testing.assert_allclose(ei_lower, expected["EI_lower"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ei_upper, expected["EI_upper"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ei_upper - ei_lower, expected["EI_gain"], rtol=1e-10, atol=1e-12)
