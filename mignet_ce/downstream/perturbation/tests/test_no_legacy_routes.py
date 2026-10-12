from __future__ import annotations

from pathlib import Path


PERTURBATION_ROOT = Path(__file__).resolve().parents[1]


def _production_sources() -> list[Path]:
    return sorted(path for path in PERTURBATION_ROOT.glob("*.py") if path.name != "__init__.py")


def test_production_sources_do_not_restore_legacy_routes() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in _production_sources())
    legacy_routes = (
        "ng_" + "condition_adapter",
        "pij_" + "bridge",
        "build_state_" + "macro_pij_builder",
        "build_existing_" + "ng_macro_callback",
        "run_prepared_" + "condition",
    )
    assert not any(route in source for route in legacy_routes)


def test_search_based_perturbation_modules_are_absent() -> None:
    prohibited = ("search.py", "feasible_range.py", "strength.py", "objectives.py", "candidates.py", "edge_ops.py")
    assert not any((PERTURBATION_ROOT / name).exists() for name in prohibited)


def test_optimal_bridge_uses_direct_state_formal_global_training() -> None:
    source = (PERTURBATION_ROOT / "optimal_bridge.py").read_text(encoding="utf-8")
    assert "maturity_cci_grn_two_stage" in source
    assert "downstream." + "application" not in source
    assert "run_local_" + "pair_analysis" not in source
    assert "prepare_direct_state_two_stage" in source
    assert "train_" + "deltaei" in source


def test_perturbation_does_not_build_macro_g_or_macro_pij() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in _production_sources())
    assert "Pool" + "(G)" not in source
    assert "macro " + "PIJ" not in source
