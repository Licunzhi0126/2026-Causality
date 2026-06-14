from __future__ import annotations

from mignet_ce.config import PIJ_METHODS
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def test_pij_registry_contains_all_four_methods() -> None:
    assert set(PIJ_METHOD_REGISTRY) == PIJ_METHODS
    assert {get_pij_method(name).name for name in PIJ_METHODS} == PIJ_METHODS
