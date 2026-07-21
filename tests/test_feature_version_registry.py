from __future__ import annotations

from mignet_ce.pij.feature_versions.recipes import FEATURE_RECIPES, get_feature_recipe, recipe_sha256, recipe_to_dict
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY, get_pij_method


def test_feature_recipes_are_frozen_valid_and_match_yaml() -> None:
    assert set(FEATURE_RECIPES) == {"nkl_splitbeta_v1", "ncomp_gcos_v2", "nshape_gcos_v3"}
    for recipe_id, recipe in FEATURE_RECIPES.items():
        assert get_feature_recipe(recipe_id) is recipe
        assert len(recipe_sha256(recipe)) == 64
        assert sum(recipe.fusion_weights.values()) == 1.0
        assert set(recipe.block_distances) == set(recipe.fusion_weights)
        assert recipe_to_dict(recipe)["nmf_seeds"] == [42]


def test_feature_recipe_mappings_cannot_be_mutated() -> None:
    recipe = get_feature_recipe("ncomp_gcos_v2")
    try:
        recipe.fusion_weights["n_out"] = 0.0  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("Frozen feature recipe mapping was mutable.")


def test_feature_version_entries_are_registered_without_rebinding_old_method() -> None:
    expected = {
        "compare_NG_kl_splitbeta_v1",
        "compare_Ncomp_Gcos_v2",
        "compare_Nshape_Gcos_v3",
    }
    assert expected.issubset(PIJ_METHOD_REGISTRY)
    assert {get_pij_method(name).name for name in expected} == expected
    assert PIJ_METHOD_REGISTRY["compare_N_kl"].__name__ == "CompareNKlPijMethod"
