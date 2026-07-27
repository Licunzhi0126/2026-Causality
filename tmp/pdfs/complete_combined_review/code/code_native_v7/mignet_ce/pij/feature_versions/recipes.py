from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from mignet_ce.pij.feature_versions.spec import FeatureRecipe


REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_ROOT = REPO_ROOT / "configs" / "feature_versions"


FEATURE_RECIPES: dict[str, FeatureRecipe] = {
    "nkl_splitbeta_v1": FeatureRecipe(
        recipe_id="nkl_splitbeta_v1",
        entry_method="compare_NG_kl_splitbeta_v1",
        algorithm_version="lightcci_feature_versions_v1",
        cci_mode="legacy_pairwise_by_layer",
        grn_mode="merged_projected_state",
        block_distances={"n": "kl", "g": "kl"},
        distance_parameters={"n": {"beta": 0.20}, "g": {"beta": 1.00}},
        fusion_weights={"n": 0.75, "g": 0.25},
        nmf_rank=5,
        nmf_max_iter=300,
        nmf_seeds=(42,),
        grn_topk_targets=50,
        projection_dim=64,
        projection_seed=20260713,
        kernel_temperature=1.0,
        pseudocount=1e-8,
    ),
    "ncomp_gcos_v2": FeatureRecipe(
        recipe_id="ncomp_gcos_v2",
        entry_method="compare_Ncomp_Gcos_v2",
        algorithm_version="lightcci_feature_versions_v2",
        cci_mode="directed_shared_core_all_non_gene",
        grn_mode="split_reg_tar_recomputed",
        block_distances={"n_out": "js", "n_in": "js", "g_reg": "cosine", "g_tar": "cosine"},
        distance_parameters={},
        fusion_weights={"n_out": 0.30, "n_in": 0.30, "g_reg": 0.20, "g_tar": 0.20},
        nmf_rank=5,
        nmf_max_iter=300,
        nmf_seeds=(42,),
        grn_topk_targets=50,
        projection_dim=64,
        projection_seed=20260713,
        kernel_temperature=1.0,
        pseudocount=1e-8,
    ),
    "nshape_gcos_v3": FeatureRecipe(
        recipe_id="nshape_gcos_v3",
        entry_method="compare_Nshape_Gcos_v3",
        algorithm_version="lightcci_feature_versions_v3",
        cci_mode="directed_shared_core_all_non_gene",
        grn_mode="split_reg_tar_recomputed",
        block_distances={
            "n_out_shape": "js",
            "n_in_shape": "js",
            "n_out_strength": "scalar_robust",
            "n_in_strength": "scalar_robust",
            "g_reg": "cosine",
            "g_tar": "cosine",
        },
        distance_parameters={},
        fusion_weights={
            "n_out_shape": 0.22,
            "n_in_shape": 0.22,
            "n_out_strength": 0.08,
            "n_in_strength": 0.08,
            "g_reg": 0.20,
            "g_tar": 0.20,
        },
        nmf_rank=5,
        nmf_max_iter=300,
        nmf_seeds=(42,),
        grn_topk_targets=50,
        projection_dim=64,
        projection_seed=20260713,
        kernel_temperature=1.0,
        pseudocount=1e-8,
    ),
}


def recipe_to_dict(recipe: FeatureRecipe) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "entry_method": recipe.entry_method,
        "algorithm_version": recipe.algorithm_version,
        "cci_mode": recipe.cci_mode,
        "grn_mode": recipe.grn_mode,
        "block_distances": dict(recipe.block_distances),
        "distance_parameters": {key: dict(value) for key, value in recipe.distance_parameters.items()},
        "fusion_weights": {key: float(value) for key, value in recipe.fusion_weights.items()},
        "nmf_rank": int(recipe.nmf_rank),
        "nmf_max_iter": int(recipe.nmf_max_iter),
        "nmf_seeds": [int(seed) for seed in recipe.nmf_seeds],
        "grn_topk_targets": int(recipe.grn_topk_targets),
        "projection_dim": int(recipe.projection_dim),
        "projection_seed": int(recipe.projection_seed),
        "kernel_temperature": float(recipe.kernel_temperature),
        "pseudocount": float(recipe.pseudocount),
        "normalization_quantiles": [float(value) for value in recipe.normalization_quantiles],
    }


def canonical_recipe_bytes(recipe: FeatureRecipe) -> bytes:
    return json.dumps(recipe_to_dict(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def recipe_sha256(recipe: FeatureRecipe) -> str:
    return hashlib.sha256(canonical_recipe_bytes(recipe)).hexdigest()


def _normalized_yaml_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["nmf_seeds"] = [int(seed) for seed in normalized.get("nmf_seeds", [])]
    normalized["normalization_quantiles"] = [
        float(value) for value in normalized.get("normalization_quantiles", [5.0, 95.0])
    ]
    normalized["fusion_weights"] = {
        str(key): float(value) for key, value in dict(normalized.get("fusion_weights", {})).items()
    }
    normalized["block_distances"] = {
        str(key): str(value) for key, value in dict(normalized.get("block_distances", {})).items()
    }
    normalized["distance_parameters"] = {
        str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in dict(value).items()}
        for key, value in dict(normalized.get("distance_parameters", {})).items()
    }
    return normalized


def validate_yaml_matches_recipe(recipe: FeatureRecipe) -> Path:
    path = RECIPE_ROOT / f"{recipe.recipe_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Feature recipe YAML is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Feature recipe YAML must contain a mapping: {path}")
    expected = recipe_to_dict(recipe)
    actual = _normalized_yaml_payload(payload)
    if actual != expected:
        raise ValueError(f"YAML and Python feature recipes differ for {recipe.recipe_id}.")
    return path


def get_feature_recipe(recipe_id: str, *, validate_yaml: bool = True) -> FeatureRecipe:
    try:
        recipe = FEATURE_RECIPES[recipe_id]
    except KeyError as exc:
        raise ValueError(f"Unknown feature recipe {recipe_id!r}; expected one of {sorted(FEATURE_RECIPES)}.") from exc
    if validate_yaml:
        validate_yaml_matches_recipe(recipe)
    return recipe
