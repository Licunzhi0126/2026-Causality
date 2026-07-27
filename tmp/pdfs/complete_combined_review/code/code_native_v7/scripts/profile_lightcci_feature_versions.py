#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver, peek_h5ad_units, read_commot_index
from mignet_ce.pij.feature_versions.recipes import FEATURE_RECIPES, get_feature_recipe, recipe_sha256


def _memory_gib(entries: int, bytes_per_value: int = 8) -> float:
    return float(entries * bytes_per_value / (1024**3))


def _unit_count(resolver: LayerDataResolver, layer: str, organ: str, stage: str) -> tuple[int | None, str | None]:
    paths = resolver.paths(layer, organ, stage)
    if paths.cci_index.exists():
        return len(read_commot_index(paths.cci_index)), str(paths.cci_index)
    if paths.h5ad.exists():
        return len(peek_h5ad_units(paths.h5ad)), str(paths.h5ad)
    return None, None


def build_profile(
    *,
    data_root: Path,
    organs: list[str],
    time_points: list[str],
    level_pairs: list[VerticalPairSpec],
    recipe_ids: list[str],
) -> dict[str, object]:
    resolver = LayerDataResolver(data_root)
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for recipe_id in recipe_ids:
        recipe = get_feature_recipe(recipe_id)
        for organ in organs:
            for layer_pair in level_pairs:
                for side, layer in (("lower", layer_pair.lower_layer), ("upper", layer_pair.upper_layer)):
                    for source_stage, target_stage in combinations(time_points, 2):
                        n_source, source_path = _unit_count(resolver, layer, organ, source_stage)
                        n_target, target_path = _unit_count(resolver, layer, organ, target_stage)
                        if n_source is None or n_target is None:
                            missing.append(
                                {
                                    "recipe_id": recipe_id,
                                    "organ": organ,
                                    "layer": layer,
                                    "time_pair": f"{source_stage}->{target_stage}",
                                }
                            )
                            continue
                        cost_entries = int(n_source * n_target)
                        adjacency_entries = int(n_source * n_source + n_target * n_target)
                        block_count = len(recipe.fusion_weights)
                        rows.append(
                            {
                                "recipe_id": recipe_id,
                                "entry_method": recipe.entry_method,
                                "recipe_sha256": recipe_sha256(recipe),
                                "organ": organ,
                                "layer_pair": layer_pair.label(),
                                "side": side,
                                "layer": layer,
                                "time_pair": f"{source_stage}->{target_stage}",
                                "source_units": int(n_source),
                                "target_units": int(n_target),
                                "source_count_path": source_path,
                                "target_count_path": target_path,
                                "nmf_dense_adjacency_float64_gib": _memory_gib(adjacency_entries),
                                "one_cost_block_float64_gib": _memory_gib(cost_entries),
                                "cost_block_count": block_count,
                                "sequential_fusion_peak_float64_gib": _memory_gib(2 * cost_entries),
                                "sequential_fusion_model": "one fused accumulator plus one current normalized block",
                                "all_cost_blocks_if_retained_float64_gib": _memory_gib(block_count * cost_entries),
                                "grn_projected_feature_float64_gib": _memory_gib(
                                    (n_source + n_target) * 2 * recipe.projection_dim
                                ),
                                "recommendation": (
                                    "profile_with_blockwise_cost_and_monitor_peak_rss"
                                    if _memory_gib(cost_entries) >= 1.0
                                    else "dense_pair_cost_is_below_1_gib"
                                ),
                            }
                        )
    return {
        "data_root": str(data_root),
        "organs": organs,
        "time_points": time_points,
        "level_pairs": [pair.label() for pair in level_pairs],
        "recipes": recipe_ids,
        "estimates": rows,
        "missing_inputs": missing,
        "notes": {
            "time_pairs_are_not_parallelized": True,
            "cost_blocks_are_computed_and_fused_sequentially": True,
            "background_residual_matrix": False,
            "v4_consensus": False,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile LightCCI feature-version memory before a full run.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--organs", nargs="+", default=["heart"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument(
        "--level-pairs",
        nargs="+",
        default=["spot:seurat_k150", "spot:seurat_k40", "seurat_k150:seurat_k40"],
    )
    parser.add_argument("--recipes", nargs="+", choices=sorted(FEATURE_RECIPES), default=sorted(FEATURE_RECIPES))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = build_profile(
        data_root=args.data_root,
        organs=list(map(str, args.organs)),
        time_points=list(map(str, args.time_points)),
        level_pairs=[VerticalPairSpec.parse(value) for value in args.level_pairs],
        recipe_ids=list(map(str, args.recipes)),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote feature-version profile: {args.output}")
    else:
        print(text)
    return 0 if not payload["missing_inputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
