from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PairFeatures, TimePair, TransitionKernels
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.feature_versions.distances import (
    distance_function,
    normalize_composition,
    row_softmax,
)
from mignet_ce.pij.feature_versions.export import (
    entropy_decomposition,
    export_pair_artifacts,
    export_run_manifest,
)
from mignet_ce.pij.feature_versions.fusion import fuse_cost_blocks
from mignet_ce.pij.feature_versions.grn_features import build_split_grn_features
from mignet_ce.pij.feature_versions.nmf_features import build_pairwise_nmf_features
from mignet_ce.pij.feature_versions.recipes import get_feature_recipe, recipe_sha256
from mignet_ce.pij.feature_versions.sources import (
    align_feature_to_context,
    load_cci_adjacency,
    load_merged_grn_state,
    load_raw_grn_inputs,
    output_side_units,
    side_layer,
    standardize_pair,
)
from mignet_ce.pij.feature_versions.spec import FeatureRecipe, PairFeatureBundle


def _l2_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > eps)


def _joint_robust_scalar(source: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([source[:, 0], target[:, 0]])
    median = float(np.median(combined)) if combined.size else 0.0
    q25, q75 = np.percentile(combined, [25.0, 75.0]) if combined.size else (0.0, 0.0)
    scale = float(q75 - q25) + eps
    return (source - median) / scale, (target - median) / scale


def _diagnostic_features(
    bundle: PairFeatureBundle,
    recipe: FeatureRecipe,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    names: list[str] = []
    for block_name, weight in recipe.fusion_weights.items():
        source = np.asarray(bundle.source_blocks[block_name], dtype=float)
        target = np.asarray(bundle.target_blocks[block_name], dtype=float)
        metric = str(recipe.block_distances[block_name])
        if metric == "kl":
            beta = float(recipe.distance_parameters[block_name]["beta"])
            source_ready = row_softmax(source, beta=beta)
            target_ready = row_softmax(target, beta=beta)
        elif metric in {"js", "hellinger"}:
            source_ready = normalize_composition(source)
            target_ready = normalize_composition(target)
        elif metric == "cosine":
            source_ready = _l2_rows(source)
            target_ready = _l2_rows(target)
        elif metric == "scalar_robust":
            source_ready, target_ready = _joint_robust_scalar(source, target)
        else:
            raise ValueError(f"Unsupported diagnostic feature metric {metric!r}.")
        scale = np.sqrt(float(weight))
        source_parts.append(source_ready * scale)
        target_parts.append(target_ready * scale)
        names.extend(f"{block_name}:{index + 1}" for index in range(source_ready.shape[1]))
    return np.hstack(source_parts), np.hstack(target_parts), names


def _cost_blocks(bundle: PairFeatureBundle, recipe: FeatureRecipe) -> dict[str, np.ndarray]:
    costs: dict[str, np.ndarray] = {}
    for block_name, metric in recipe.block_distances.items():
        function = distance_function(str(metric))
        parameters = dict(recipe.distance_parameters.get(block_name, {}))
        if metric in {"js", "hellinger"}:
            parameters.setdefault("pseudocount", 0.0)
        costs[block_name] = function(
            bundle.source_blocks[block_name],
            bundle.target_blocks[block_name],
            **parameters,
        )
    return costs


def _collect_context_input_paths(context: NetworkContext) -> list[Path]:
    paths: list[Path] = []
    for graph in [*context.lower_graphs, *context.upper_graphs]:
        for key in ("adjacency_path", "grn_path"):
            value = graph.metadata.get(key)
            if value:
                paths.append(Path(str(value)))
    return paths


class FeatureVersionPijMethod:
    name: str
    recipe_id: str

    def _validate_context(self, context: NetworkContext, recipe: FeatureRecipe) -> None:
        if context.network_method != "light_cci_grn":
            raise ValueError(f"{recipe.entry_method} requires network_method='light_cci_grn'.")
        if context.pair.lower_layer == "gene" or context.pair.upper_layer == "gene":
            raise ValueError(f"{recipe.entry_method} supports only non-gene layer pairs in the first implementation batch.")
        if self.name != recipe.entry_method:
            raise ValueError(f"Entry class name {self.name!r} does not match recipe method {recipe.entry_method!r}.")

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        recipe = get_feature_recipe(self.recipe_id)
        self._validate_context(context, recipe)
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)
        cci_cache: dict[tuple[str, int], tuple[object, dict[str, object]]] = {}
        grn_cache: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, object]]] = {}
        input_paths = _collect_context_input_paths(context)

        for side in ("lower", "upper"):
            for time_index in range(len(context.time_points)):
                cci_cache[(side, time_index)] = load_cci_adjacency(context, cfg, side, time_index)
                if recipe.grn_mode == "merged_projected_state":
                    merged, metadata = load_merged_grn_state(context, side, time_index)
                    grn_cache[(side, time_index)] = ({"g": merged}, metadata)
                elif recipe.grn_mode == "split_reg_tar_recomputed":
                    raw = load_raw_grn_inputs(context, cfg, side, time_index)
                    raw_blocks, metadata, _ = build_split_grn_features(raw, recipe)
                    aligned_blocks: dict[str, np.ndarray] = {}
                    alignments: dict[str, object] = {}
                    for name, values in raw_blocks.items():
                        aligned, alignment = align_feature_to_context(values, context, side, time_index)
                        aligned_blocks[name] = aligned
                        alignments[name] = alignment
                    metadata = {**metadata, "alignments": alignments}
                    grn_cache[(side, time_index)] = (aligned_blocks, metadata)
                    input_paths.extend([raw.h5ad_path, raw.grn_path])
                else:
                    raise ValueError(f"Unsupported GRN mode {recipe.grn_mode!r}.")

        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": recipe.entry_method,
                "entry_method": recipe.entry_method,
                "recipe_id": recipe.recipe_id,
                "recipe_sha256": recipe_sha256(recipe),
                "algorithm_version": recipe.algorithm_version,
                "feature_blocks": list(recipe.fusion_weights),
                "distance_per_block": dict(recipe.block_distances),
                "fusion_weights": dict(recipe.fusion_weights),
                "kernel_temperature": float(recipe.kernel_temperature),
                "row_stochastic": True,
                "transductive_pairwise_fit": True,
                "uses_target_for_nmf_fit": True,
                "uses_old_compare_N_kl_code": False,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        pairwise_lower: PairFeatures = {}
        pairwise_upper: PairFeatures = {}
        entropy_rows: list[dict[str, object]] = []
        feature_names: list[str] = []

        for pair in pairs:
            source_index, target_index = pair
            pair_label = f"{context.time_points[source_index]}->{context.time_points[target_index]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_kernels, target_features in (
                ("lower", kernels.p_lower, pairwise_lower),
                ("upper", kernels.p_upper, pairwise_upper),
            ):
                source_matrix, source_cci_metadata = cci_cache[(side, source_index)]
                target_matrix, target_cci_metadata = cci_cache[(side, target_index)]
                nmf = build_pairwise_nmf_features(
                    source_matrix,
                    target_matrix,
                    layer=side_layer(context, side),
                    recipe=recipe,
                )
                source_blocks: dict[str, np.ndarray] = {}
                target_blocks: dict[str, np.ndarray] = {}
                nmf_alignments: dict[str, object] = {}
                for name in nmf.source_blocks:
                    source_aligned, source_alignment = align_feature_to_context(
                        nmf.source_blocks[name], context, side, source_index
                    )
                    target_aligned, target_alignment = align_feature_to_context(
                        nmf.target_blocks[name], context, side, target_index
                    )
                    if recipe.entry_method == "compare_NG_kl_splitbeta_v1":
                        source_aligned, target_aligned, standardization = standardize_pair(
                            source_aligned, target_aligned
                        )
                    else:
                        standardization = {"mode": "semantic_feature_native_scale"}
                    source_blocks[name] = source_aligned
                    target_blocks[name] = target_aligned
                    nmf_alignments[name] = {
                        "source": source_alignment,
                        "target": target_alignment,
                        "standardization": standardization,
                    }

                source_grn, source_grn_metadata = grn_cache[(side, source_index)]
                target_grn, target_grn_metadata = grn_cache[(side, target_index)]
                for name in source_grn:
                    source_value = source_grn[name]
                    target_value = target_grn[name]
                    if recipe.entry_method == "compare_NG_kl_splitbeta_v1":
                        source_value, target_value, grn_standardization = standardize_pair(
                            source_value, target_value
                        )
                    else:
                        grn_standardization = {"mode": "separate_role_projection_native_scale"}
                    source_blocks[name] = source_value
                    target_blocks[name] = target_value

                bundle = PairFeatureBundle(
                    source_blocks=source_blocks,
                    target_blocks=target_blocks,
                    metadata={
                        "pair": pair_label,
                        "side": side,
                        "layer": side_layer(context, side),
                        "nmf": nmf.metadata,
                        "nmf_alignments": nmf_alignments,
                        "source_cci": source_cci_metadata,
                        "target_cci": target_cci_metadata,
                        "source_grn": source_grn_metadata,
                        "target_grn": target_grn_metadata,
                        "grn_standardization": grn_standardization,
                    },
                    artifacts=nmf.artifacts,
                )
                bundle.validate()
                if set(bundle.source_blocks) != set(recipe.fusion_weights):
                    raise ValueError(
                        f"Recipe blocks and extracted blocks differ: {sorted(recipe.fusion_weights)} vs "
                        f"{sorted(bundle.source_blocks)}."
                    )
                raw_costs = _cost_blocks(bundle, recipe)
                fused_cost, fusion_diagnostics, normalized_costs = fuse_cost_blocks(
                    raw_costs,
                    recipe.fusion_weights,
                    quantiles=recipe.normalization_quantiles,
                )
                _, pij = row_normalized_kernel_from_cost(fused_cost, tau=recipe.kernel_temperature)
                if not np.all(np.isfinite(pij)) or np.any(pij < 0.0):
                    raise ValueError("Feature-version PIJ is not finite and nonnegative.")
                if pij.shape[0] and not np.allclose(pij.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
                    raise ValueError("Feature-version PIJ is not row stochastic.")
                target_kernels[pair] = pij
                diagnostic_source, diagnostic_target, current_feature_names = _diagnostic_features(bundle, recipe)
                target_features[pair] = (diagnostic_source, diagnostic_target)
                if not feature_names:
                    feature_names = current_feature_names

                entropy = entropy_decomposition(pij)
                entropy_row: dict[str, object] = {
                    "entry_method": recipe.entry_method,
                    "recipe_id": recipe.recipe_id,
                    "organ": context.organ,
                    "layer_pair": context.pair.label(),
                    "time_pair": pair_label,
                    "side": side,
                    **entropy,
                }
                entropy_rows.append(entropy_row)
                kernels.kernel_metadata[pair_label][side] = {
                    "feature_blocks": list(bundle.source_blocks),
                    "source_shape": list(pij.shape),
                    "nmf": nmf.metadata,
                    "grn": {
                        "source": source_grn_metadata,
                        "target": target_grn_metadata,
                        "regulator_target_summed": False
                        if recipe.grn_mode == "split_reg_tar_recomputed"
                        else True,
                    },
                    "fusion": fusion_diagnostics,
                    "entropy": entropy,
                }
                if cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0:
                    kernels.kernel_diagnostics[side][pair] = {"main_cost": fused_cost}
                if should_export:
                    export_pair_artifacts(
                        cfg=cfg,
                        context=context,
                        recipe=recipe,
                        pair=pair,
                        side=side,
                        bundle=bundle,
                        raw_costs=raw_costs,
                        normalized_costs=normalized_costs,
                        fused_cost=fused_cost,
                        cost_diagnostics=fusion_diagnostics,
                        pij=pij,
                        entropy_row=entropy_row,
                    )

        if should_export:
            export_run_manifest(
                cfg=cfg,
                context=context,
                recipe=recipe,
                input_paths=input_paths,
                entropy_rows=entropy_rows,
            )
        lower_empty = [
            np.zeros((len(output_side_units(context, "lower", index)), 0), dtype=float)
            for index in range(len(context.time_points))
        ]
        upper_empty = [
            np.zeros((len(output_side_units(context, "upper", index)), 0), dtype=float)
            for index in range(len(context.time_points))
        ]
        result = MethodResult(
            lower_features=lower_empty,
            upper_features=upper_empty,
            lower_coords=context.lower_coords_by_time
            if context.feature_alignment_space == "native_units"
            else context.upper_coords_by_time,
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=pairwise_lower,
            pairwise_upper_features=pairwise_upper,
            method_metadata={
                "pij_method": recipe.entry_method,
                "entry_method": recipe.entry_method,
                "recipe_id": recipe.recipe_id,
                "recipe_sha256": recipe_sha256(recipe),
                "algorithm_version": recipe.algorithm_version,
                "feature_blocks": list(recipe.fusion_weights),
                "distance_per_block": dict(recipe.block_distances),
                "fusion_weights": dict(recipe.fusion_weights),
                "feature_names": feature_names,
                "feature_role": "pairwise_diagnostics_for_TE_DI_only",
                "pij_uses_explicit_fused_cost": True,
                "transductive_pairwise_fit": True,
                "uses_target_for_nmf_fit": True,
                "uses_old_compare_N_kl_code": False,
                "recipe_is_authoritative_over_legacy_cli_nmf_and_kl_fields": True,
            },
        )
        return result, kernels
