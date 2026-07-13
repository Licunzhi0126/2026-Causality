from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, PairFeatures, TimePair, TransitionKernels
from mignet_ce.pij.compare.common import export_compare_pair_artifacts
from mignet_ce.pij.compare.distances import (
    pairwise_scalar_absolute_distance,
    pairwise_vector_distance,
    robust_normalize_cost,
    summarize_dense_cost,
)
from mignet_ce.pij.compare.features import CompareFeatureSet, build_compare_feature_set
from mignet_ce.pij.compare.sparse_ot import run_sparse_semi_relaxed_ot_from_cost


_ALLOWED_COMPONENT_KEYS = ("L", "E", "Sr")
_COMPONENT_WEIGHT_FIELDS = {
    "L": "compare_cost_weight_l",
    "E": "compare_cost_weight_e",
    "Sr": "compare_cost_weight_sr",
}


def _component_distance_rule(component_key: str, vector_metric: str) -> str:
    return "scalar_absolute_difference" if component_key == "Sr" else f"vector_{vector_metric}_distance"


def _select_source_target(
    feature_set: CompareFeatureSet,
    side: str,
    pair: TimePair,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if side == "lower":
        feature_lists = feature_set.lower_features
        pairwise = feature_set.pairwise_lower_features
    elif side == "upper":
        feature_lists = feature_set.upper_features
        pairwise = feature_set.pairwise_upper_features
    else:
        raise ValueError("side must be one of 'lower' or 'upper'.")
    if pairwise is not None and pair in pairwise:
        source, target = pairwise[pair]
        return np.asarray(source, dtype=float), np.asarray(target, dtype=float), True
    return (
        np.asarray(feature_lists[pair[0]], dtype=float),
        np.asarray(feature_lists[pair[1]], dtype=float),
        False,
    )


def _build_fused_pre_cost(
    component_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    vector_metric: str,
    weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    if not component_pairs:
        raise ValueError("component_pairs cannot be empty.")
    if vector_metric not in {"cosine", "euclidean"}:
        raise ValueError("vector_metric must be one of ['cosine', 'euclidean'].")

    total_weight = sum(float(weights[key]) for key in component_pairs)
    if any(float(weights[key]) < 0.0 for key in component_pairs):
        raise ValueError("Cost-fusion component weights must be nonnegative.")
    if total_weight <= 0.0:
        raise ValueError("At least one active cost-fusion component weight must be positive.")

    fused_cost: np.ndarray | None = None
    component_metadata: dict[str, object] = {}
    expected_shape: tuple[int, int] | None = None
    for key, (source, target) in component_pairs.items():
        if key not in _ALLOWED_COMPONENT_KEYS:
            raise ValueError(f"Unsupported cost-fusion component {key!r}.")
        source_arr = np.asarray(source, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        if key == "Sr":
            component_cost = pairwise_scalar_absolute_distance(source_arr, target_arr)
        else:
            component_cost = pairwise_vector_distance(source_arr, target_arr, vector_metric)
        if expected_shape is None:
            expected_shape = component_cost.shape
        elif component_cost.shape != expected_shape:
            raise ValueError(
                f"Cost component matrix shapes differ: expected {expected_shape}, "
                f"got {key}={component_cost.shape}."
            )

        component_cost, normalization = robust_normalize_cost(component_cost, copy=False)
        component_metadata[key] = {
            "weight": float(weights[key]),
            "distance_rule": _component_distance_rule(key, vector_metric),
            "source_shape": list(source_arr.shape),
            "target_shape": list(target_arr.shape),
            "normalization": normalization,
        }
        np.multiply(component_cost, float(weights[key]), out=component_cost)
        if fused_cost is None:
            fused_cost = component_cost
        else:
            fused_cost += component_cost
        del component_cost

    assert fused_cost is not None
    fused_cost /= total_weight
    metadata: dict[str, object] = {
        "vector_metric": vector_metric,
        "component_keys": list(component_pairs),
        "component_weights": {key: float(weights[key]) for key in component_pairs},
        "component_distance_rules": {
            key: _component_distance_rule(key, vector_metric) for key in component_pairs
        },
        "component_normalization": "robust_5_95_before_fusion",
        "candidate_cost_rescaling": "existing_candidate_minmax",
        "components": component_metadata,
        "fused_pre_cost_summary": summarize_dense_cost(fused_cost),
    }
    return fused_cost, metadata


def _diagnostic_feature_set(
    component_feature_sets: Mapping[str, CompareFeatureSet],
    weights: Mapping[str, float],
    pairs: Sequence[TimePair],
) -> CompareFeatureSet:
    keys = tuple(component_feature_sets)

    def combine_timewise(side: str) -> list[np.ndarray]:
        first = component_feature_sets[keys[0]]
        count = len(first.lower_features if side == "lower" else first.upper_features)
        output: list[np.ndarray] = []
        for time_index in range(count):
            parts = []
            for key in keys:
                feature_set = component_feature_sets[key]
                matrices = feature_set.lower_features if side == "lower" else feature_set.upper_features
                parts.append(np.asarray(matrices[time_index], dtype=float) * np.sqrt(float(weights[key])))
            output.append(np.hstack(parts))
        return output

    def combine_pairwise(side: str) -> PairFeatures | None:
        output: PairFeatures = {}
        for pair in pairs:
            selected = [_select_source_target(feature_set, side, pair) for feature_set in component_feature_sets.values()]
            if not any(pairwise_used for _, _, pairwise_used in selected):
                continue
            source_parts = [source * np.sqrt(float(weights[key])) for key, (source, _, _) in zip(keys, selected)]
            target_parts = [target * np.sqrt(float(weights[key])) for key, (_, target, _) in zip(keys, selected)]
            output[pair] = (np.hstack(source_parts), np.hstack(target_parts))
        return output or None

    feature_names = [
        f"diagnostic_{key}:{name}"
        for key, feature_set in component_feature_sets.items()
        for name in feature_set.feature_names
    ]
    artifacts: dict[str, dict[str, dict[str, object]]] = {"lower": {}, "upper": {}}
    for feature_set in component_feature_sets.values():
        for side in ("lower", "upper"):
            artifacts[side].update(feature_set.artifacts.get(side, {}))
    return CompareFeatureSet(
        lower_features=combine_timewise("lower"),
        upper_features=combine_timewise("upper"),
        feature_names=feature_names,
        metadata={
            "component_keys": list(keys),
            "component_feature_metadata": {
                key: feature_set.metadata for key, feature_set in component_feature_sets.items()
            },
            "transition_construction": "cost_mix",
            "method_result_feature_role": "diagnostics_and_TE_DI_only",
            "method_result_features_used_for_P": False,
            "diagnostic_combination_rule": "concat_sqrt_cost_weighted_standardized_components",
        },
        artifacts=artifacts,
        pairwise_lower_features=combine_pairwise("lower"),
        pairwise_upper_features=combine_pairwise("upper"),
    )


class CompareCostFusionSotBase:
    name: str
    component_keys: tuple[str, ...]
    vector_metric: str

    def _weights(self, cfg: TemporalRunConfig) -> dict[str, float]:
        if not self.component_keys or len(set(self.component_keys)) != len(self.component_keys):
            raise ValueError(f"{self.name}: component_keys must be nonempty and unique.")
        unknown = set(self.component_keys) - set(_ALLOWED_COMPONENT_KEYS)
        if unknown:
            raise ValueError(f"{self.name}: unsupported component keys {sorted(unknown)}.")
        if self.vector_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"{self.name}: unsupported vector_metric {self.vector_metric!r}.")
        weights = {
            key: float(getattr(cfg, _COMPONENT_WEIGHT_FIELDS[key]))
            for key in self.component_keys
        }
        if any(value < 0.0 for value in weights.values()):
            raise ValueError(f"{self.name}: component weights must be nonnegative; got {weights}.")
        if sum(weights.values()) <= 0.0:
            raise ValueError(f"{self.name}: at least one active component weight must be positive; got {weights}.")
        return weights

    def _validate_component_rows(
        self,
        *,
        context: NetworkContext,
        pair: TimePair,
        side: str,
        component_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        shapes = {
            key: {"source": list(np.asarray(source).shape), "target": list(np.asarray(target).shape)}
            for key, (source, target) in component_pairs.items()
        }
        source_rows = {np.asarray(source).shape[0] for source, _ in component_pairs.values()}
        target_rows = {np.asarray(target).shape[0] for _, target in component_pairs.values()}
        if len(source_rows) != 1 or len(target_rows) != 1:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            raise ValueError(
                f"{self.name}: component row mismatch for organ={context.organ}, "
                f"layer_pair={context.pair.label()}, time_pair={pair_label}, side={side}; shapes={shapes}."
            )

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        weights = self._weights(cfg)
        component_feature_sets = {
            key: build_compare_feature_set(
                context,
                cfg,
                (key,),
                apply_feature_weights=False,
            )
            for key in self.component_keys
        }
        for pair in pairs:
            for side in ("lower", "upper"):
                validation_pairs = {
                    key: _select_source_target(feature_set, side, pair)[:2]
                    for key, feature_set in component_feature_sets.items()
                }
                self._validate_component_rows(
                    context=context,
                    pair=pair,
                    side=side,
                    component_pairs=validation_pairs,
                )
        diagnostic_features = _diagnostic_feature_set(component_feature_sets, weights, pairs)
        component_rules = {
            key: _component_distance_rule(key, self.vector_metric) for key in self.component_keys
        }
        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "fusion_mode": "cost_mix",
                "transition_construction": "cost_mix",
                "vector_metric": self.vector_metric,
                "component_keys": list(self.component_keys),
                "component_weights": weights,
                "component_distance_rules": component_rules,
                "component_normalization": "robust_5_95_before_fusion",
                "candidate_cost_rescaling": "existing_candidate_minmax",
                "method_result_feature_role": "diagnostics_and_TE_DI_only",
                "method_result_features_used_for_P": False,
                "row_stochastic": True,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_dict in (("lower", kernels.p_lower), ("upper", kernels.p_upper)):
                component_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                feature_sources: dict[str, str] = {}
                for key, feature_set in component_feature_sets.items():
                    source, target, pairwise_used = _select_source_target(feature_set, side, pair)
                    component_pairs[key] = (source, target)
                    feature_sources[key] = "pairwise" if pairwise_used else "timewise"
                self._validate_component_rows(
                    context=context,
                    pair=pair,
                    side=side,
                    component_pairs=component_pairs,
                )
                fused_pre_cost, fusion_metadata = _build_fused_pre_cost(
                    component_pairs,
                    self.vector_metric,
                    weights,
                )
                ot_result = run_sparse_semi_relaxed_ot_from_cost(
                    fused_pre_cost,
                    epsilon=cfg.ot_epsilon,
                    gamma=cfg.ot_gamma,
                    max_iter=cfg.ot_max_iter,
                    source_k=cfg.ot_dist_k,
                    target_k=cfg.ot_sim_k,
                    raw_cost_column="raw_fused_pre_cost",
                    cost_source=f"cost_mix:{'+'.join(self.component_keys)}:{self.vector_metric}",
                )
                target_dict[pair] = ot_result.pij_row_normalized_sparse.toarray()
                source_shape = list(next(iter(component_pairs.values()))[0].shape)
                target_shape = list(next(iter(component_pairs.values()))[1].shape)
                pair_metadata = {
                    "pij_method": self.name,
                    "fusion_mode": "cost_mix",
                    "transition_construction": "cost_mix",
                    **fusion_metadata,
                    "component_feature_sources": feature_sources,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                    "candidate_edges": int(ot_result.cost_sparse.nnz),
                    "ot_convergence": ot_result.convergence,
                    "row_stochastic": True,
                }
                kernels.kernel_metadata[pair_label][side] = pair_metadata
                if cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0:
                    kernels.kernel_diagnostics[side][pair] = {"main_cost": fused_pre_cost}

                if should_export:
                    diagnostic_source, diagnostic_target, _ = _select_source_target(
                        diagnostic_features,
                        side,
                        pair,
                    )
                    export_compare_pair_artifacts(
                        cfg=cfg,
                        context=context,
                        method_name=self.name,
                        feature_keys=self.component_keys,
                        pij_key="sot",
                        feature_set=diagnostic_features,
                        pair=pair,
                        side=side,
                        source_features=diagnostic_source,
                        target_features=diagnostic_target,
                        raw_sparse=ot_result.transport_sparse,
                        pij_sparse=ot_result.pij_row_normalized_sparse,
                        diagnostics={
                            "kind": "cost_fusion_sparse_semi_relaxed_ot",
                            **pair_metadata,
                        },
                        sparse_ot_result=ot_result,
                        metadata_extra=pair_metadata,
                    )

        return (
            MethodResult(
                lower_features=diagnostic_features.lower_features,
                upper_features=diagnostic_features.upper_features,
                lower_coords=(
                    context.lower_coords_by_time
                    if context.feature_alignment_space == "native_units"
                    else context.upper_coords_by_time
                ),
                upper_coords=context.upper_coords_by_time,
                pairwise_lower_features=diagnostic_features.pairwise_lower_features,
                pairwise_upper_features=diagnostic_features.pairwise_upper_features,
                method_metadata={
                    "pij_method": self.name,
                    "representation": "lightcci_compare_cost_fusion",
                    "fusion_mode": "cost_mix",
                    "transition_construction": "cost_mix",
                    "vector_metric": self.vector_metric,
                    "component_keys": list(self.component_keys),
                    "component_weights": weights,
                    "component_distance_rules": component_rules,
                    "component_normalization": "robust_5_95_before_fusion",
                    "candidate_cost_rescaling": "existing_candidate_minmax",
                    "method_result_feature_role": "diagnostics_and_TE_DI_only",
                    "method_result_features_used_for_P": False,
                    "feature_names": diagnostic_features.feature_names,
                    "feature_metadata": diagnostic_features.metadata,
                },
            ),
            kernels,
        )
