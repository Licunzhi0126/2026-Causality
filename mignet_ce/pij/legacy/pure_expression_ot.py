from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.legacy._ot_common import run_ot_pij_method_with_result
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.representations.expression_only import build_expression_only_feature_result


def _pure_expression_cosine_cost(source: np.ndarray, target: np.ndarray, eps: float) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError(f"source and target must be 2D arrays, got {source_arr.shape} and {target_arr.shape}.")
    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source_arr.shape[1]} != {target_arr.shape[1]}.")
    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        return np.zeros((source_arr.shape[0], target_arr.shape[0]), dtype=float)

    source_norms = np.linalg.norm(source_arr, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
    source_unit = np.divide(source_arr, source_norms, out=np.zeros_like(source_arr, dtype=float), where=source_norms > eps)
    target_unit = np.divide(target_arr, target_norms, out=np.zeros_like(target_arr, dtype=float), where=target_norms > eps)
    similarity = source_unit @ target_unit.T
    return np.clip(1.0 - similarity, 0.0, 2.0)


class PureExpressionOTPijMethod:
    name = "pure_expression_ot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        if context.network_method != "expression_only":
            raise ValueError("pure_expression_ot requires network_method='expression_only'.")

        result = build_expression_only_feature_result(
            context=context,
            cfg=cfg,
            n_components=cfg.pure_expression_pca_components,
            seed=cfg.nmf_seed,
        )

        def component_builder(
            source_features: np.ndarray,
            target_features: np.ndarray,
            source_coords: np.ndarray | None,
            target_coords: np.ndarray | None,
            space: str,
            t0: int,
            t1: int,
        ):
            cost = _pure_expression_cosine_cost(
                source_features,
                target_features,
                eps=cfg.pure_expression_cosine_eps,
            )
            return (
                {"pure_expression": cost},
                {"pure_expression": cfg.pij_expr_weight},
                {
                    "feature_source": "pure_expression",
                    "uses_grn": False,
                    "uses_cci": False,
                    "uses_legacy_graph": False,
                },
            )

        return run_ot_pij_method_with_result(
            context=context,
            cfg=cfg,
            pairs=pairs,
            method_name=self.name,
            component_builder=component_builder,
            result=result,
        )
