from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels

from .config import AblationConfig
from .costs import build_controlled_cost
from .features import AblationFeatureProvider
from .methods import AblationMethodSpec
from .transitions import transition_from_cost


class ControlledFeatureAblationMethod:
    """PijMethod-compatible controlled feature/OT ablation entry point."""

    def __init__(self, spec: AblationMethodSpec, ablation_cfg: AblationConfig | None = None) -> None:
        self.spec = spec
        self.name = spec.method_id
        self.ablation_cfg = ablation_cfg or AblationConfig()
        self.ablation_cfg.validate()

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels]:
        provider = AblationFeatureProvider(context, cfg)
        kernels = TransitionKernels(
            kernel_metadata={
                "method_id": self.spec.method_id,
                "input_group": self.spec.input_group,
                "feature_method": self.spec.feature_method,
                "pij_construction": self.spec.pij_construction,
                "ablation_contract": self.ablation_cfg.contract(),
            }
        )
        pairwise_lower: dict[TimePair, tuple[np.ndarray, np.ndarray]] = {}
        pairwise_upper: dict[TimePair, tuple[np.ndarray, np.ndarray]] = {}
        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, target_dict, feature_dict in (
                ("lower", kernels.p_lower, pairwise_lower),
                ("upper", kernels.p_upper, pairwise_upper),
            ):
                blocks = provider.pair_blocks(
                    side=side,
                    pair=pair,
                    cci_representation=self.spec.cci_representation,
                    use_grn=self.spec.use_grn,
                )
                cost_result = build_controlled_cost(blocks, self.ablation_cfg)
                transition = transition_from_cost(
                    cost_result.cost,
                    use_ot=self.spec.use_ot,
                    cfg=self.ablation_cfg,
                )
                target_dict[pair] = transition.pij
                kernels.kernel_diagnostics[side][pair] = {"main_cost": cost_result.cost}
                components = [
                    arr for arr in (blocks.cci_source, blocks.grn_source) if arr is not None
                ]
                targets = [
                    arr for arr in (blocks.cci_target, blocks.grn_target) if arr is not None
                ]
                feature_dict[pair] = (
                    np.hstack(components) if len(components) > 1 else components[0].copy(),
                    np.hstack(targets) if len(targets) > 1 else targets[0].copy(),
                )
                kernels.kernel_metadata[pair_label][side] = {
                    "cost": cost_result.metadata,
                    "transition": transition.metadata,
                    "feature_metadata": blocks.metadata,
                }

        lower_empty = [np.zeros((len(units), 0), dtype=float) for units in context.lower_units_by_time]
        upper_empty = [np.zeros((len(units), 0), dtype=float) for units in context.upper_units_by_time]
        result = MethodResult(
            lower_features=lower_empty,
            upper_features=upper_empty,
            lower_coords=(
                context.lower_coords_by_time
                if context.feature_alignment_space == "native_units"
                else context.upper_coords_by_time
            ),
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=pairwise_lower,
            pairwise_upper_features=pairwise_upper,
            method_metadata={
                "representation": "controlled_pij_feature_ablation_v1",
                "method_id": self.spec.method_id,
                "input_group": self.spec.input_group,
                "feature_method": self.spec.feature_method,
                "pij_construction": self.spec.pij_construction,
                "ablation_contract": self.ablation_cfg.contract(),
            },
        )
        return result, kernels
