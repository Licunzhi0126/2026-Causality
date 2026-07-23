from __future__ import annotations

import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import ExpressionData, LayerPaths, read_grn_edges
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.light_cci_grn import (
    GRNStateResult,
    LightCCIGRNNetworkBuilder,
    deterministic_projection_matrix,
    double_end_grn_state,
    prepare_grn_inputs,
)

from .grn_residual import PearsonResidualConfig, positive_pearson_residual


PGR_CONFIG = PearsonResidualConfig(theta=1.0, positive_only=True, eps=1e-8)


def build_projected_pgr_state(
    prepared,
    *,
    output_dim: int,
    seed: int,
    gate_mode: str = "double_end",
) -> GRNStateResult:
    if gate_mode != "double_end":
        raise ValueError("gate_mode must be 'double_end'.")

    transformed = positive_pearson_residual(prepared.expression, config=PGR_CONFIG)
    regulator_state, target_state = double_end_grn_state(
        transformed,
        prepared.adjacency,
    )
    regulator_projection = deterministic_projection_matrix(
        prepared.genes,
        role="reg",
        output_dim=output_dim,
        seed=seed,
    )
    target_projection = deterministic_projection_matrix(
        prepared.genes,
        role="tar",
        output_dim=output_dim,
        seed=seed,
    )
    projected = regulator_state @ regulator_projection + target_state @ target_projection

    return GRNStateResult(
        projected=projected,
        regulator_state=regulator_state,
        target_state=target_state,
        metadata={
            **prepared.metadata,
            "grn_gate_mode": gate_mode,
            "grn_expression_transform": "positive_pearson_residual",
            "grn_pearson_theta": PGR_CONFIG.theta,
            "grn_pearson_positive_only": PGR_CONFIG.positive_only,
            "grn_transform_scope": "independent_within_layer_and_time",
            "uses_target_time_for_transform": False,
            "uses_cci_for_transform": False,
            "uses_pij_or_ei_for_transform": False,
            "grn_projection_seed": int(seed),
            "grn_state_dim": int(output_dim),
            "grn_state_shape": list(projected.shape),
        },
    )


class LightCCIGRNPGRNetworkBuilder(LightCCIGRNNetworkBuilder):
    """New network method; existing light_cci_grn remains untouched."""

    network_method = "light_cci_grn_pgr"
    grn_integration = "unit_grn_positive_pearson_residual_block_kl"

    def build_pair_context(self, *args, **kwargs) -> NetworkContext:
        context = super().build_pair_context(*args, **kwargs)
        context.network_method = self.network_method
        context.metadata.update(
            {
                "network_method": self.network_method,
                "feature_source": "light_cci_with_pgr_grn_payload",
                "grn_integration": self.grn_integration,
                "grn_expression_transform": "positive_pearson_residual",
                "grn_transform_scope": "independent_within_layer_and_time",
            }
        )
        return context

    def _augment_cci_graph(
        self,
        *,
        graph: LayerGraph,
        expression: ExpressionData,
        paths: LayerPaths,
        cfg: TemporalRunConfig,
    ) -> LayerGraph:
        if not paths.grn_edges.exists():
            raise FileNotFoundError(
                f"{self.network_method} requires a sample GRN for "
                f"{paths.layer} {paths.stage}: {paths.grn_edges}"
            )
        grn_edges = read_grn_edges(paths.grn_edges, top_k_targets_per_regulator=None)
        prepared = prepare_grn_inputs(
            expression.expr,
            graph.units,
            grn_edges,
            top_k_targets=cfg.grn_topk_targets,
        )
        state = build_projected_pgr_state(
            prepared,
            output_dim=cfg.grn_state_dim,
            seed=cfg.grn_projection_seed,
            gate_mode=cfg.grn_gate_mode,
        )
        graph.metadata.update(
            {
                "network_method": self.network_method,
                "uses_grn": True,
                "grn_integration": self.grn_integration,
                "grn_path": str(paths.grn_edges),
                "grn_state_csr": sp.csr_matrix(state.projected),
                "grn_state_units": list(prepared.units),
                "grn_state_shape": list(state.projected.shape),
                "grn_state_metadata": state.metadata,
            }
        )
        return graph
