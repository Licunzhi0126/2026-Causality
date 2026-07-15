from __future__ import annotations

import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import ExpressionData, LayerPaths, read_grn_edges
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.grn_state import build_projected_grn_state, prepare_grn_inputs
from mignet_ce.networks.light_cci import LightCCINetworkBuilder


class LightCCIGRNNetworkBuilder(LightCCINetworkBuilder):
    network_method = "light_cci_grn"
    grn_integration = "unit_grn_state_block_kl"
    build_projected_state = True
    retain_joint_inputs = False

    def build_pair_context(self, *args, **kwargs) -> NetworkContext:
        context = super().build_pair_context(*args, **kwargs)
        context.metadata.update(
            {
                "network_method": self.network_method,
                "feature_source": "light_cci_with_grn_payload",
                "grn_integration": self.grn_integration,
                "grn_applies_to": "non_gene_unit_layers",
                "grn_gate_mode": "double_end",
                "uses_grn": True,
                "uses_cci": True,
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
                f"{self.network_method} requires a sample GRN for {paths.layer} {paths.stage}: {paths.grn_edges}"
            )
        grn_edges = read_grn_edges(paths.grn_edges, top_k_targets_per_regulator=None)
        prepared = prepare_grn_inputs(
            expression.expr,
            graph.units,
            grn_edges,
            top_k_targets=cfg.grn_topk_targets,
        )
        graph.metadata.update(
            {
                "network_method": self.network_method,
                "uses_grn": True,
                "grn_integration": self.grn_integration,
                "grn_path": str(paths.grn_edges),
                "grn_weight_mode": "abs",
                "grn_gate_mode": cfg.grn_gate_mode,
                "grn_input_metadata": prepared.metadata,
            }
        )
        if self.build_projected_state:
            state = build_projected_grn_state(
                prepared,
                output_dim=cfg.grn_state_dim,
                seed=cfg.grn_projection_seed,
                gate_mode=cfg.grn_gate_mode,
            )
            graph.metadata.update(
                {
                    "grn_state_csr": sp.csr_matrix(state.projected),
                    "grn_state_units": list(prepared.units),
                    "grn_state_shape": list(state.projected.shape),
                    "grn_state_metadata": state.metadata,
                }
            )
        if self.retain_joint_inputs:
            graph.metadata.update(
                {
                    "grn_genes": list(prepared.genes),
                    "grn_adjacency_csr": prepared.adjacency,
                    "grn_expression_csr": sp.csr_matrix(prepared.expression),
                    "grn_expression_units": list(prepared.units),
                }
            )
        return graph

    def _stage_summary(self, stage: str, lower_graph: LayerGraph, upper_graph: LayerGraph) -> dict[str, object]:
        summary = super()._stage_summary(stage, lower_graph, upper_graph)
        summary.update(
            {
                "grn_integration": self.grn_integration,
                "lower_grn_state_shape": lower_graph.metadata.get("grn_state_shape"),
                "upper_grn_state_shape": upper_graph.metadata.get("grn_state_shape"),
                "lower_grn_gene_count": lower_graph.metadata.get("grn_input_metadata", {}).get("gene_count"),
                "upper_grn_gene_count": upper_graph.metadata.get("grn_input_metadata", {}).get("gene_count"),
            }
        )
        return summary
