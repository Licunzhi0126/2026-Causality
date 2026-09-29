from __future__ import annotations

"""CCI + H5AD regulatory-activity similarity network.

Migration sources:
  - reference/network_only_coarse_grain/scripts/build_integrated_cci_grn_network_v58.py
  - reference/network_only_coarse_grain/extract_grn_obs_features_v48.py

The WYT source called the activity-similarity component "GRN". This adapter
uses the scientifically explicit RegSim name and never reads gene-gene GRN
edge tables.
"""

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import (
    ExpressionData,
    LayerDataResolver,
    LayerPaths,
    read_commot_index,
)
from mignet_ce.io.regsim_h5ad import (
    RegSimFeatureBlock,
    build_regsim_feature_block,
    common_regulatory_features,
)
from mignet_ce.networks.base import NetworkContext
from mignet_ce.networks.light_cci import (
    LightCCINetworkBuilder,
    _cci_edges_from_adjacency,
)


EPS = 1e-12


def row_normalize_sparse(matrix: sp.spmatrix) -> sp.csr_matrix:
    values = matrix.tocsr(copy=True).astype(np.float32)
    if values.nnz:
        values.data[~np.isfinite(values.data)] = 0.0
        values.data[values.data < 0.0] = 0.0
        values.eliminate_zeros()
    row_sum = np.asarray(values.sum(axis=1)).ravel()
    inverse = 1.0 / (row_sum + EPS)
    return (sp.diags(inverse.astype(np.float32)) @ values).tocsr()


def build_regsim_similarity_network(
    features: np.ndarray,
    *,
    k: int = 50,
) -> sp.csr_matrix:
    """Exact v58 StandardScaler -> cosine kNN -> max-symmetrize -> RowNorm."""
    values = np.asarray(features, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"RegSim features must be a non-empty 2D matrix; got {values.shape}.")
    if int(k) <= 0:
        raise ValueError("RegSim k must be positive.")
    values[~np.isfinite(values)] = 0.0
    standardized = StandardScaler().fit_transform(values).astype(np.float32)
    unit_count = standardized.shape[0]
    effective_k = min(int(k) + 1, unit_count)
    neighbors = NearestNeighbors(
        n_neighbors=effective_k,
        metric="cosine",
        algorithm="auto",
    )
    neighbors.fit(standardized)
    distances, indices = neighbors.kneighbors(standardized)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for source in range(unit_count):
        for distance, target in zip(distances[source], indices[source]):
            if int(target) == source:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            if similarity <= 0.0:
                continue
            rows.append(source)
            columns.append(int(target))
            data.append(similarity)
    graph = sp.csr_matrix(
        (data, (rows, columns)),
        shape=(unit_count, unit_count),
        dtype=np.float32,
    )
    graph = graph.maximum(graph.T)
    return row_normalize_sparse(graph)


def integrate_cci_regsim(
    cci: sp.spmatrix,
    regsim: sp.spmatrix,
    *,
    regsim_weight: float = 0.2,
) -> sp.csr_matrix:
    weight = float(regsim_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("regsim_weight must be within [0, 1].")
    cci_normalized = row_normalize_sparse(cci)
    regsim_normalized = row_normalize_sparse(regsim)
    if cci_normalized.shape != regsim_normalized.shape:
        raise ValueError(
            f"CCI and RegSim shapes differ: {cci_normalized.shape} vs {regsim_normalized.shape}."
        )
    integrated = (1.0 - weight) * cci_normalized + weight * regsim_normalized
    return row_normalize_sparse(integrated).astype(np.float32).tocsr()


class WYTCCIRegSimNetworkBuilder(LightCCINetworkBuilder):
    network_method = "wyt_cci_regsim"

    def __init__(self) -> None:
        self._regsim_blocks: dict[tuple[str, str], RegSimFeatureBlock] = {}

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        if "gene" in {pair.lower_layer, pair.upper_layer}:
            raise ValueError(
                "wyt_cci_regsim supports spot/domain unit layers only; it never reads a gene-gene GRN."
            )
        self._regsim_blocks = {}
        for layer in dict.fromkeys((pair.lower_layer, pair.upper_layer)):
            stage_paths = [
                resolver.paths(layer, organ, str(stage))
                for stage in cfg.time_points
            ]
            activity_paths = [
                paths.h5ad
                if layer == "spot"
                else self._require_spots_with_domain(paths)
                for paths in stage_paths
            ]
            common_features = common_regulatory_features(activity_paths)
            for paths in stage_paths:
                if not paths.cci_index.exists():
                    raise FileNotFoundError(
                        f"wyt_cci_regsim requires the CCI index for unit alignment: {paths.cci_index}"
                    )
                cci_units = read_commot_index(paths.cci_index)
                self._regsim_blocks[(layer, str(paths.stage))] = build_regsim_feature_block(
                    unit_h5ad_path=paths.h5ad,
                    cci_unit_ids=cci_units,
                    feature_names=common_features,
                    spots_with_domain_h5ad_path=(
                        None if layer == "spot" else self._require_spots_with_domain(paths)
                    ),
                )
        context = super().build_pair_context(organ, pair, cfg, resolver)
        context.feature_blocks = {
            "regsim": sorted(
                {
                    feature
                    for block in self._regsim_blocks.values()
                    for feature in block.feature_names
                }
            )
        }
        context.metadata.update(
            {
                "network_method": self.network_method,
                "feature_source": "cci_plus_h5ad_regsim",
                "regsim_knn_k": int(cfg.regsim_knn_k),
                "regsim_weight": float(cfg.regsim_weight),
                "uses_grn": False,
                "uses_true_grn": False,
                "uses_cci": True,
                "regsim_semantics": "unit regulatory-activity cosine similarity",
            }
        )
        return context

    @staticmethod
    def _require_spots_with_domain(paths: LayerPaths):
        path = paths.spots_with_domain_h5ad
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Domain RegSim requires the corresponding spots-with-domain H5AD: {path}"
            )
        return path

    def _augment_cci_graph(
        self,
        *,
        graph: LayerGraph,
        expression: ExpressionData,
        paths: LayerPaths,
        cfg: TemporalRunConfig,
    ) -> LayerGraph:
        block = self._regsim_blocks.get((paths.layer, str(paths.stage)))
        if block is None:
            raise RuntimeError(
                f"RegSim block was not prepared for {paths.layer} {paths.stage}."
            )
        if block.unit_ids != list(map(str, graph.units)):
            raise ValueError(
                f"RegSim unit order does not match graph/CCI order for {paths.layer} {paths.stage}."
            )
        cci_stored = graph.metadata.get("adjacency_csr")
        if cci_stored is None:
            raise ValueError("LightCCI graph metadata is missing adjacency_csr.")
        cci = cci_stored.tocsr() if sp.issparse(cci_stored) else sp.csr_matrix(cci_stored)
        regsim = build_regsim_similarity_network(
            block.values,
            k=cfg.regsim_knn_k,
        )
        integrated = integrate_cci_regsim(
            cci,
            regsim,
            regsim_weight=cfg.regsim_weight,
        )
        integrated_edges = _cci_edges_from_adjacency(
            integrated,
            graph.units,
            graph.layer,
            graph.time_point,
        )
        if not integrated_edges.empty:
            integrated_edges.loc[:, "edge_type"] = "cci_regsim"
        graph.inter_edges = integrated_edges
        graph.metadata.update(
            {
                "network_method": self.network_method,
                "edge_source": "cci_regsim",
                "adjacency_source": "row_normalized_cci_plus_h5ad_regsim",
                "adjacency_csr": integrated,
                "adjacency_shape": list(integrated.shape),
                "adjacency_nnz": int(integrated.nnz),
                "cci_adjacency_csr": cci,
                "cci_total_path": str(paths.cci_total),
                "cci_index_path": str(paths.cci_index),
                "unit_h5ad_path": str(block.unit_h5ad_path),
                "spots_with_domain_h5ad_path": (
                    str(block.activity_h5ad_path) if paths.layer != "spot" else None
                ),
                "regsim_source_columns_raw": list(block.raw_column_names),
                "regsim_source_columns_canonical": list(block.feature_names),
                "regsim_feature_shape": list(block.values.shape),
                "regsim_knn_k": int(cfg.regsim_knn_k),
                "regsim_weight": float(cfg.regsim_weight),
                "regsim_adjacency_shape": list(regsim.shape),
                "regsim_adjacency_nnz": int(regsim.nnz),
                "regsim_adjacency_csr": regsim,
                "regsim_feature_csr": sp.csr_matrix(block.values),
                "regsim_feature_units": list(block.unit_ids),
                "regsim_feature_names": list(block.feature_names),
                "regsim_input_metadata": dict(block.metadata),
                "uses_grn": False,
                "uses_true_grn": False,
                "uses_cci": True,
            }
        )
        return graph

    def _stage_summary(
        self,
        stage: str,
        lower_graph: LayerGraph,
        upper_graph: LayerGraph,
    ) -> dict[str, object]:
        summary = super()._stage_summary(stage, lower_graph, upper_graph)
        summary.update(
            {
                "feature_source": "cci_plus_h5ad_regsim",
                "uses_true_grn": False,
                "lower_regsim_feature_shape": lower_graph.metadata.get("regsim_feature_shape"),
                "upper_regsim_feature_shape": upper_graph.metadata.get("regsim_feature_shape"),
                "lower_regsim_adjacency_nnz": lower_graph.metadata.get("regsim_adjacency_nnz"),
                "upper_regsim_adjacency_nnz": upper_graph.metadata.get("regsim_adjacency_nnz"),
            }
        )
        return summary
