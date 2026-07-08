from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import EDGE_COLUMNS, LayerGraph
from mignet_ce.io.loaders import (
    LayerDataResolver,
    LayerPaths,
    natural_sort,
    peek_h5ad_genes,
    read_commot_index,
    read_commot_manifest,
    read_expression_h5ad,
    read_grn_edges,
)
from mignet_ce.mapping import OverlapMapping
from mignet_ce.networks.base import NetworkContext
from mignet_ce.utils.coords import align_coords


def _empty_edge_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


def _empty_overlap(lower_units: Sequence[str], upper_units: Sequence[str]) -> OverlapMapping:
    lower = list(map(str, lower_units))
    upper = list(map(str, upper_units))
    counts = np.zeros((len(lower), len(upper)), dtype=float)
    return OverlapMapping(lower_units=lower, upper_units=upper, counts=counts, weights=counts.copy())


def _as_nonnegative_csr(matrix: sp.spmatrix, cci_min: float = 0.0) -> sp.csr_matrix:
    out = matrix.tocsr(copy=True).astype(float)
    if out.nnz:
        out.data = np.nan_to_num(out.data, nan=0.0, posinf=0.0, neginf=0.0)
        out.data[out.data < 0.0] = 0.0
        if cci_min > 0:
            out.data[out.data < float(cci_min)] = 0.0
        out.eliminate_zeros()
    return out


def _align_square_matrix(matrix: sp.spmatrix, index_names: Sequence[str], units: Sequence[str]) -> sp.csr_matrix:
    units = list(map(str, units))
    lookup = {unit: idx for idx, unit in enumerate(map(str, index_names))}
    target_rows: List[int] = []
    source_rows: List[int] = []
    for out_idx, unit in enumerate(units):
        src_idx = lookup.get(unit)
        if src_idx is not None:
            target_rows.append(out_idx)
            source_rows.append(src_idx)
    if not target_rows:
        return sp.csr_matrix((len(units), len(units)), dtype=float)
    sub = matrix.tocsr()[source_rows, :][:, source_rows].tocoo()
    rows = np.asarray([target_rows[int(row)] for row in sub.row], dtype=int)
    cols = np.asarray([target_rows[int(col)] for col in sub.col], dtype=int)
    return sp.coo_matrix((sub.data, (rows, cols)), shape=(len(units), len(units)), dtype=float).tocsr()


def _normalize_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.ones_like(arr, dtype=float)
    return (arr - vmin) / (vmax - vmin)


def _cci_edges_from_adjacency(matrix: sp.spmatrix, units: Sequence[str], layer: str, stage: str) -> pd.DataFrame:
    units = list(map(str, units))
    coo = matrix.tocoo()
    if coo.nnz == 0:
        return _empty_edge_frame()
    raw = np.asarray(coo.data, dtype=float)
    norm = _normalize_values(raw)
    return pd.DataFrame(
        {
            "src_layer": layer,
            "src_unit": [units[int(row)] for row in coo.row],
            "src_gene": np.nan,
            "dst_layer": layer,
            "dst_unit": [units[int(col)] for col in coo.col],
            "dst_gene": np.nan,
            "edge_type": "cci",
            "commot_lr_key": np.nan,
            "commot_ligand": np.nan,
            "commot_receptor": np.nan,
            "grn_weight_raw": np.nan,
            "grn_weight_norm": np.nan,
            "cci_score_raw": raw,
            "cci_score_norm": norm,
            "distance_raw": np.nan,
            "influence_score": raw,
        },
        columns=EDGE_COLUMNS,
    )


def _grn_edges_to_frame(grn: pd.DataFrame, layer: str) -> pd.DataFrame:
    if grn.empty:
        return _empty_edge_frame()
    work = grn.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work = work.dropna(subset=["weight"])
    if work.empty:
        return _empty_edge_frame()
    raw = work["weight"].to_numpy(dtype=float)
    strength = np.abs(raw)
    norm = _normalize_values(strength)
    return pd.DataFrame(
        {
            "src_layer": layer,
            "src_unit": work["regulator"].astype(str).to_numpy(),
            "src_gene": work["regulator"].astype(str).to_numpy(),
            "dst_layer": layer,
            "dst_unit": work["target"].astype(str).to_numpy(),
            "dst_gene": work["target"].astype(str).to_numpy(),
            "edge_type": "grn",
            "commot_lr_key": np.nan,
            "commot_ligand": np.nan,
            "commot_receptor": np.nan,
            "grn_weight_raw": raw,
            "grn_weight_norm": norm,
            "cci_score_raw": np.nan,
            "cci_score_norm": np.nan,
            "distance_raw": np.nan,
            "influence_score": strength,
        },
        columns=EDGE_COLUMNS,
    )


def _adjacency_from_grn_edges(edge_table: pd.DataFrame, units: Sequence[str]) -> sp.csr_matrix:
    units = list(map(str, units))
    index = {unit: idx for idx, unit in enumerate(units)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in edge_table.itertuples(index=False):
        src = str(row.src_unit)
        dst = str(row.dst_unit)
        if src not in index or dst not in index:
            continue
        value = float(row.influence_score)
        if value <= 0:
            continue
        rows.append(index[src])
        cols.append(index[dst])
        data.append(value)
    return sp.coo_matrix((data, (rows, cols)), shape=(len(units), len(units)), dtype=float).tocsr()


class LightCCINetworkBuilder:
    network_method = "light_cci"

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        lower_graphs: List[LayerGraph] = []
        upper_graphs: List[LayerGraph] = []
        lower_units_by_time: List[List[str]] = []
        upper_units_by_time: List[List[str]] = []
        lower_coords_by_time: List[np.ndarray] = []
        upper_coords_by_time: List[np.ndarray] = []
        lower_mats: List[np.ndarray] = []
        upper_mats: List[np.ndarray] = []
        overlaps: List[OverlapMapping] = []
        graph_summaries: List[dict[str, object]] = []
        shared_gene_sets: List[set[str]] = []
        stage_metadata: List[dict[str, object]] = []

        for stage in map(str, cfg.time_points):
            lower_graph, lower_coords, lower_shared = self._build_layer_graph(
                resolver=resolver,
                layer=pair.lower_layer,
                organ=organ,
                stage=stage,
                cfg=cfg,
            )
            upper_graph, upper_coords, upper_shared = self._build_layer_graph(
                resolver=resolver,
                layer=pair.upper_layer,
                organ=organ,
                stage=stage,
                cfg=cfg,
            )
            lower_units = list(map(str, lower_graph.units))
            upper_units = list(map(str, upper_graph.units))
            lower_graphs.append(lower_graph)
            upper_graphs.append(upper_graph)
            lower_units_by_time.append(lower_units)
            upper_units_by_time.append(upper_units)
            lower_coords_by_time.append(lower_coords)
            upper_coords_by_time.append(upper_coords)
            lower_mats.append(np.zeros((len(lower_units), 0), dtype=float))
            upper_mats.append(np.zeros((len(upper_units), 0), dtype=float))
            overlaps.append(_empty_overlap(lower_units, upper_units))
            shared_gene_sets.append(set(lower_shared) & set(upper_shared))
            graph_summaries.append(self._stage_summary(stage, lower_graph, upper_graph))
            stage_metadata.append(
                {
                    "stage": stage,
                    "lower_layer": pair.lower_layer,
                    "upper_layer": pair.upper_layer,
                    "lower_edge_source": lower_graph.metadata.get("edge_source"),
                    "upper_edge_source": upper_graph.metadata.get("edge_source"),
                    "lower_units": len(lower_units),
                    "upper_units": len(upper_units),
                    "lower_edges": int(lower_graph.metadata.get("adjacency_nnz", 0)),
                    "upper_edges": int(upper_graph.metadata.get("adjacency_nnz", 0)),
                }
            )

        shared_genes = natural_sort(set.intersection(*shared_gene_sets)) if shared_gene_sets else []
        stable_upper_units = natural_sort({unit for units in upper_units_by_time for unit in units})
        return NetworkContext(
            organ=organ,
            pair=pair,
            time_points=list(map(str, cfg.time_points)),
            network_method=self.network_method,
            stable_upper_units=stable_upper_units,
            shared_genes=shared_genes,
            lower_mats=lower_mats,
            upper_mats=upper_mats,
            overlaps=overlaps,
            lower_units_by_time=lower_units_by_time,
            upper_units_by_time=upper_units_by_time,
            upper_coords_by_time=upper_coords_by_time,
            feature_names=[],
            feature_blocks={"light_cci_graph_only": []},
            graph_summaries=graph_summaries,
            lower_coords_by_time=lower_coords_by_time,
            feature_alignment_space="native_units",
            exports={},
            metadata={
                "network_method": self.network_method,
                "feature_source": "light_cci_graph_only",
                "feature_alignment_space": "native_units",
                "uses_expression_feature": False,
                "uses_grn": True,
                "uses_cci": True,
                "uses_legacy_graph": False,
                "stages": stage_metadata,
            },
            lower_graphs=lower_graphs,
            upper_graphs=upper_graphs,
            coverage_tables=[],
            spot_correspondence_tables=[],
            overlap_edge_tables=[],
            overlap_quality_summaries=[],
        )

    def _build_layer_graph(
        self,
        *,
        resolver: LayerDataResolver,
        layer: str,
        organ: str,
        stage: str,
        cfg: TemporalRunConfig,
    ) -> tuple[LayerGraph, np.ndarray, List[str]]:
        paths = resolver.paths(layer, organ, stage)
        if layer == "gene":
            return self._build_gene_graph(paths=paths, layer=layer, stage=stage)
        return self._build_cci_graph(paths=paths, layer=layer, stage=stage, cfg=cfg)

    def _build_cci_graph(
        self,
        *,
        paths: LayerPaths,
        layer: str,
        stage: str,
        cfg: TemporalRunConfig,
    ) -> tuple[LayerGraph, np.ndarray, List[str]]:
        if not paths.h5ad.exists():
            raise FileNotFoundError(f"LightCCI {layer} layer requires h5ad for nodes and coordinates: {paths.h5ad}")
        if not paths.cci_index.exists():
            raise FileNotFoundError(f"LightCCI {layer} layer requires COMMOT/CCI index: {paths.cci_index}")
        expression = read_expression_h5ad(paths.h5ad)
        index_units = read_commot_index(paths.cci_index)
        units = list(map(str, index_units))
        if paths.cci_total.exists():
            raw = sp.load_npz(paths.cci_total)
            if raw.shape != (len(units), len(units)):
                raise ValueError(f"CCI total shape {raw.shape} does not match index length {len(units)} for {paths.cci_total}.")
            source = "cci_total"
            source_path: Path = paths.cci_total
            lr_files = 0
        else:
            if not paths.cci_manifest.exists() or not paths.cci_lr_dir.exists():
                raise FileNotFoundError(
                    f"LightCCI {layer} layer needs {paths.cci_total} or LR fallback files "
                    f"{paths.cci_manifest} and {paths.cci_lr_dir}."
                )
            manifest = read_commot_manifest(paths.cci_manifest)
            total: sp.csr_matrix | None = None
            lr_files = 0
            for row in manifest.itertuples(index=False):
                lr_path = paths.cci_lr_dir / str(row.filename)
                lr_matrix = sp.load_npz(lr_path)
                if lr_matrix.shape != (len(units), len(units)):
                    raise ValueError(f"COMMOT LR matrix shape {lr_matrix.shape} does not match index length {len(units)} for {lr_path}.")
                total = lr_matrix.tocsr() if total is None else total + lr_matrix.tocsr()
                lr_files += 1
            raw = total if total is not None else sp.csr_matrix((len(units), len(units)), dtype=float)
            source = "commot_lr_aggregate"
            source_path = paths.cci_lr_dir
        adjacency = _as_nonnegative_csr(_align_square_matrix(raw, index_units, units), cci_min=cfg.cci_min)
        inter_edges = _cci_edges_from_adjacency(adjacency, units, layer, stage)
        coords = align_coords(expression.coords, units)
        graph = LayerGraph(
            layer=layer,
            time_point=stage,
            units=units,
            genes=expression.genes,
            intra_edges=_empty_edge_frame(),
            inter_edges=inter_edges,
            shared_genes=expression.genes,
            metadata={
                "network_method": self.network_method,
                "edge_source": "cci",
                "adjacency_source": source,
                "adjacency_path": str(source_path),
                "index_path": str(paths.cci_index),
                "lr_files": int(lr_files),
                "adjacency_shape": list(adjacency.shape),
                "adjacency_nnz": int(adjacency.nnz),
                "adjacency_csr": adjacency,
                "uses_grn": False,
                "uses_cci": True,
                "layer_semantics": "cell_or_domain_cci",
                "has_spatial": True,
            },
        )
        return graph, coords, expression.genes

    def _build_gene_graph(self, *, paths: LayerPaths, layer: str, stage: str) -> tuple[LayerGraph, np.ndarray, List[str]]:
        if not paths.grn_edges.exists():
            raise FileNotFoundError(
                "LightCCI gene layer requires a GRN edge file with regulator,target,weight columns; "
                f"tried {paths.grn_edges}."
            )
        grn = read_grn_edges(paths.grn_edges, top_k_targets_per_regulator=None)
        edge_table = _grn_edges_to_frame(grn, layer=layer)
        units = natural_sort(set(edge_table["src_unit"].astype(str)) | set(edge_table["dst_unit"].astype(str))) if not edge_table.empty else []
        if not units:
            raise ValueError(f"LightCCI gene layer GRN has no usable positive-strength edges: {paths.grn_edges}")
        adjacency = _adjacency_from_grn_edges(edge_table, units)
        if adjacency.nnz == 0:
            raise ValueError(f"LightCCI gene layer GRN has no usable positive-strength edges: {paths.grn_edges}")
        coords = np.zeros((len(units), 2), dtype=float)
        graph = LayerGraph(
            layer=layer,
            time_point=stage,
            units=units,
            genes=units,
            intra_edges=edge_table,
            inter_edges=_empty_edge_frame(),
            shared_genes=units,
            metadata={
                "network_method": self.network_method,
                "edge_source": "grn",
                "adjacency_source": "grn_edges",
                "adjacency_path": str(paths.grn_edges),
                "adjacency_shape": list(adjacency.shape),
                "adjacency_nnz": int(adjacency.nnz),
                "adjacency_csr": adjacency,
                "grn_weight_mode": "abs",
                "uses_grn": True,
                "uses_cci": False,
                "layer_semantics": "gene_grn",
                "has_spatial": False,
            },
        )
        return graph, coords, units

    @staticmethod
    def _stage_summary(stage: str, lower_graph: LayerGraph, upper_graph: LayerGraph) -> dict[str, object]:
        return {
            "time_point": stage,
            "network_method": LightCCINetworkBuilder.network_method,
            "feature_source": "light_cci_graph_only",
            "feature_alignment_space": "native_units",
            "lower_layer": lower_graph.layer,
            "upper_layer": upper_graph.layer,
            "lower_edge_source": lower_graph.metadata.get("edge_source"),
            "upper_edge_source": upper_graph.metadata.get("edge_source"),
            "lower_units": len(lower_graph.units),
            "upper_units": len(upper_graph.units),
            "lower_edges": int(lower_graph.metadata.get("adjacency_nnz", 0)),
            "upper_edges": int(upper_graph.metadata.get("adjacency_nnz", 0)),
            "lower_has_spatial": bool(lower_graph.metadata.get("has_spatial", False)),
            "upper_has_spatial": bool(upper_graph.metadata.get("has_spatial", False)),
            "lower_adjacency_shape": lower_graph.metadata.get("adjacency_shape"),
            "upper_adjacency_shape": upper_graph.metadata.get("adjacency_shape"),
        }
