from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from mignet_ce.graph.builder import LayerGraph


IntraKey = Tuple[str, str]
InterKey = Tuple[str, str, str]


def _clean_edge_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _inter_key_from_row(row: object) -> InterKey:
    lr_key = _clean_edge_value(getattr(row, "commot_lr_key", ""))
    ligand = _clean_edge_value(getattr(row, "commot_ligand", ""))
    receptor = _clean_edge_value(getattr(row, "commot_receptor", ""))
    if not ligand:
        ligand = _clean_edge_value(getattr(row, "src_gene", ""))
    if not receptor:
        receptor = _clean_edge_value(getattr(row, "dst_gene", ""))
    if not lr_key:
        lr_key = f"{ligand}->{receptor}"
    return lr_key, ligand, receptor


@dataclass(frozen=True)
class NativeFeatureSchema:
    intra_pairs: List[IntraKey]
    inter_pairs: List[InterKey]
    feature_names: List[str]
    feature_blocks: Dict[str, List[str]]


def build_native_feature_schema(graphs: Iterable[LayerGraph]) -> NativeFeatureSchema:
    intra_pairs: set[IntraKey] = set()
    inter_pairs: set[InterKey] = set()
    for graph in graphs:
        if not graph.intra_edges.empty:
            for row in graph.intra_edges.itertuples(index=False):
                intra_pairs.add((str(row.src_gene), str(row.dst_gene)))
        if not graph.inter_edges.empty:
            for row in graph.inter_edges.itertuples(index=False):
                inter_pairs.add(_inter_key_from_row(row))

    ordered_intra = sorted(intra_pairs)
    ordered_inter = sorted(inter_pairs)
    intra_names = [f"intra_grn:{src}->{dst}" for src, dst in ordered_intra]
    inter_names = [
        f"inter_cci:{ligand}->{receptor}[{lr_key}]"
        for lr_key, ligand, receptor in ordered_inter
    ]
    return NativeFeatureSchema(
        intra_pairs=ordered_intra,
        inter_pairs=ordered_inter,
        feature_names=[*intra_names, *inter_names],
        feature_blocks={
            "intra_grn": intra_names,
            "inter_cci": inter_names,
        },
    )


def build_native_graph_matrix(
    graph: LayerGraph,
    schema: NativeFeatureSchema,
    feature_log1p: bool = True,
) -> np.ndarray:
    units = list(map(str, graph.units))
    unit_to_row = {unit: idx for idx, unit in enumerate(units)}
    intra_to_col = {pair: idx for idx, pair in enumerate(schema.intra_pairs)}
    inter_offset = len(schema.intra_pairs)
    inter_to_col = {
        pair: inter_offset + idx
        for idx, pair in enumerate(schema.inter_pairs)
    }
    matrix = np.zeros((len(units), len(schema.feature_names)), dtype=float)

    if not graph.intra_edges.empty:
        grouped = (
            graph.intra_edges
            .groupby(["src_unit", "src_gene", "dst_gene"], dropna=False)["influence_score"]
            .sum()
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            unit_row = unit_to_row.get(str(row.src_unit))
            feature_col = intra_to_col.get((str(row.src_gene), str(row.dst_gene)))
            if unit_row is not None and feature_col is not None:
                matrix[unit_row, feature_col] += float(row.influence_score)

    if not graph.inter_edges.empty:
        outgoing: Dict[Tuple[str, str, str, str, str], float] = {}
        for row in graph.inter_edges.itertuples(index=False):
            src_unit = str(row.src_unit)
            dst_unit = str(row.dst_unit)
            if src_unit == dst_unit or src_unit not in unit_to_row:
                continue
            lr_key, ligand, receptor = _inter_key_from_row(row)
            edge_key = (src_unit, dst_unit, lr_key, ligand, receptor)
            score = float(row.influence_score)
            previous = outgoing.get(edge_key)
            outgoing[edge_key] = score if previous is None else max(previous, score)

        for (src_unit, _dst_unit, lr_key, ligand, receptor), score in outgoing.items():
            feature_col = inter_to_col.get((lr_key, ligand, receptor))
            if feature_col is not None:
                matrix[unit_to_row[src_unit], feature_col] += score

    return np.log1p(matrix) if feature_log1p else matrix


def build_native_feature_block_summary(
    matrix: np.ndarray,
    units: Iterable[str],
    feature_names: List[str],
    feature_blocks: Dict[str, List[str]],
    *,
    stage: str,
    layer_role: str,
) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    unit_list = list(map(str, units))
    if values.ndim != 2 or values.shape[0] != len(unit_list):
        raise ValueError(
            f"Feature matrix shape {values.shape} does not match {len(unit_list)} units."
        )
    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature matrix has {values.shape[1]} columns but schema has {len(feature_names)} names."
        )
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    intra_indices = [
        name_to_index[name]
        for name in feature_blocks.get("intra_grn", [])
        if name in name_to_index
    ]
    inter_indices = [
        name_to_index[name]
        for name in feature_blocks.get("inter_cci", [])
        if name in name_to_index
    ]
    intra = values[:, intra_indices] if intra_indices else np.zeros((len(unit_list), 0))
    inter = values[:, inter_indices] if inter_indices else np.zeros((len(unit_list), 0))
    return pd.DataFrame(
        {
            "stage": str(stage),
            "layer_role": str(layer_role),
            "unit_id": unit_list,
            "intra_sum": intra.sum(axis=1),
            "inter_sum": inter.sum(axis=1),
            "intra_nonzero": np.count_nonzero(intra, axis=1),
            "inter_nonzero": np.count_nonzero(inter, axis=1),
            "feature_norm": np.linalg.norm(values, axis=1),
        }
    )
