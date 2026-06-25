from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.loaders import ExpressionData, LayerPaths, read_commot_index, read_commot_manifest, read_grn_edges


EDGE_COLUMNS = [
    "src_layer",
    "src_unit",
    "src_gene",
    "dst_layer",
    "dst_unit",
    "dst_gene",
    "edge_type",
    "commot_lr_key",
    "commot_ligand",
    "commot_receptor",
    "grn_weight_raw",
    "grn_weight_norm",
    "cci_score_raw",
    "cci_score_norm",
    "distance_raw",
    "influence_score",
]


@dataclass
class LayerGraph:
    layer: str
    time_point: str
    units: List[str]
    genes: List[str]
    intra_edges: pd.DataFrame
    inter_edges: pd.DataFrame
    shared_genes: List[str]


def _minmax_scale(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full(arr.shape, 1.0, dtype=float)
    return floor + (1.0 - floor) * (arr - vmin) / (vmax - vmin)


def _normalize_scalar(value: float, vmin: float, vmax: float, floor: float = 1e-6) -> float:
    if vmax <= vmin:
        return 1.0
    return float(floor + (1.0 - floor) * (value - vmin) / (vmax - vmin))


def _split_complex_genes(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in re.split(r"[_+|;/,]", text) if part]


def _restrict_and_normalize_grn(grn: pd.DataFrame, shared_genes: Sequence[str]) -> pd.DataFrame:
    shared = set(shared_genes)
    out = grn[grn["regulator"].isin(shared) & grn["target"].isin(shared)].copy()
    out["grn_weight_norm"] = _minmax_scale(out["weight"].to_numpy())
    return out


def _make_regulator_dict(grn: pd.DataFrame) -> Dict[str, List[Tuple[str, float, float]]]:
    regulator_to_targets: Dict[str, List[Tuple[str, float, float]]] = {}
    for reg, sub in grn.groupby("regulator"):
        regulator_to_targets[str(reg)] = list(
            zip(
                sub["target"].astype(str).tolist(),
                sub["weight"].astype(float).tolist(),
                sub["grn_weight_norm"].astype(float).tolist(),
            )
        )
    return regulator_to_targets


def _make_pair_lookup(grn: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for row in grn.itertuples(index=False):
        lookup[(str(row.regulator), str(row.target))] = (float(row.weight), float(row.grn_weight_norm))
    return lookup


def _resolve_inter_influence(
    cci_norm: float,
    pair_key: Tuple[str, str],
    pair_lookup: Dict[Tuple[str, str], Tuple[float, float]],
    mode: str,
    additive_cci_weight: float,
    additive_grn_weight: float,
    grn_pair_policy: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if mode not in {"product", "cci_only", "additive"}:
        raise ValueError(f"Unsupported inter_influence_mode {mode!r}.")
    if grn_pair_policy not in {"require_pair", "zero_if_missing"}:
        raise ValueError(f"Unsupported inter_grn_pair_policy {grn_pair_policy!r}.")
    if additive_cci_weight < 0 or additive_grn_weight < 0:
        raise ValueError("Inter additive weights must be nonnegative.")
    if mode == "additive" and additive_cci_weight + additive_grn_weight <= 0:
        raise ValueError("Inter additive weights must have a positive sum.")

    pair_weights = pair_lookup.get(pair_key)
    if mode == "cci_only":
        raw_w, norm_w = pair_weights if pair_weights is not None else (None, None)
        return raw_w, norm_w, float(cci_norm)

    if pair_weights is None:
        if mode == "product" or grn_pair_policy == "require_pair":
            return None, None, None
        raw_w, norm_w = None, 0.0
    else:
        raw_w, norm_w = pair_weights

    if mode == "product":
        influence_score = float(cci_norm) * float(norm_w)
    else:
        total_weight = additive_cci_weight + additive_grn_weight
        influence_score = (
            additive_cci_weight * float(cci_norm) + additive_grn_weight * float(norm_w)
        ) / total_weight
    return raw_w, norm_w, influence_score


def _scan_commot_score_range(manifest: pd.DataFrame, lr_dir: Path, cci_min: float) -> Tuple[Optional[float], Optional[float], int]:
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    kept_nnz = 0
    for row in manifest.itertuples(index=False):
        matrix_path = lr_dir / str(row.filename)
        mat = sp.load_npz(matrix_path)
        if mat.nnz == 0:
            continue
        data = np.asarray(mat.data, dtype=float)
        if cci_min > 0:
            data = data[data >= cci_min]
        if data.size == 0:
            continue
        kept_nnz += int(data.size)
        current_min = float(data.min())
        current_max = float(data.max())
        vmin = current_min if vmin is None else min(vmin, current_min)
        vmax = current_max if vmax is None else max(vmax, current_max)
    return vmin, vmax, kept_nnz


def _build_intra_edges(
    layer_name: str,
    expr: pd.DataFrame,
    active_mask: np.ndarray,
    regulator_to_targets: Dict[str, List[Tuple[str, float, float]]],
    use_expression_mask: bool = True,
) -> pd.DataFrame:
    gene_names = expr.columns.tolist()
    unit_names = expr.index.tolist()
    records: List[Tuple] = []
    for unit_idx, unit in enumerate(unit_names):
        if use_expression_mask:
            active_gene_idx = np.where(active_mask[unit_idx])[0]
            active_gene_set = {gene_names[gidx] for gidx in active_gene_idx}
        else:
            active_gene_set = set(gene_names)
        for src_gene in active_gene_set:
            for dst_gene, raw_w, norm_w in regulator_to_targets.get(src_gene, []):
                if dst_gene not in active_gene_set:
                    continue
                records.append(
                    (
                        layer_name,
                        unit,
                        src_gene,
                        layer_name,
                        unit,
                        dst_gene,
                        f"{layer_name}_intra",
                        np.nan,
                        np.nan,
                        np.nan,
                        raw_w,
                        norm_w,
                        np.nan,
                        np.nan,
                        np.nan,
                        norm_w,
                    )
                )
    return pd.DataFrame.from_records(records, columns=EDGE_COLUMNS)


def _build_commot_inter_edges(
    layer_name: str,
    manifest: pd.DataFrame,
    lr_dir: Path,
    index_names: Sequence[str],
    expr: pd.DataFrame,
    coords: pd.DataFrame,
    active_mask: np.ndarray,
    unit_index: Dict[str, int],
    gene_to_idx: Dict[str, int],
    pair_lookup: Dict[Tuple[str, str], Tuple[float, float]],
    score_range: Tuple[Optional[float], Optional[float], int],
    cci_min: float,
    require_target_expression: bool,
    inter_influence_mode: str,
    inter_additive_cci_weight: float,
    inter_additive_grn_weight: float,
    inter_grn_pair_policy: str,
    use_expression_mask: bool = True,
    require_coords: bool = True,
) -> pd.DataFrame:
    vmin, vmax, kept_nnz = score_range
    if kept_nnz == 0 or vmin is None or vmax is None:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    records: List[Tuple] = []
    for row in manifest.itertuples(index=False):
        ligand_genes = _split_complex_genes(row.ligand)
        receptor_genes = _split_complex_genes(row.receptor)
        if use_expression_mask:
            ligand_genes = [g for g in ligand_genes if g in gene_to_idx]
            receptor_genes = [g for g in receptor_genes if g in gene_to_idx]
        if not ligand_genes or not receptor_genes:
            continue

        matrix_path = lr_dir / str(row.filename)
        mat = sp.load_npz(matrix_path).tocoo()
        if mat.shape[0] != len(index_names) or mat.shape[1] != len(index_names):
            raise ValueError(f"COMMOT matrix shape {mat.shape} does not match index length {len(index_names)} for {matrix_path}")

        rows = np.asarray(mat.row)
        cols = np.asarray(mat.col)
        data = np.asarray(mat.data, dtype=float)
        if cci_min > 0:
            keep = data >= cci_min
            rows = rows[keep]
            cols = cols[keep]
            data = data[keep]
        if data.size == 0:
            continue

        for src_pos, dst_pos, cci_raw in zip(rows, cols, data):
            if src_pos == dst_pos:
                continue
            src_unit = index_names[int(src_pos)]
            dst_unit = index_names[int(dst_pos)]
            if src_unit not in unit_index or dst_unit not in unit_index:
                continue
            has_coords = src_unit in coords.index and dst_unit in coords.index
            if require_coords and not has_coords:
                continue

            src_idx = unit_index[src_unit]
            dst_idx = unit_index[dst_unit]
            cci_norm = _normalize_scalar(float(cci_raw), vmin=vmin, vmax=vmax)
            if has_coords:
                src_xy = coords.loc[src_unit, ["x", "y"]].to_numpy(dtype=float)
                dst_xy = coords.loc[dst_unit, ["x", "y"]].to_numpy(dtype=float)
                distance_raw = float(np.sqrt(((src_xy - dst_xy) ** 2).sum()))
            else:
                distance_raw = np.nan

            for ligand_gene in ligand_genes:
                ligand_idx = gene_to_idx.get(ligand_gene)
                if use_expression_mask and (ligand_idx is None or not active_mask[src_idx, ligand_idx]):
                    continue
                for receptor_gene in receptor_genes:
                    receptor_idx = gene_to_idx.get(receptor_gene)
                    if use_expression_mask and receptor_idx is None:
                        continue
                    if (
                        use_expression_mask
                        and require_target_expression
                        and not active_mask[dst_idx, receptor_idx]
                    ):
                        continue
                    raw_w, norm_w, influence_score = _resolve_inter_influence(
                        cci_norm=cci_norm,
                        pair_key=(ligand_gene, receptor_gene),
                        pair_lookup=pair_lookup,
                        mode=inter_influence_mode,
                        additive_cci_weight=inter_additive_cci_weight,
                        additive_grn_weight=inter_additive_grn_weight,
                        grn_pair_policy=inter_grn_pair_policy,
                    )
                    if influence_score is None:
                        continue
                    if influence_score <= 0:
                        continue
                    records.append(
                        (
                            layer_name,
                            src_unit,
                            ligand_gene,
                            layer_name,
                            dst_unit,
                            receptor_gene,
                            f"{layer_name}_inter",
                            str(row.lr_key),
                            str(row.ligand),
                            str(row.receptor),
                            raw_w,
                            norm_w,
                            float(cci_raw),
                            cci_norm,
                            distance_raw,
                            influence_score,
                        )
                    )
    return pd.DataFrame.from_records(records, columns=EDGE_COLUMNS)


def build_layer_graph(
    layer_name: str,
    time_point: str,
    expression: ExpressionData,
    paths: LayerPaths,
    shared_genes: Sequence[str],
    expr_threshold: float = 0.0,
    cci_min: float = 0.0,
    top_k_targets_per_regulator: int = 20,
    require_target_expression_for_inter: bool = True,
    inter_influence_mode: str = "product",
    inter_additive_cci_weight: float = 1.0,
    inter_additive_grn_weight: float = 1.0,
    inter_grn_pair_policy: str = "require_pair",
    include_intra_grn: bool = True,
    intra_use_expression_mask: bool = True,
    cci_inter_use_expression_mask: bool = True,
    cci_inter_require_coords: bool = True,
) -> LayerGraph:
    _resolve_inter_influence(
        cci_norm=1.0,
        pair_key=("", ""),
        pair_lookup={},
        mode=inter_influence_mode,
        additive_cci_weight=inter_additive_cci_weight,
        additive_grn_weight=inter_additive_grn_weight,
        grn_pair_policy=inter_grn_pair_policy,
    )
    grn = read_grn_edges(paths.grn_edges, top_k_targets_per_regulator=top_k_targets_per_regulator)
    grn = _restrict_and_normalize_grn(grn, shared_genes)
    expr = expression.expr.loc[:, list(shared_genes)].copy()
    active = expr.to_numpy() > expr_threshold

    regulator_to_targets = _make_regulator_dict(grn)
    pair_lookup = _make_pair_lookup(grn)
    manifest = read_commot_manifest(paths.cci_manifest)
    index_names = read_commot_index(paths.cci_index)
    score_range = _scan_commot_score_range(manifest, paths.cci_lr_dir, cci_min=cci_min)
    unit_index = {unit: idx for idx, unit in enumerate(expr.index.tolist())}
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.tolist())}

    intra = (
        _build_intra_edges(
            layer_name,
            expr,
            active,
            regulator_to_targets,
            use_expression_mask=intra_use_expression_mask,
        )
        if include_intra_grn
        else pd.DataFrame(columns=EDGE_COLUMNS)
    )
    inter = _build_commot_inter_edges(
        layer_name=layer_name,
        manifest=manifest,
        lr_dir=paths.cci_lr_dir,
        index_names=index_names,
        expr=expr,
        coords=expression.coords,
        active_mask=active,
        unit_index=unit_index,
        gene_to_idx=gene_to_idx,
        pair_lookup=pair_lookup,
        score_range=score_range,
        cci_min=cci_min,
        require_target_expression=require_target_expression_for_inter,
        inter_influence_mode=inter_influence_mode,
        inter_additive_cci_weight=inter_additive_cci_weight,
        inter_additive_grn_weight=inter_additive_grn_weight,
        inter_grn_pair_policy=inter_grn_pair_policy,
        use_expression_mask=cci_inter_use_expression_mask,
        require_coords=cci_inter_require_coords,
    )
    return LayerGraph(
        layer=layer_name,
        time_point=time_point,
        units=expr.index.astype(str).tolist(),
        genes=expr.columns.astype(str).tolist(),
        intra_edges=intra,
        inter_edges=inter,
        shared_genes=list(shared_genes),
    )
