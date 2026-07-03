from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.loaders import (
    ExpressionData,
    LayerPaths,
    read_commot_index,
    read_commot_manifest,
    read_grn_edges,
    read_unit_grn_edges,
)


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
    metadata: dict[str, object] = field(default_factory=dict)


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


def _restrict_and_normalize_unit_grn(grn: pd.DataFrame, shared_genes: Sequence[str]) -> pd.DataFrame:
    shared = set(shared_genes)
    out = grn[grn["regulator"].isin(shared) & grn["target"].isin(shared)].copy()
    if out.empty:
        out["grn_weight_norm"] = pd.Series(dtype=float)
        return out
    normalized_parts = []
    for _unit, sub in out.groupby("unit_id", sort=False):
        part = sub.copy()
        if "weight_norm" in part.columns and part["weight_norm"].notna().any():
            supplied = pd.to_numeric(part["weight_norm"], errors="coerce").to_numpy(dtype=float)
            computed = _minmax_scale(part["weight"].to_numpy(dtype=float))
            part["grn_weight_norm"] = np.clip(
                np.where(np.isfinite(supplied), supplied, computed),
                0.0,
                1.0,
            )
        else:
            part["grn_weight_norm"] = _minmax_scale(part["weight"].to_numpy(dtype=float))
        normalized_parts.append(part)
    return pd.concat(normalized_parts, ignore_index=True)


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


def _make_unit_regulator_dict(
    grn: pd.DataFrame,
) -> Dict[str, Dict[str, List[Tuple[str, float, float]]]]:
    result: Dict[str, Dict[str, List[Tuple[str, float, float]]]] = {}
    for unit, sub in grn.groupby("unit_id"):
        result[str(unit)] = _make_regulator_dict(sub)
    return result


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


def _manifest_chunks(manifest: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    size = max(1, int(chunk_size))
    return [
        manifest.iloc[start : start + size].copy()
        for start in range(0, len(manifest), size)
    ]


def _scan_commot_score_range(
    manifest: pd.DataFrame,
    lr_dir: Path,
    cci_min: float,
    workers: int = 1,
    chunk_size: int = 64,
) -> Tuple[Optional[float], Optional[float], int]:
    if workers > 1 and len(manifest) > 1:
        chunks = _manifest_chunks(manifest, chunk_size)
        actual_workers = max(1, min(int(workers), len(chunks)))
        results: list[Tuple[Optional[float], Optional[float], int] | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_index = {
                pool.submit(
                    _scan_commot_score_range,
                    chunk,
                    lr_dir,
                    cci_min,
                    1,
                    chunk_size,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
        vmins = [result[0] for result in results if result is not None and result[0] is not None]
        vmaxs = [result[1] for result in results if result is not None and result[1] is not None]
        kept_nnz = sum(result[2] for result in results if result is not None)
        return (
            min(vmins) if vmins else None,
            max(vmaxs) if vmaxs else None,
            int(kept_nnz),
        )

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


def _transform_expression_activity(expr: pd.DataFrame, transform: str) -> np.ndarray:
    if transform not in {"log1p_minmax", "log1p_zscore", "none"}:
        raise ValueError(
            "expression_transform must be one of ['log1p_minmax', 'log1p_zscore', 'none']."
        )
    values = np.clip(expr.to_numpy(dtype=float), 0.0, None)
    if transform == "none":
        return values
    logged = np.log1p(values)
    if transform == "log1p_minmax":
        mins = np.nanmin(logged, axis=0)
        maxs = np.nanmax(logged, axis=0)
        denom = maxs - mins
        scaled = np.divide(
            logged - mins,
            denom,
            out=np.zeros_like(logged),
            where=denom > 0,
        )
        constant_positive = (denom <= 0) & (maxs > 0)
        if np.any(constant_positive):
            scaled[:, constant_positive] = 1.0
        return scaled
    means = np.nanmean(logged, axis=0)
    stds = np.nanstd(logged, axis=0)
    scaled = np.divide(
        logged - means,
        stds,
        out=np.zeros_like(logged),
        where=stds > 0,
    )
    constant_positive = (stds <= 0) & (means > 0)
    if np.any(constant_positive):
        scaled[:, constant_positive] = 1.0
    return np.clip(scaled, 0.0, None)


def _expression_activity(src_value: float, dst_value: float, mode: str, floor: float) -> float:
    if mode not in {"none", "geometric_mean", "product", "min"}:
        raise ValueError(
            "expression_weight_mode must be one of ['none', 'geometric_mean', 'product', 'min']."
        )
    if floor < 0:
        raise ValueError("expression_weight_floor must be nonnegative.")
    if mode == "none":
        return 1.0
    if mode == "geometric_mean":
        activity = float(np.sqrt(max(0.0, src_value) * max(0.0, dst_value)))
    elif mode == "product":
        activity = float(max(0.0, src_value) * max(0.0, dst_value))
    else:
        activity = float(min(max(0.0, src_value), max(0.0, dst_value)))
    return max(float(floor), activity)


def _build_intra_edges(
    layer_name: str,
    expr: pd.DataFrame,
    active_mask: np.ndarray,
    regulator_to_targets: Dict[str, List[Tuple[str, float, float]]],
    use_expression_mask: bool = True,
    expression_weight_mode: str = "none",
    expression_transform: str = "log1p_minmax",
    expression_weight_floor: float = 0.0,
    unit_regulator_to_targets: Dict[str, Dict[str, List[Tuple[str, float, float]]]] | None = None,
    unit_specific_fallback: str = "sample_grn_expression_weighted",
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, dict[str, object]]:
    if unit_specific_fallback not in {
        "error",
        "sample_grn_masked",
        "sample_grn_expression_weighted",
        "skip_unit_intra",
    }:
        raise ValueError(
            "unit_specific_fallback must be one of "
            "['error', 'sample_grn_masked', 'sample_grn_expression_weighted', 'skip_unit_intra']."
        )
    gene_names = expr.columns.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    unit_names = expr.index.tolist()
    activity_values = _transform_expression_activity(expr, expression_transform)
    records: List[Tuple] = []
    fallback_units: List[str] = []
    for unit_idx, unit in enumerate(unit_names):
        unit_key = str(unit)
        unit_targets = None if unit_regulator_to_targets is None else unit_regulator_to_targets.get(unit_key)
        using_unit_specific = unit_targets is not None
        if unit_regulator_to_targets is not None and unit_targets is None:
            fallback_units.append(unit_key)
            if unit_specific_fallback == "error":
                continue
            if unit_specific_fallback == "skip_unit_intra":
                continue
            unit_targets = regulator_to_targets
        if unit_targets is None:
            unit_targets = regulator_to_targets

        if using_unit_specific:
            effective_mask = False
            effective_weight_mode = "none"
        elif unit_regulator_to_targets is not None and unit_specific_fallback == "sample_grn_masked":
            effective_mask = True
            effective_weight_mode = "none"
        else:
            effective_mask = use_expression_mask
            effective_weight_mode = expression_weight_mode

        if effective_mask:
            active_gene_idx = np.where(active_mask[unit_idx])[0]
            active_gene_set = {gene_names[gidx] for gidx in active_gene_idx}
        else:
            active_gene_set = set(gene_names)
        for src_gene in active_gene_set:
            for dst_gene, raw_w, norm_w in unit_targets.get(src_gene, []):
                if dst_gene not in active_gene_set:
                    continue
                src_idx = gene_to_idx[src_gene]
                dst_idx = gene_to_idx[dst_gene]
                activity = _expression_activity(
                    activity_values[unit_idx, src_idx],
                    activity_values[unit_idx, dst_idx],
                    effective_weight_mode,
                    expression_weight_floor,
                )
                influence_score = float(norm_w) * activity
                if influence_score <= 0:
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
                        influence_score,
                    )
                )
    if unit_regulator_to_targets is not None and unit_specific_fallback == "error" and fallback_units:
        raise ValueError(
            "Unit-specific GRN is missing units: "
            + ", ".join(fallback_units[:20])
            + (f" ... ({len(fallback_units)} total)" if len(fallback_units) > 20 else "")
        )
    edge_table = pd.DataFrame.from_records(records, columns=EDGE_COLUMNS)
    metadata = {
        "unit_specific_units": int(
            0 if unit_regulator_to_targets is None else len(set(unit_names) & set(unit_regulator_to_targets))
        ),
        "unit_specific_fallback_units": fallback_units,
        "expression_weight_mode": expression_weight_mode,
        "expression_transform": expression_transform,
        "expression_weight_floor": float(expression_weight_floor),
    }
    return (edge_table, metadata) if return_metadata else edge_table


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
    workers: int = 1,
    chunk_size: int = 64,
) -> pd.DataFrame:
    if workers > 1 and len(manifest) > 1:
        chunks = _manifest_chunks(manifest, chunk_size)
        actual_workers = max(1, min(int(workers), len(chunks)))
        parts: list[pd.DataFrame | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_index = {
                pool.submit(
                    _build_commot_inter_edges,
                    layer_name=layer_name,
                    manifest=chunk,
                    lr_dir=lr_dir,
                    index_names=index_names,
                    expr=expr,
                    coords=coords,
                    active_mask=active_mask,
                    unit_index=unit_index,
                    gene_to_idx=gene_to_idx,
                    pair_lookup=pair_lookup,
                    score_range=score_range,
                    cci_min=cci_min,
                    require_target_expression=require_target_expression,
                    inter_influence_mode=inter_influence_mode,
                    inter_additive_cci_weight=inter_additive_cci_weight,
                    inter_additive_grn_weight=inter_additive_grn_weight,
                    inter_grn_pair_policy=inter_grn_pair_policy,
                    use_expression_mask=use_expression_mask,
                    require_coords=require_coords,
                    workers=1,
                    chunk_size=chunk_size,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                parts[future_to_index[future]] = future.result()
        non_empty = [part for part in parts if part is not None and not part.empty]
        if not non_empty:
            return pd.DataFrame(columns=EDGE_COLUMNS)
        return pd.concat(non_empty, ignore_index=True)

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


def build_layer_cci_graph(
    layer_name: str,
    time_point: str,
    expression: ExpressionData,
    paths: LayerPaths,
    shared_genes: Sequence[str],
    expr_threshold: float = 0.0,
    cci_min: float = 0.0,
    require_target_expression_for_inter: bool = True,
    cci_inter_use_expression_mask: bool = True,
    cci_inter_require_coords: bool = False,
    cci_workers: int = 1,
    cci_chunk_size: int = 64,
) -> LayerGraph:
    expr = expression.expr.loc[:, list(shared_genes)].copy()
    active = expr.to_numpy() > expr_threshold
    manifest = read_commot_manifest(paths.cci_manifest)
    index_names = read_commot_index(paths.cci_index)
    score_range = _scan_commot_score_range(
        manifest,
        paths.cci_lr_dir,
        cci_min=cci_min,
        workers=cci_workers,
        chunk_size=cci_chunk_size,
    )
    unit_index = {unit: idx for idx, unit in enumerate(expr.index.tolist())}
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.tolist())}
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
        pair_lookup={},
        score_range=score_range,
        cci_min=cci_min,
        require_target_expression=require_target_expression_for_inter,
        inter_influence_mode="cci_only",
        inter_additive_cci_weight=1.0,
        inter_additive_grn_weight=0.0,
        inter_grn_pair_policy="zero_if_missing",
        use_expression_mask=cci_inter_use_expression_mask,
        require_coords=cci_inter_require_coords,
        workers=cci_workers,
        chunk_size=cci_chunk_size,
    )
    return LayerGraph(
        layer=layer_name,
        time_point=time_point,
        units=expr.index.astype(str).tolist(),
        genes=expr.columns.astype(str).tolist(),
        intra_edges=pd.DataFrame(columns=EDGE_COLUMNS),
        inter_edges=inter,
        shared_genes=list(shared_genes),
        metadata={
            "grn_source": "not_used",
            "intra_source": "expression_matrix",
            "inter_source": "cci_only",
            "inter_influence_mode": "cci_only",
            "inter_additive_cci_weight": 1.0,
            "inter_additive_grn_weight": 0.0,
            "inter_grn_pair_policy": "zero_if_missing",
        },
    )


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
    grn_source: str = "sample",
    expression_weight_mode: str = "none",
    expression_transform: str = "log1p_minmax",
    expression_weight_floor: float = 0.0,
    unit_specific_fallback: str = "sample_grn_expression_weighted",
    cci_workers: int = 1,
    cci_chunk_size: int = 64,
) -> LayerGraph:
    if grn_source not in {"sample", "sample_expression_weighted", "unit_specific"}:
        raise ValueError(
            "grn_source must be one of ['sample', 'sample_expression_weighted', 'unit_specific']."
        )
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
    unit_grn = None
    unit_grn_path = paths.unit_grn_edges
    if grn_source == "unit_specific" and unit_grn_path is not None and unit_grn_path.exists():
        unit_grn = read_unit_grn_edges(
            unit_grn_path,
            top_k_targets_per_regulator=top_k_targets_per_regulator,
        )
        unit_grn = _restrict_and_normalize_unit_grn(unit_grn, shared_genes)
    expr = expression.expr.loc[:, list(shared_genes)].copy()
    active = expr.to_numpy() > expr_threshold

    regulator_to_targets = _make_regulator_dict(grn)
    unit_regulator_to_targets = None if unit_grn is None else _make_unit_regulator_dict(unit_grn)
    pair_lookup = _make_pair_lookup(grn)
    manifest = read_commot_manifest(paths.cci_manifest)
    index_names = read_commot_index(paths.cci_index)
    score_range = _scan_commot_score_range(
        manifest,
        paths.cci_lr_dir,
        cci_min=cci_min,
        workers=cci_workers,
        chunk_size=cci_chunk_size,
    )
    unit_index = {unit: idx for idx, unit in enumerate(expr.index.tolist())}
    gene_to_idx = {gene: idx for idx, gene in enumerate(expr.columns.tolist())}

    effective_expression_weight_mode = (
        expression_weight_mode if grn_source in {"sample_expression_weighted", "unit_specific"} else "none"
    )
    if include_intra_grn:
        intra, intra_metadata = _build_intra_edges(
            layer_name,
            expr,
            active,
            regulator_to_targets,
            use_expression_mask=intra_use_expression_mask,
            expression_weight_mode=effective_expression_weight_mode,
            expression_transform=expression_transform,
            expression_weight_floor=expression_weight_floor,
            unit_regulator_to_targets=(
                (unit_regulator_to_targets or {})
                if grn_source == "unit_specific"
                else None
            ),
            unit_specific_fallback=unit_specific_fallback,
            return_metadata=True,
        )
    else:
        intra = pd.DataFrame(columns=EDGE_COLUMNS)
        intra_metadata = {}
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
        workers=cci_workers,
        chunk_size=cci_chunk_size,
    )
    return LayerGraph(
        layer=layer_name,
        time_point=time_point,
        units=expr.index.astype(str).tolist(),
        genes=expr.columns.astype(str).tolist(),
        intra_edges=intra,
        inter_edges=inter,
        shared_genes=list(shared_genes),
        metadata={
            "grn_source": grn_source,
            "unit_grn_path": str(unit_grn_path) if unit_grn_path is not None else None,
            "unit_grn_file_found": bool(unit_grn_path is not None and unit_grn_path.exists()),
            "unit_specific_fallback": unit_specific_fallback,
            **intra_metadata,
        },
    )
