#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the already audited exact V7 and WYT helpers.
from scripts.run_native_v7_wyt_study import (
    StageData,
    grn_project,
    hard_assignment,
    native_v7_pij_numpy,
    n_pair,
    pairwise_zscore,
    pool,
)
from mignet_ce.io.loaders import read_grn_edges
from mignet_ce.metrics import effective_information
from mignet_ce.networks.light_cci_grn import (
    deterministic_projection_matrix,
    prepare_grn_inputs,
)
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
    row_normalize_sparse,
)
from mignet_ce.pij.compare.native_v7_torch import native_v7_pij_torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from mignet_ce.representations.wyt_network80 import joint_fixed_pca
from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei

EPS = 1e-12


def _decode(values) -> list[str]:
    out: list[str] = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def _df_index(group: h5py.Group) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return _decode(np.asarray(group[str(key)]))


def _read_csr(group: h5py.Group) -> sp.csr_matrix:
    shape = tuple(int(v) for v in group.attrs["shape"])
    encoding = group.attrs.get("encoding-type", "csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    payload = (
        np.asarray(group["data"]),
        np.asarray(group["indices"], dtype=np.int32),
        np.asarray(group["indptr"], dtype=np.int32),
    )
    if str(encoding) == "csc_matrix":
        return sp.csc_matrix(payload, shape=shape).tocsr()
    return sp.csr_matrix(payload, shape=shape)


def read_h5ad_counts(path: Path) -> tuple[sp.csr_matrix, list[str], list[str], np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        units = _df_index(handle["obs"])
        genes = _df_index(handle["var"])
        spatial = handle["obsm/spatial"]
        if isinstance(spatial, h5py.Group):
            # AnnData may encode obsm arrays as an HDF5 dataframe group.
            raw_order = spatial.attrs.get("column-order", None)
            if raw_order is None:
                columns = [key for key in spatial.keys() if key != "_index"]
            else:
                columns = [
                    item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                    for item in raw_order
                ]
            if len(columns) < 2:
                columns = [key for key in ("x", "y") if key in spatial]
            coords = np.column_stack([np.asarray(spatial[key], dtype=np.float32) for key in columns])
        else:
            coords = np.asarray(spatial, dtype=np.float32)
        if "layers" in handle and "count" in handle["layers"]:
            node = handle["layers/count"]
            source = "layers/count"
        elif "layers" in handle and "counts" in handle["layers"]:
            node = handle["layers/counts"]
            source = "layers/counts"
        else:
            node = handle["X"]
            source = "X"
        if isinstance(node, h5py.Group):
            matrix = _read_csr(node)
        else:
            matrix = sp.csr_matrix(np.asarray(node))
    matrix = matrix.astype(np.float32)
    if matrix.shape != (len(units), len(genes)):
        raise ValueError(f"H5AD matrix {matrix.shape} disagrees with obs/var {len(units), len(genes)}")
    return matrix, units, genes, coords[:, :2], source


def read_index(path: Path) -> list[str]:
    if path.suffix.lower() == ".tsv":
        frame = pd.read_csv(path, sep="\t", dtype=str)
    else:
        frame = pd.read_csv(path, dtype=str)
    if frame.empty:
        return []
    preferred = ["unit_id", "spot_id", "index", "id"]
    column = next((c for c in preferred if c in frame.columns), frame.columns[0])
    return frame[column].astype(str).tolist()


def infer_index_path(cci_path: Path) -> Path | None:
    name = cci_path.name
    candidates: list[Path] = []
    if name.endswith("_CCI_total.npz"):
        prefix = name[: -len("_CCI_total.npz")]
        candidates.extend([
            cci_path.with_name(prefix + "_index.tsv"),
            cci_path.with_name(prefix + "_index.csv"),
            cci_path.with_name(prefix + "_units.csv"),
        ])
    candidates.extend([
        cci_path.with_name("index.tsv"),
        cci_path.with_name("index.csv"),
        cci_path.with_name("units.csv"),
    ])
    return next((p for p in candidates if p.exists()), None)


def align_stage_to_cci(
    counts: sp.csr_matrix,
    h5_units: Sequence[str],
    coords: np.ndarray,
    cci: sp.csr_matrix,
    cci_units: Sequence[str],
) -> tuple[sp.csr_matrix, list[str], np.ndarray, sp.csr_matrix, dict[str, object]]:
    if cci.shape != (len(cci_units), len(cci_units)):
        raise ValueError(f"CCI shape {cci.shape} does not match index length {len(cci_units)}")
    lookup = {str(unit): idx for idx, unit in enumerate(h5_units)}
    missing = [str(unit) for unit in cci_units if str(unit) not in lookup]
    if missing:
        raise ValueError(f"CCI index has {len(missing)} units absent from H5AD, e.g. {missing[:5]}")
    order = np.asarray([lookup[str(unit)] for unit in cci_units], dtype=np.int32)
    aligned_counts = counts[order, :].tocsr()
    aligned_coords = np.asarray(coords[order], dtype=np.float32)
    values = cci.tocsr().astype(np.float32)
    if values.nnz:
        if not np.isfinite(values.data).all():
            raise ValueError("CCI contains non-finite values")
        values.data = np.maximum(values.data, 0.0)
        values.eliminate_zeros()
    return aligned_counts, list(map(str, cci_units)), aligned_coords, values, {
        "h5ad_unit_count": len(h5_units),
        "cci_unit_count": len(cci_units),
        "unit_alignment": "CCI index order; H5AD rows reordered by exact ID",
        "missing_cci_units_in_h5ad": 0,
        "cci_shape": list(values.shape),
        "cci_nnz": int(values.nnz),
    }


def prepare_true_grn(
    counts: sp.csr_matrix,
    units: Sequence[str],
    genes: Sequence[str],
    grn_path: Path,
    *,
    top_k_targets: int,
    projection_dim: int,
    projection_seed: int,
):
    edges = read_grn_edges(grn_path, top_k_targets_per_regulator=None)
    gene_lookup = {str(g): i for i, g in enumerate(genes)}
    candidate = sorted(
        ({str(g) for g in edges["regulator"]} | {str(g) for g in edges["target"]})
        & set(gene_lookup)
    )
    if not candidate:
        raise ValueError(f"No GRN genes overlap H5AD genes for {grn_path}")
    columns = np.asarray([gene_lookup[g] for g in candidate], dtype=np.int32)
    dense = counts[:, columns].toarray().astype(np.float64)
    expression = pd.DataFrame(dense, index=list(map(str, units)), columns=candidate)
    prepared = prepare_grn_inputs(
        expression,
        units,
        edges,
        top_k_targets=top_k_targets,
    )
    q_reg = deterministic_projection_matrix(
        prepared.genes, role="reg", output_dim=projection_dim, seed=projection_seed
    ).astype(np.float32)
    q_tar = deterministic_projection_matrix(
        prepared.genes, role="tar", output_dim=projection_dim, seed=projection_seed
    ).astype(np.float32)
    g = grn_project(prepared.expression, prepared.adjacency, q_reg, q_tar)
    return prepared, q_reg, q_tar, g, {
        "grn_path": str(grn_path),
        "grn_source": "original_grn_edges_csv",
        "uses_true_grn": True,
        "h5ad_grn_candidate_genes": len(candidate),
        **prepared.metadata,
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
    }


def load_real_stage(
    stage: str,
    *,
    h5ad_path: Path,
    cci_path: Path,
    cci_index_path: Path | None,
    grn_path: Path,
    top_k_targets: int,
    projection_dim: int,
    projection_seed: int,
) -> StageData:
    counts, h5_units, genes, coords, count_source = read_h5ad_counts(h5ad_path)
    cci = sp.load_npz(cci_path).tocsr()
    index_path = cci_index_path or infer_index_path(cci_path)
    if index_path is None:
        raise FileNotFoundError(
            f"No CCI index sidecar found for {cci_path}. Provide --cci-index-t/--cci-index-tp."
        )
    cci_units = read_index(index_path)
    counts, units, coords, cci, alignment = align_stage_to_cci(
        counts, h5_units, coords, cci, cci_units
    )
    prepared, q_reg, q_tar, g, grn_meta = prepare_true_grn(
        counts,
        units,
        genes,
        grn_path,
        top_k_targets=top_k_targets,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    return StageData(
        stage=stage,
        h5ad_path=str(h5ad_path),
        units=units,
        coords=coords,
        cci_proxy=cci,
        g_features=g,
        r_names=[],
        r_values=np.zeros((len(units), 0), dtype=np.float32),
        grn_genes=list(prepared.genes),
        expression_grn=np.asarray(prepared.expression, dtype=np.float32),
        grn_adjacency=prepared.adjacency.astype(np.float32),
        projection_reg=q_reg,
        projection_tar=q_tar,
        metadata={
            "stage": stage,
            "h5ad": str(h5ad_path),
            "count_matrix_source": count_source,
            "cci_path": str(cci_path),
            "cci_index_path": str(index_path),
            "cci_source": "true_CCI_total_npz",
            "uses_true_commot_cci": True,
            **alignment,
            **grn_meta,
        },
    )


def prepare_native_pair(left: StageData, right: StageData, *, nmf_components: int, nmf_max_iter: int, seed: int):
    n_t, n_tp, n_meta = n_pair(left.cci_proxy, right.cci_proxy, nmf_components, nmf_max_iter, seed)
    g_t, g_tp = pairwise_zscore(left.g_features, right.g_features)
    _, pij, v7_meta = native_v7_pij_numpy(n_t, n_tp, g_t, g_tp)
    encoder_t = np.hstack([n_t, g_t]).astype(np.float32)
    encoder_tp = np.hstack([n_tp, g_tp]).astype(np.float32)
    micro_t, micro_tp = joint_fixed_pca(encoder_t, encoder_tp, output_dim=32)
    return {
        "N_t": n_t,
        "N_tp": n_tp,
        "G_t": g_t,
        "G_tp": g_tp,
        "native_pij": pij,
        "native_micro_ei": float(effective_information(pij)),
        "encoder_t": encoder_t,
        "encoder_tp": encoder_tp,
        "micro_features_t": micro_t,
        "micro_features_tp": micro_tp,
        "N_meta": n_meta,
        "V7_meta": v7_meta,
    }


def make_prepared(
    left: StageData,
    right: StageData,
    pair: dict,
    *,
    graph_mode: str,
    grn_graph_weight: float,
    macro_feature_mode: str = "pooled_ng",
) -> PreparedCoarseInput:
    if graph_mode == "cci_only":
        network_t = row_normalize_sparse(left.cci_proxy)
        network_tp = row_normalize_sparse(right.cci_proxy)
    elif graph_mode == "cci_g_integrated":
        network_t = integrate_cci_regsim(
            left.cci_proxy,
            build_regsim_similarity_network(pair["G_t"], k=50),
            regsim_weight=grn_graph_weight,
        )
        network_tp = integrate_cci_regsim(
            right.cci_proxy,
            build_regsim_similarity_network(pair["G_tp"], k=50),
            regsim_weight=grn_graph_weight,
        )
    else:
        raise ValueError(graph_mode)

    if macro_feature_mode not in {"pooled_ng", "recompute_g"}:
        raise ValueError(f"Unknown macro_feature_mode={macro_feature_mode!r}")

    def _pairwise_zscore_torch(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([source, target], dim=0)
        mean = combined.mean(dim=0, keepdim=True)
        std = combined.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        values = (combined - mean) / std
        return values[: source.shape[0]], values[source.shape[0] :]

    sparse_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _torch_grn_assets(stage: StageData, device: torch.device, label: str):
        key = (label, str(device))
        if key not in sparse_cache:
            coo = stage.grn_adjacency.tocoo()
            indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
            values = torch.tensor(coo.data, dtype=torch.float32, device=device)
            adjacency = torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()
            sparse_cache[key] = adjacency
            sparse_cache[(label + "_reg", str(device))] = torch.tensor(stage.projection_reg, dtype=torch.float32, device=device)
            sparse_cache[(label + "_tar", str(device))] = torch.tensor(stage.projection_tar, dtype=torch.float32, device=device)
        return (
            sparse_cache[key],
            sparse_cache[(label + "_reg", str(device))],
            sparse_cache[(label + "_tar", str(device))],
        )

    def _grn_project_torch(expression: torch.Tensor, stage: StageData, label: str) -> torch.Tensor:
        values = torch.clamp(torch.nan_to_num(expression), min=0.0)
        adjacency, q_reg, q_tar = _torch_grn_assets(stage, values.device, label)
        regulator_program = torch.sparse.mm(adjacency, values.T).T
        target_program = torch.sparse.mm(adjacency.transpose(0, 1), values.T).T
        return (values * regulator_program) @ q_reg + (values * target_program) @ q_tar

    def macro_builder(inputs: MacroPijInputs):
        # Both modes use the exact V7 cost and balanced Sinkhorn.  recompute_g additionally
        # reconstructs macro GRN state from pooled expression before applying V7.
        n_t_macro, n_tp_macro = _pairwise_zscore_torch(
            inputs.feature_blocks_t["N"], inputs.feature_blocks_tp["N"]
        )
        if macro_feature_mode == "recompute_g":
            g_t_raw = _grn_project_torch(inputs.feature_blocks_t["X"], left, "t")
            g_tp_raw = _grn_project_torch(inputs.feature_blocks_tp["X"], right, "tp")
            g_t_macro, g_tp_macro = _pairwise_zscore_torch(g_t_raw, g_tp_raw)
        else:
            g_t_macro, g_tp_macro = _pairwise_zscore_torch(
                inputs.feature_blocks_t["G"], inputs.feature_blocks_tp["G"]
            )
        return native_v7_pij_torch(n_t_macro, n_tp_macro, g_t_macro, g_tp_macro)

    feature_blocks_t = {"N": pair["N_t"]}
    feature_blocks_tp = {"N": pair["N_tp"]}
    if macro_feature_mode == "recompute_g":
        feature_blocks_t["X"] = left.expression_grn
        feature_blocks_tp["X"] = right.expression_grn
    else:
        feature_blocks_t["G"] = pair["G_t"]
        feature_blocks_tp["G"] = pair["G_tp"]

    prepared = PreparedCoarseInput(
        method=f"wyt_cg_native_v7__{graph_mode}__macro_{macro_feature_mode}",
        unit_ids_t=left.units,
        unit_ids_tp=right.units,
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=pair["encoder_t"],
        encoder_features_tp=pair["encoder_tp"],
        micro_features_t=pair["micro_features_t"],
        micro_features_tp=pair["micro_features_tp"],
        micro_pij=pair["native_pij"].astype(np.float32),
        micro_ei=pair["native_micro_ei"],
        macro_pij_builder=macro_builder,
        feature_blocks_t=feature_blocks_t,
        feature_blocks_tp=feature_blocks_tp,
        coords_t=left.coords,
        coords_tp=right.coords,
        provenance={
            "feature_extractor": "exact_native_V7_N_plus_true_GRN_G",
            "uses_original_grn_edges_csv": True,
            "uses_true_commot_cci": True,
            "count_matrix_source_t": left.metadata["count_matrix_source"],
            "count_matrix_source_tp": right.metadata["count_matrix_source"],
            "graph_mode": graph_mode,
            "strict_posthoc_recompute_required": True,
            "macro_feature_mode": macro_feature_mode,
            "macro_G_training_interface": (
                "pool_expression_then_recompute_true_GRN_G"
                if macro_feature_mode == "recompute_g"
                else "pool_precomputed_spot_G"
            ),
        },
    )
    prepared.validate()
    return prepared



def project_macro_cci_raw(network: sp.csr_matrix, assignment: np.ndarray) -> sp.csr_matrix:
    """Project true spot CCI to macro units without changing its raw weight scale."""
    macro = assignment.T @ (network @ assignment)
    result = sp.csr_matrix(np.asarray(macro, dtype=np.float32))
    if result.nnz:
        result.data = np.maximum(np.nan_to_num(result.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        result.eliminate_zeros()
    return result


def exact_macro_native_v7_variant(
    left: StageData,
    right: StageData,
    s_t: np.ndarray,
    s_tp: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    normalize_macro_cci: bool,
) -> dict[str, object]:
    a_t = project_macro_cci_raw(left.cci_proxy, s_t)
    a_tp = project_macro_cci_raw(right.cci_proxy, s_tp)
    if normalize_macro_cci:
        a_t = row_normalize_sparse(a_t)
        a_tp = row_normalize_sparse(a_tp)
    x_t = pool(left.expression_grn, s_t)
    x_tp = pool(right.expression_grn, s_tp)
    g_t_raw = grn_project(x_t, left.grn_adjacency, left.projection_reg, left.projection_tar)
    g_tp_raw = grn_project(x_tp, right.grn_adjacency, right.projection_reg, right.projection_tar)
    g_t, g_tp = pairwise_zscore(g_t_raw, g_tp_raw)
    n_t, n_tp, n_meta = n_pair(a_t, a_tp, nmf_components, nmf_max_iter, seed)
    _, pij, v7_meta = native_v7_pij_numpy(n_t, n_tp, g_t, g_tp)
    return {
        "EI_macro": float(effective_information(pij)),
        "macro_cci_normalization": "row_normalized_after_projection" if normalize_macro_cci else "raw_S_transpose_A_S",
        "macro_cci_shape_t": list(a_t.shape),
        "macro_cci_shape_tp": list(a_tp.shape),
        "macro_cci_nnz_t": int(a_t.nnz),
        "macro_cci_nnz_tp": int(a_tp.nnz),
        "N_meta": n_meta,
        "V7_meta": v7_meta,
    }


def strict_eval(
    left: StageData,
    right: StageData,
    pair: dict,
    s_t: np.ndarray,
    s_tp: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
) -> dict[str, object]:
    pooled_n_t = pool(pair["N_t"], s_t)
    pooled_n_tp = pool(pair["N_tp"], s_tp)
    pooled_g_t = pool(pair["G_t"], s_t)
    pooled_g_tp = pool(pair["G_tp"], s_tp)
    _, pooled_pij, pooled_meta = native_v7_pij_numpy(pooled_n_t, pooled_n_tp, pooled_g_t, pooled_g_tp)
    exact_raw = exact_macro_native_v7_variant(
        left,
        right,
        s_t,
        s_tp,
        nmf_components=nmf_components,
        nmf_max_iter=nmf_max_iter,
        seed=seed,
        normalize_macro_cci=False,
    )
    exact_rownorm = exact_macro_native_v7_variant(
        left,
        right,
        s_t,
        s_tp,
        nmf_components=nmf_components,
        nmf_max_iter=nmf_max_iter,
        seed=seed,
        normalize_macro_cci=True,
    )
    micro = pair["native_micro_ei"]
    pooled_ei = float(effective_information(pooled_pij))
    exact_raw_ei = float(exact_raw["EI_macro"])
    exact_rownorm_ei = float(exact_rownorm["EI_macro"])
    return {
        "EI_micro_native_v7": micro,
        "EI_macro_training_interface_pool_NG": pooled_ei,
        "deltaEI_training_interface_pool_NG": pooled_ei - micro,
        "EI_macro_strict_raw_projected_CCI_reextract_N_recompute_G": exact_raw_ei,
        "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G": exact_raw_ei - micro,
        "EI_macro_strict_rownorm_projected_CCI_reextract_N_recompute_G": exact_rownorm_ei,
        "deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G": exact_rownorm_ei - micro,
        "pooled_v7_metadata": pooled_meta,
        "strict_raw_metadata": exact_raw,
        "strict_rownorm_metadata": exact_rownorm,
    }


def assignment_stats(s: np.ndarray) -> dict[str, object]:
    hard = np.argmax(s, axis=1)
    counts = np.bincount(hard, minlength=s.shape[1])
    usage = np.clip(s.mean(axis=0), 1e-12, None)
    positive = counts[counts > 0]
    return {
        "K_requested": int(s.shape[1]),
        "hardK": int(np.count_nonzero(counts)),
        "Keff": float(np.exp(-(usage * np.log(usage)).sum())),
        "max_cluster_fraction": float(counts.max() / len(hard)),
        "min_active_cluster_size": int(positive.min()) if positive.size else 0,
        "max_active_cluster_size": int(positive.max()) if positive.size else 0,
        "mean_assignment_confidence": float(np.max(s, axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="True CCI + true GRN native V7 with WYT coarse-graining")
    parser.add_argument("--stage-t", default="11.5")
    parser.add_argument("--stage-tp", default="12.5")
    parser.add_argument("--h5ad-t", type=Path, required=True)
    parser.add_argument("--h5ad-tp", type=Path, required=True)
    parser.add_argument("--cci-t", type=Path, required=True)
    parser.add_argument("--cci-tp", type=Path, required=True)
    parser.add_argument("--cci-index-t", type=Path)
    parser.add_argument("--cci-index-tp", type=Path)
    parser.add_argument("--grn-t", type=Path, required=True)
    parser.add_argument("--grn-tp", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--graph-modes", nargs="+", default=["cci_only", "cci_g_integrated"])
    parser.add_argument("--local-graph-modes", nargs="+", default=["legacy_features", "coords", "all_features"])
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--grn-topk-targets", type=int, default=50)
    parser.add_argument("--grn-state-dim", type=int, default=64)
    parser.add_argument("--grn-projection-seed", type=int, default=20260713)
    parser.add_argument("--grn-graph-weight", type=float, default=0.2)
    parser.add_argument(
        "--macro-feature-mode",
        choices=["pooled_ng", "recompute_g"],
        default="pooled_ng",
        help="Training macro V7 interface: pool spot N/G, or pool expression and recompute true macro G.",
    )
    parser.add_argument("--lambda-min-usage", type=float, default=10.0)
    parser.add_argument("--lambda-max-usage", type=float, default=10.0)
    parser.add_argument("--min-usage-frac", type=float, default=0.01)
    parser.add_argument("--max-usage-frac", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fixed-map-t", type=Path)
    parser.add_argument("--fixed-map-tp", type=Path)
    args = parser.parse_args()

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    left = load_real_stage(
        args.stage_t,
        h5ad_path=args.h5ad_t,
        cci_path=args.cci_t,
        cci_index_path=args.cci_index_t,
        grn_path=args.grn_t,
        top_k_targets=args.grn_topk_targets,
        projection_dim=args.grn_state_dim,
        projection_seed=args.grn_projection_seed,
    )
    right = load_real_stage(
        args.stage_tp,
        h5ad_path=args.h5ad_tp,
        cci_path=args.cci_tp,
        cci_index_path=args.cci_index_tp,
        grn_path=args.grn_tp,
        top_k_targets=args.grn_topk_targets,
        projection_dim=args.grn_state_dim,
        projection_seed=args.grn_projection_seed,
    )
    pair = prepare_native_pair(
        left, right,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        seed=42,
    )
    with (out / "input_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "left": left.metadata,
            "right": right.metadata,
            "pair": {
                "N": pair["N_meta"],
                "native_micro_ei": pair["native_micro_ei"],
                "V7": pair["V7_meta"],
            },
        }, handle, ensure_ascii=False, indent=2)

    rows: list[dict[str, object]] = []
    if args.fixed_map_t and args.fixed_map_tp:
        s_t, _ = hard_assignment(left.units, args.fixed_map_t)
        s_tp, _ = hard_assignment(right.units, args.fixed_map_tp)
        fixed = strict_eval(
            left, right, pair, s_t, s_tp,
            nmf_components=args.nmf_components,
            nmf_max_iter=args.nmf_max_iter,
            seed=6400,
        )
        fixed.update({
            "run": "fixed_external_mapping",
            "source": "fixed_map",
            "assignment_t": assignment_stats(s_t),
            "assignment_tp": assignment_stats(s_tp),
        })
        rows.append(fixed)

    for graph_mode in args.graph_modes:
        prepared = make_prepared(
            left, right, pair,
            graph_mode=graph_mode,
            grn_graph_weight=args.grn_graph_weight,
            macro_feature_mode=args.macro_feature_mode,
        )
        for local_mode in args.local_graph_modes:
            for seed in args.seeds:
                run_name = f"native_v7__{graph_mode}__macro_{args.macro_feature_mode}__local_{local_mode}__K{args.k}__seed{seed}"
                run_dir = out / run_name
                cfg = WYTDeltaEIConfig(
                    k=args.k,
                    out_dir=run_dir,
                    hidden_dim=64,
                    mid_dim=32,
                    gnn_layers=2,
                    macro_layers=2,
                    knn_k=30,
                    local_dims=2,
                    local_graph_mode=local_mode,
                    temperature=0.07,
                    align_temperature=1.0,
                    epochs=args.epochs,
                    lr=5e-4,
                    lambda_align=1.0,
                    lambda_ei=2.0,
                    lambda_var=1.0,
                    lambda_local=0.0,
                    lambda_sharp=0.02,
                    lambda_proto=0.2,
                    lambda_min_usage=args.lambda_min_usage,
                    lambda_max_usage=args.lambda_max_usage,
                    embedding_target_std=0.05,
                    prototype_max_cosine=0.2,
                    min_usage_frac=args.min_usage_frac,
                    max_usage_frac=args.max_usage_frac,
                    seed=seed,
                    device=args.device,
                    log_every=max(1, args.epochs // 10),
                )
                result = train_deltaei(prepared, cfg)
                s_t = np.asarray(np.load(run_dir / "S_t.npy"), dtype=np.float32)
                s_tp = np.asarray(np.load(run_dir / "S_tp.npy"), dtype=np.float32)
                strict = strict_eval(
                    left, right, pair, s_t, s_tp,
                    nmf_components=args.nmf_components,
                    nmf_max_iter=args.nmf_max_iter,
                    seed=9000 + seed,
                )
                strict.update({
                    "run": run_name,
                    "source": "wyt_learned",
                    "graph_mode": graph_mode,
                    "local_graph_mode": local_mode,
                    "macro_feature_mode": args.macro_feature_mode,
                    "seed": seed,
                    "training_best_epoch": result.best_epoch,
                    "training_EI_macro": result.final_ei_macro,
                    "training_deltaEI": result.final_delta_ei,
                    "assignment_t": assignment_stats(s_t),
                    "assignment_tp": assignment_stats(s_tp),
                })
                with (run_dir / "strict_native_v7_evaluation.json").open("w", encoding="utf-8") as handle:
                    json.dump(strict, handle, ensure_ascii=False, indent=2)
                rows.append(strict)
                flat = []
                for row in rows:
                    flat.append({
                        k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                        for k, v in row.items()
                    })
                pd.DataFrame(flat).to_csv(out / "all_results.csv", index=False)
                print(json.dumps({k: v for k, v in strict.items() if not isinstance(v, dict)}, ensure_ascii=False), flush=True)

    with (out / "formal_data_status.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "true_commot_cci_present": True,
            "original_grn_edges_csv_present": True,
            "proxy_inputs_used": False,
            "experiment_status": "formal_true_CCI_true_GRN_native_V7_WYT",
        }, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
