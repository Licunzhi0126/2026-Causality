#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.metrics import effective_information, pairwise_shared_core_directed_nmf
from mignet_ce.networks.light_cci_grn import deterministic_projection_matrix
from mignet_ce.networks.wyt_cci_regsim import (
    build_regsim_similarity_network,
    integrate_cci_regsim,
    row_normalize_sparse,
)
from mignet_ce.pij.compare.compare_NG_kl_grnanchor_v5 import build_grnanchored_kl_cost
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn
from mignet_ce.pij.compare.compare_NR_kl_sinkhorn_regsim_v7 import (
    regsim_v7_pij_numpy,
    regsim_v7_pij_torch,
)
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from mignet_ce.representations.wyt_network80 import joint_fixed_pca
from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei

EPS = 1e-12


def decode_strings(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def read_csr(group: h5py.Group) -> sp.csr_matrix:
    shape = tuple(int(v) for v in group.attrs["shape"])
    return sp.csr_matrix(
        (
            np.asarray(group["data"]),
            np.asarray(group["indices"], dtype=np.int32),
            np.asarray(group["indptr"], dtype=np.int32),
        ),
        shape=shape,
    )


def dataframe_index(group: h5py.Group) -> list[str]:
    index_name = group.attrs.get("_index", "_index")
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    return decode_strings(np.asarray(group[str(index_name)]))


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class StageData:
    stage: str
    h5ad_path: str
    units: list[str]
    coords: np.ndarray
    cci_proxy: sp.csr_matrix
    g_features: np.ndarray
    r_names: list[str]
    r_values: np.ndarray
    grn_genes: list[str]
    expression_grn: np.ndarray
    grn_adjacency: sp.csr_matrix
    projection_reg: np.ndarray
    projection_tar: np.ndarray
    metadata: dict[str, object]


def load_h5ad_core(path: Path):
    with h5py.File(path, "r") as handle:
        x = read_csr(handle["X"]).astype(np.float32)
        units = dataframe_index(handle["obs"])
        genes = dataframe_index(handle["var"])
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float32)
        obs_keys = list(handle["obs"].keys())
        var_keys = list(handle["var"].keys())
        r_names = sorted(
            key for key in obs_keys
            if key.startswith("Regulon - ") or key.startswith("Module_")
        )
        r_values = np.column_stack(
            [np.asarray(handle["obs"][name], dtype=np.float32) for name in r_names]
        ) if r_names else np.zeros((len(units), 0), dtype=np.float32)
        regulon_names = sorted(key for key in var_keys if key.startswith("Regulon - "))
        regulon_targets = {
            name: np.asarray(handle["var"][name], dtype=bool)
            for name in regulon_names
        }
    return x, units, genes, coords, r_names, r_values, regulon_targets


def build_regulon_grn(
    x: sp.csr_matrix,
    genes: list[str],
    regulon_targets: dict[str, np.ndarray],
    *,
    top_k_targets: int,
    projection_dim: int,
    projection_seed: int,
):
    gene_to_global = {gene: i for i, gene in enumerate(genes)}
    mean_expression = np.asarray(x.mean(axis=0)).ravel()
    edge_triplets: list[tuple[str, str, float]] = []
    skipped_regulators = 0
    for column, mask in regulon_targets.items():
        regulator = column.split("Regulon - ", 1)[1]
        regulator_index = gene_to_global.get(regulator)
        if regulator_index is None:
            skipped_regulators += 1
            continue
        targets = np.flatnonzero(mask)
        targets = targets[targets != regulator_index]
        if targets.size == 0:
            continue
        order = np.argsort(mean_expression[targets])[::-1]
        selected = targets[order[:top_k_targets]]
        for target_index in selected:
            # Binary regulon topology is the only GRN topology recoverable from the uploaded H5AD.
            # A mild expression weight breaks ties without using future time points.
            weight = float(1.0 + np.log1p(max(mean_expression[target_index], 0.0)))
            edge_triplets.append((regulator, genes[int(target_index)], weight))
    if not edge_triplets:
        raise RuntimeError("No regulon-derived GRN edges could be recovered from H5AD var columns.")
    selected_genes = sorted({g for edge in edge_triplets for g in edge[:2]})
    local_index = {gene: i for i, gene in enumerate(selected_genes)}
    rows = np.fromiter((local_index[r] for r, _, _ in edge_triplets), dtype=np.int32)
    cols = np.fromiter((local_index[t] for _, t, _ in edge_triplets), dtype=np.int32)
    weights = np.fromiter((w for _, _, w in edge_triplets), dtype=np.float32)
    adjacency = sp.coo_matrix(
        (weights, (rows, cols)),
        shape=(len(selected_genes), len(selected_genes)),
        dtype=np.float32,
    ).tocsr()
    adjacency.sum_duplicates()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    adjacency = (sp.diags(inverse) @ adjacency).tocsr().astype(np.float32)
    global_indices = np.asarray([gene_to_global[g] for g in selected_genes], dtype=np.int32)
    expression = x[:, global_indices].toarray().astype(np.float32)
    expression = np.maximum(np.nan_to_num(expression), 0.0)
    projection_reg = deterministic_projection_matrix(
        selected_genes, role="reg", output_dim=projection_dim, seed=projection_seed
    ).astype(np.float32)
    projection_tar = deterministic_projection_matrix(
        selected_genes, role="tar", output_dim=projection_dim, seed=projection_seed
    ).astype(np.float32)
    g_features = grn_project(expression, adjacency, projection_reg, projection_tar)
    return (
        selected_genes,
        expression,
        adjacency,
        projection_reg,
        projection_tar,
        g_features,
        {
            "grn_source": "H5AD_var_regulon_membership_topology",
            "is_original_grn_edges_csv": False,
            "regulon_columns": len(regulon_targets),
            "skipped_regulators_missing_as_genes": skipped_regulators,
            "retained_edges": int(adjacency.nnz),
            "retained_genes": len(selected_genes),
            "top_k_targets": top_k_targets,
            "projection_dim": projection_dim,
            "projection_seed": projection_seed,
        },
    )


def grn_project(
    expression: np.ndarray,
    adjacency: sp.csr_matrix,
    projection_reg: np.ndarray,
    projection_tar: np.ndarray,
) -> np.ndarray:
    values = np.maximum(np.nan_to_num(np.asarray(expression, dtype=np.float32)), 0.0)
    regulator_program = np.asarray(adjacency @ values.T, dtype=np.float32).T
    target_program = np.asarray(adjacency.T @ values.T, dtype=np.float32).T
    regulator_state = values * regulator_program
    target_state = values * target_program
    projected = regulator_state @ projection_reg + target_state @ projection_tar
    return np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_expression_spatial_cci_proxy(
    x: sp.csr_matrix,
    coords: np.ndarray,
    *,
    knn_k: int,
    seed: int,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    # This is a deterministic diagnostic proxy, not COMMOT CCI.
    components = min(8, max(2, min(x.shape) - 1))
    latent = TruncatedSVD(n_components=components, random_state=seed).fit_transform(x)
    latent = StandardScaler().fit_transform(latent).astype(np.float32)
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    normalized = latent / np.maximum(norms, 1e-8)
    send = sigmoid(latent[:, 0])
    receive = sigmoid(latent[:, 1] if latent.shape[1] > 1 else -latent[:, 0])
    effective = min(knn_k + 1, len(coords))
    nn = NearestNeighbors(n_neighbors=effective, metric="euclidean").fit(coords)
    distances, indices = nn.kneighbors(coords)
    positive_distances = distances[:, 1:].reshape(-1)
    sigma = float(np.median(positive_distances[positive_distances > 0]))
    sigma = max(sigma, 1e-6)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(len(coords)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            similarity = max(0.0, float((normalized[i] @ normalized[j] + 1.0) * 0.5))
            spatial = math.exp(-0.5 * (float(distance) / sigma) ** 2)
            weight = (
                spatial
                * (0.25 + 0.75 * float(send[i]))
                * (0.25 + 0.75 * float(receive[j]))
                * (0.2 + 0.8 * similarity)
            )
            if weight > 0:
                rows.append(i)
                cols.append(int(j))
                data.append(weight)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(coords), len(coords)), dtype=np.float32)
    matrix.eliminate_zeros()
    return matrix, {
        "cci_source": "expression_spatial_directed_proxy",
        "is_true_commot_cci": False,
        "knn_k": knn_k,
        "latent_components": components,
        "nnz": int(matrix.nnz),
        "sigma": sigma,
    }


def prepare_stage(
    stage: str,
    h5ad_path: Path,
    cache_path: Path,
    *,
    force: bool,
    top_k_targets: int,
    projection_dim: int,
    projection_seed: int,
    cci_knn_k: int,
    seed: int,
) -> StageData:
    if cache_path.exists() and not force:
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    x, units, genes, coords, r_names, r_values, regulon_targets = load_h5ad_core(h5ad_path)
    (
        grn_genes,
        expression_grn,
        grn_adjacency,
        projection_reg,
        projection_tar,
        g_features,
        grn_meta,
    ) = build_regulon_grn(
        x,
        genes,
        regulon_targets,
        top_k_targets=top_k_targets,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
    )
    cci_proxy, cci_meta = build_expression_spatial_cci_proxy(
        x, coords, knn_k=cci_knn_k, seed=seed
    )
    stage_data = StageData(
        stage=stage,
        h5ad_path=str(h5ad_path),
        units=units,
        coords=coords,
        cci_proxy=cci_proxy,
        g_features=g_features,
        r_names=r_names,
        r_values=r_values,
        grn_genes=grn_genes,
        expression_grn=expression_grn,
        grn_adjacency=grn_adjacency,
        projection_reg=projection_reg,
        projection_tar=projection_tar,
        metadata={
            "stage": stage,
            "h5ad": str(h5ad_path),
            "n_units": len(units),
            "n_genes_h5ad": len(genes),
            "regsim_dim": len(r_names),
            **grn_meta,
            **cci_meta,
        },
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(stage_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return stage_data


def pairwise_zscore(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    combined = np.vstack([source, target])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    values = (combined - mean) / std
    return values[: len(source)].astype(np.float32), values[len(source):].astype(np.float32)


def n_pair(a_t: sp.csr_matrix, a_tp: sp.csr_matrix, components: int, max_iter: int, seed: int):
    """Exact shared-core directed NMF with sparse adjacency multiplications.

    This follows the same multiplicative updates as
    ``pairwise_shared_core_directed_nmf`` but avoids densifying nearly dense
    spot CCI matrices.  It changes only the linear-algebra backend, not the
    objective, initialization, update order, rank, or iteration count.
    """
    eps = 1e-10
    source = sp.csr_matrix(a_t, dtype=np.float64)
    target = sp.csr_matrix(a_tp, dtype=np.float64)
    if source.shape[0] != source.shape[1] or target.shape[0] != target.shape[1]:
        raise ValueError("Directed NMF requires square source and target adjacency matrices.")
    if source.nnz:
        source.data = np.maximum(np.nan_to_num(source.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        source.eliminate_zeros()
    if target.nnz:
        target.data = np.maximum(np.nan_to_num(target.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        target.eliminate_zeros()

    rng = np.random.default_rng(seed)
    u_t = rng.random((source.shape[0], components), dtype=float) + eps
    v_t = rng.random((source.shape[0], components), dtype=float) + eps
    u_tp = rng.random((target.shape[0], components), dtype=float) + eps
    v_tp = rng.random((target.shape[0], components), dtype=float) + eps
    core = rng.random((components, components), dtype=float) + eps

    for _ in range(max_iter):
        vtv_t = v_t.T @ v_t
        utu_t = u_t.T @ u_t
        vtv_tp = v_tp.T @ v_tp
        utu_tp = u_tp.T @ u_tp

        u_t *= ((source @ v_t) @ core.T) / (u_t @ core @ vtv_t @ core.T + eps)
        u_tp *= ((target @ v_tp) @ core.T) / (u_tp @ core @ vtv_tp @ core.T + eps)
        u_t = np.maximum(np.nan_to_num(u_t, nan=eps, posinf=eps, neginf=eps), eps)
        u_tp = np.maximum(np.nan_to_num(u_tp, nan=eps, posinf=eps, neginf=eps), eps)

        utu_t = u_t.T @ u_t
        utu_tp = u_tp.T @ u_tp
        v_t *= ((source.T @ u_t) @ core) / (v_t @ core.T @ utu_t @ core + eps)
        v_tp *= ((target.T @ u_tp) @ core) / (v_tp @ core.T @ utu_tp @ core + eps)
        v_t = np.maximum(np.nan_to_num(v_t, nan=eps, posinf=eps, neginf=eps), eps)
        v_tp = np.maximum(np.nan_to_num(v_tp, nan=eps, posinf=eps, neginf=eps), eps)

        vtv_t = v_t.T @ v_t
        vtv_tp = v_tp.T @ v_tp
        numerator = u_t.T @ (source @ v_t) + u_tp.T @ (target @ v_tp)
        denominator = utu_t @ core @ vtv_t + utu_tp @ core @ vtv_tp + eps
        core *= numerator / denominator
        core = np.maximum(np.nan_to_num(core, nan=eps, posinf=eps, neginf=eps), eps)

    n_t, n_tp = pairwise_zscore(np.hstack([u_t, v_t]), np.hstack([u_tp, v_tp]))
    return n_t, n_tp, {
        "core_shape": list(core.shape),
        "components": components,
        "max_iter": max_iter,
        "linear_algebra_backend": "sparse_exact_multiplicative_updates",
    }



def log_domain_balanced_sinkhorn_from_cost(cost: np.ndarray, *, iterations: int = 5000) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.asarray(cost, dtype=np.float64)
    n, m = values.shape
    log_kernel = -values
    log_a = np.full(n, -np.log(n), dtype=np.float64)
    log_b = np.full(m, -np.log(m), dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    for _ in range(iterations):
        f = log_a - logsumexp(log_kernel + g[None, :], axis=1)
        g = log_b - logsumexp(log_kernel + f[:, None], axis=0)
    log_joint = log_kernel + f[:, None] + g[None, :]
    joint = np.exp(log_joint)
    conditional = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)
    source_residual = float(np.max(np.abs(joint.sum(axis=1) - 1.0 / n)))
    target_residual = float(np.max(np.abs(joint.sum(axis=0) - 1.0 / m)))
    return joint, conditional, {
        "mode": "log_domain_balanced_sinkhorn_uniform_marginals",
        "iterations": iterations,
        "source_residual": source_residual,
        "target_residual": target_residual,
        "converged_at_reporting_tolerance_1e-6": max(source_residual, target_residual) <= 1e-6,
    }

def native_v7_pij_numpy(n_t, n_tp, g_t, g_tp):
    cost, metadata = build_grnanchored_kl_cost(n_t, n_tp, g_t, g_tp)
    kernel, pre = row_normalized_kernel_from_cost(cost, tau=1.0)
    try:
        joint, pij, sink_meta = balance_kernel_sinkhorn(kernel)
        sink_meta["study_solver_fallback"] = False
    except RuntimeError as exc:
        # Same balanced entropic-OT equations, evaluated in log-domain to avoid underflow
        # on highly concentrated macro costs. The fallback is recorded in every artifact.
        joint, pij, sink_meta = log_domain_balanced_sinkhorn_from_cost(cost)
        sink_meta["study_solver_fallback"] = True
        sink_meta["original_error"] = str(exc)
    return joint, pij, {"cost": metadata, "sinkhorn": sink_meta, "prebalanced_ei": effective_information(pre)}


def pool(values: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    mass = assignment.sum(axis=0) + EPS
    return (assignment.T @ values) / mass[:, None]


def macro_network(network: sp.csr_matrix, assignment: np.ndarray) -> sp.csr_matrix:
    projected = network @ assignment
    macro = assignment.T @ projected
    return row_normalize_sparse(sp.csr_matrix(macro))


def hard_assignment(units: list[str], map_csv: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(map_csv, dtype=str)
    lookup = dict(zip(frame["spot_id"].astype(str), frame["domain_id"].astype(str), strict=True))
    missing = [u for u in units if u not in lookup]
    if missing:
        raise ValueError(f"{map_csv} is missing {len(missing)} spot IDs, e.g. {missing[:5]}")
    domains = sorted({lookup[u] for u in units})
    index = {d: i for i, d in enumerate(domains)}
    assignment = np.zeros((len(units), len(domains)), dtype=np.float32)
    assignment[np.arange(len(units)), [index[lookup[u]] for u in units]] = 1.0
    return assignment, domains


def exact_macro_native_v7(
    left: StageData,
    right: StageData,
    s_left: np.ndarray,
    s_right: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
):
    a_left = macro_network(left.cci_proxy, s_left)
    a_right = macro_network(right.cci_proxy, s_right)
    x_left = pool(left.expression_grn, s_left)
    x_right = pool(right.expression_grn, s_right)
    g_left_raw = grn_project(x_left, left.grn_adjacency, left.projection_reg, left.projection_tar)
    g_right_raw = grn_project(x_right, right.grn_adjacency, right.projection_reg, right.projection_tar)
    g_left, g_right = pairwise_zscore(g_left_raw, g_right_raw)
    n_left, n_right, n_meta = n_pair(a_left, a_right, nmf_components, nmf_max_iter, seed)
    _, pij, meta = native_v7_pij_numpy(n_left, n_right, g_left, g_right)
    return {
        "EI_macro_exact_native_v7": effective_information(pij),
        "macro_pij": pij,
        "macro_network_t": a_left,
        "macro_network_tp": a_right,
        "N_meta": n_meta,
        "V7_meta": meta,
    }


def common_regsim(left: StageData, right: StageData):
    common = sorted(set(left.r_names).intersection(right.r_names))
    li = [left.r_names.index(name) for name in common]
    ri = [right.r_names.index(name) for name in common]
    return common, left.r_values[:, li], right.r_values[:, ri]


def prepare_pair(left: StageData, right: StageData, nmf_components: int, nmf_max_iter: int, seed: int):
    n_left, n_right, n_meta = n_pair(left.cci_proxy, right.cci_proxy, nmf_components, nmf_max_iter, seed)
    g_left, g_right = pairwise_zscore(left.g_features, right.g_features)
    r_names, r_left_raw, r_right_raw = common_regsim(left, right)
    r_left, r_right = pairwise_zscore(r_left_raw, r_right_raw)
    _, native_pij, native_meta = native_v7_pij_numpy(n_left, n_right, g_left, g_right)
    _, regsim_pij, _, regsim_meta = regsim_v7_pij_numpy(n_left, n_right, r_left, r_right)
    native_encoder_left, native_encoder_right = pairwise_zscore(
        np.hstack([n_left, g_left]), np.hstack([n_right, g_right])
    )
    regsim_encoder_left, regsim_encoder_right = pairwise_zscore(
        np.hstack([n_left, r_left]), np.hstack([n_right, r_right])
    )
    native_micro_left, native_micro_right = joint_fixed_pca(native_encoder_left, native_encoder_right, output_dim=32)
    regsim_micro_left, regsim_micro_right = joint_fixed_pca(regsim_encoder_left, regsim_encoder_right, output_dim=32)
    return {
        "N_t": n_left,
        "N_tp": n_right,
        "G_t": g_left,
        "G_tp": g_right,
        "R_t": r_left,
        "R_tp": r_right,
        "R_names": r_names,
        "native_pij": native_pij,
        "regsim_pij": regsim_pij,
        "native_micro_ei": effective_information(native_pij),
        "regsim_micro_ei": effective_information(regsim_pij),
        "native_encoder_t": native_encoder_left,
        "native_encoder_tp": native_encoder_right,
        "regsim_encoder_t": regsim_encoder_left,
        "regsim_encoder_tp": regsim_encoder_right,
        "native_micro_features_t": native_micro_left,
        "native_micro_features_tp": native_micro_right,
        "regsim_micro_features_t": regsim_micro_left,
        "regsim_micro_features_tp": regsim_micro_right,
        "N_meta": n_meta,
        "native_meta": native_meta,
        "regsim_meta": regsim_meta,
    }


def make_prepared(
    method: str,
    left: StageData,
    right: StageData,
    pair_data: dict,
    *,
    feature_kind: str,
    network_mode: str,
    regsim_weight: float,
):
    if feature_kind == "native_g":
        block_name = "G"
        anchor_left, anchor_right = pair_data["G_t"], pair_data["G_tp"]
        encoder_left, encoder_right = pair_data["native_encoder_t"], pair_data["native_encoder_tp"]
        micro_left, micro_right = pair_data["native_micro_features_t"], pair_data["native_micro_features_tp"]
        micro_pij = pair_data["native_pij"]
        micro_ei = pair_data["native_micro_ei"]
    elif feature_kind == "regsim_r":
        block_name = "R"
        anchor_left, anchor_right = pair_data["R_t"], pair_data["R_tp"]
        encoder_left, encoder_right = pair_data["regsim_encoder_t"], pair_data["regsim_encoder_tp"]
        micro_left, micro_right = pair_data["regsim_micro_features_t"], pair_data["regsim_micro_features_tp"]
        micro_pij = pair_data["regsim_pij"]
        micro_ei = pair_data["regsim_micro_ei"]
    else:
        raise ValueError(feature_kind)
    if network_mode == "cci_only":
        network_left = row_normalize_sparse(left.cci_proxy)
        network_right = row_normalize_sparse(right.cci_proxy)
    elif network_mode == "cci_anchor_integrated":
        network_left = integrate_cci_regsim(
            left.cci_proxy,
            build_regsim_similarity_network(anchor_left, k=50),
            regsim_weight=regsim_weight,
        )
        network_right = integrate_cci_regsim(
            right.cci_proxy,
            build_regsim_similarity_network(anchor_right, k=50),
            regsim_weight=regsim_weight,
        )
    else:
        raise ValueError(network_mode)

    def macro_builder(inputs: MacroPijInputs):
        return regsim_v7_pij_torch(
            inputs.feature_blocks_t["N"],
            inputs.feature_blocks_tp["N"],
            inputs.feature_blocks_t[block_name],
            inputs.feature_blocks_tp[block_name],
        )

    prepared = PreparedCoarseInput(
        method=method,
        unit_ids_t=left.units,
        unit_ids_tp=right.units,
        network_t=network_left,
        network_tp=network_right,
        encoder_features_t=encoder_left,
        encoder_features_tp=encoder_right,
        micro_features_t=micro_left,
        micro_features_tp=micro_right,
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=float(micro_ei),
        macro_pij_builder=macro_builder,
        feature_blocks_t={"N": pair_data["N_t"], block_name: anchor_left},
        feature_blocks_tp={"N": pair_data["N_tp"], block_name: anchor_right},
        coords_t=left.coords,
        coords_tp=right.coords,
        provenance={
            "feature_kind": feature_kind,
            "network_mode": network_mode,
            "uses_original_grn_edges_csv": False,
            "uses_true_commot_cci": False,
            "strict_posthoc_recompute_required": True,
        },
    )
    prepared.validate()
    return prepared


def load_assignment(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def evaluate_assignment_set(
    label: str,
    left: StageData,
    right: StageData,
    pair_data: dict,
    s_left: np.ndarray,
    s_right: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
):
    pooled_n_left = pool(pair_data["N_t"], s_left)
    pooled_n_right = pool(pair_data["N_tp"], s_right)
    pooled_g_left = pool(pair_data["G_t"], s_left)
    pooled_g_right = pool(pair_data["G_tp"], s_right)
    pooled_r_left = pool(pair_data["R_t"], s_left)
    pooled_r_right = pool(pair_data["R_tp"], s_right)
    _, pooled_g_pij, _ = native_v7_pij_numpy(
        pooled_n_left, pooled_n_right, pooled_g_left, pooled_g_right
    )
    _, pooled_r_pij, _ = native_v7_pij_numpy(
        pooled_n_left, pooled_n_right, pooled_r_left, pooled_r_right
    )
    exact = exact_macro_native_v7(
        left,
        right,
        s_left,
        s_right,
        nmf_components=nmf_components,
        nmf_max_iter=nmf_max_iter,
        seed=seed,
    )
    return {
        "label": label,
        "K_t": int(s_left.shape[1]),
        "K_tp": int(s_right.shape[1]),
        "EI_micro_native": float(pair_data["native_micro_ei"]),
        "EI_micro_regsim": float(pair_data["regsim_micro_ei"]),
        "EI_macro_pooled_NG": float(effective_information(pooled_g_pij)),
        "deltaEI_pooled_NG_vs_native_micro": float(effective_information(pooled_g_pij) - pair_data["native_micro_ei"]),
        "EI_macro_pooled_NR": float(effective_information(pooled_r_pij)),
        "deltaEI_pooled_NR_vs_regsim_micro": float(effective_information(pooled_r_pij) - pair_data["regsim_micro_ei"]),
        "EI_macro_exact_native": float(exact["EI_macro_exact_native_v7"]),
        "deltaEI_exact_native": float(exact["EI_macro_exact_native_v7"] - pair_data["native_micro_ei"]),
    }


def parse_pair(text: str) -> tuple[str, str]:
    left, right = text.split("->", 1)
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=["11.5->12.5", "12.5->13.5", "13.5->14.5"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[40, 150])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=100)
    parser.add_argument("--top-k-targets", type=int, default=50)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--cci-knn-k", type=int, default=30)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--feature-kinds", nargs="+", default=["native_g", "regsim_r"])
    parser.add_argument("--network-modes", nargs="+", default=["cci_only", "cci_anchor_integrated"])
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_root / "cache"
    stages = sorted({stage for pair in args.pairs for stage in parse_pair(pair)}, key=float)
    stage_data: dict[str, StageData] = {}
    for stage in stages:
        h5ad = args.data_root / "spot" / "heart" / f"spot_heart_{stage}.h5ad"
        stage_data[stage] = prepare_stage(
            stage,
            h5ad,
            cache_dir / f"stage_{stage.replace('.', 'p')}.pkl",
            force=args.force_cache,
            top_k_targets=args.top_k_targets,
            projection_dim=args.projection_dim,
            projection_seed=args.projection_seed,
            cci_knn_k=args.cci_knn_k,
            seed=42 + int(float(stage) * 10),
        )
        print(f"[stage] {stage}: {stage_data[stage].metadata}", flush=True)

    all_rows: list[dict[str, object]] = []
    for pair_text in args.pairs:
        left_stage, right_stage = parse_pair(pair_text)
        left, right = stage_data[left_stage], stage_data[right_stage]
        pair_dir = args.out_root / pair_text.replace("->", "_to_")
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_data = prepare_pair(left, right, args.nmf_components, args.nmf_max_iter, seed=42)
        with (pair_dir / "pair_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pair": pair_text,
                    "left": left.metadata,
                    "right": right.metadata,
                    "N": pair_data["N_meta"],
                    "R_common_dim": len(pair_data["R_names"]),
                    "native_micro_ei": pair_data["native_micro_ei"],
                    "regsim_micro_ei": pair_data["regsim_micro_ei"],
                }, handle, indent=2, ensure_ascii=False
            )
        # Fixed Seurat maps: this is the clean B-style interface test.
        for k in args.k_values:
            prefix = "seurat" if k == 40 else "seurat150"
            folder = "seurat_k40" if k == 40 else "seurat_k150"
            map_left = args.data_root / folder / "heart" / f"{prefix}_heart_{left_stage}_spot_domain_map.csv"
            map_right = args.data_root / folder / "heart" / f"{prefix}_heart_{right_stage}_spot_domain_map.csv"
            s_left, _ = hard_assignment(left.units, map_left)
            s_right, _ = hard_assignment(right.units, map_right)
            row = evaluate_assignment_set(
                f"seurat_k{k}_fixed",
                left, right, pair_data, s_left, s_right,
                nmf_components=args.nmf_components,
                nmf_max_iter=args.nmf_max_iter,
                seed=4200 + k,
            )
            row.update({"pair": pair_text, "source": "fixed_seurat", "K_requested": k})
            all_rows.append(row)
            print(f"[fixed] {pair_text} K{k}: {row}", flush=True)

        if args.skip_training:
            continue
        for feature_kind in args.feature_kinds:
            for network_mode in args.network_modes:
                for k in args.k_values:
                    for seed in args.seeds:
                        run_name = f"{feature_kind}__{network_mode}__K{k}__seed{seed}"
                        run_dir = pair_dir / run_name
                        prepared = make_prepared(
                            method=f"wyt_cg_{feature_kind}_{network_mode}",
                            left=left,
                            right=right,
                            pair_data=pair_data,
                            feature_kind=feature_kind,
                            network_mode=network_mode,
                            regsim_weight=0.2,
                        )
                        config = WYTDeltaEIConfig(
                            k=k,
                            out_dir=run_dir,
                            hidden_dim=64,
                            mid_dim=32,
                            gnn_layers=2,
                            macro_layers=2,
                            knn_k=30,
                            local_dims=2,
                            local_graph_mode="coords",
                            temperature=0.07,
                            align_temperature=1.0,
                            epochs=args.epochs,
                            lr=5e-4,
                            lambda_align=1.0,
                            lambda_ei=1.0,
                            lambda_var=1.0,
                            lambda_local=0.1,
                            lambda_sharp=0.02,
                            lambda_proto=0.2,
                            lambda_min_usage=10.0,
                            lambda_max_usage=10.0,
                            embedding_target_std=0.05,
                            prototype_max_cosine=0.2,
                            min_usage_frac=0.01,
                            max_usage_frac=8.0,
                            seed=seed,
                            device="cpu",
                            log_every=max(1, args.epochs // 6),
                        )
                        if (run_dir / "S_t.npy").exists() and (run_dir / "summary.json").exists():
                            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                                summary = json.load(handle)
                            class ExistingResult:
                                final_delta_ei = float(summary["delta_EI_best_checkpoint"])
                                final_ei_macro = float(summary["EI_macro_best_checkpoint"])
                                best_epoch = int(summary["best_epoch"])
                            result = ExistingResult()
                        else:
                            result = train_deltaei(prepared, config)
                        s_left = load_assignment(run_dir / "S_t.npy")
                        s_right = load_assignment(run_dir / "S_tp.npy")
                        row = evaluate_assignment_set(
                            run_name,
                            left, right, pair_data, s_left, s_right,
                            nmf_components=args.nmf_components,
                            nmf_max_iter=args.nmf_max_iter,
                            seed=seed + 9000,
                        )
                        row.update({
                            "pair": pair_text,
                            "source": "wyt_learned",
                            "feature_kind": feature_kind,
                            "network_mode": network_mode,
                            "K_requested": k,
                            "seed": seed,
                            "training_deltaEI_best_checkpoint": result.final_delta_ei,
                            "training_EI_macro_best_checkpoint": result.final_ei_macro,
                            "best_epoch": result.best_epoch,
                        })
                        with (run_dir / "strict_native_v7_evaluation.json").open("w", encoding="utf-8") as handle:
                            json.dump(row, handle, indent=2, ensure_ascii=False)
                        all_rows.append(row)
                        pd.DataFrame(all_rows).to_csv(args.out_root / "all_results.csv", index=False)
                        print(f"[run] {pair_text} {run_name}: {row}", flush=True)
    pd.DataFrame(all_rows).to_csv(args.out_root / "all_results.csv", index=False)
    with (args.out_root / "data_limitations.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "true_commot_cci_present": False,
            "original_grn_edges_csv_present": False,
            "experiment_status": "diagnostic_proxy_run_not_final_true_CCI_reproduction",
            "what_is_exact": [
                "V7 directed shared-core NMF feature path",
                "V7 pairwise z-score",
                "V7 double-end expression-gated GRN state equation",
                "V7 deterministic gene-role projection",
                "V7 raw G KL + 0.25 normalized N KL cost",
                "V7 balanced Sinkhorn",
                "WYT coarse-grain architecture and losses",
                "post-hoc macro feature recomputation",
            ],
            "what_is_proxy": [
                "CCI reconstructed as expression-spatial directed proxy because uploaded data contain no CCI_total.npz",
                "GRN topology recovered from H5AD regulon memberships because uploaded data contain no grn_edges.csv",
            ],
        }, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
