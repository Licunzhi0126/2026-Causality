from __future__ import annotations

"""Complete-combined coarse-graining infrastructure.

This module owns data loading, feature construction, Native-V7 PIJ evaluation,
and strict post-hoc diagnostics.  ``mignet_ce.coarse_frontends`` only keeps the
thin method entrypoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import logsumexp

from mignet_ce.io.loaders import read_grn_edges
from mignet_ce.metrics import effective_information
from mignet_ce.networks.light_cci_grn import (
    deterministic_projection_matrix,
    prepare_grn_inputs,
)
from mignet_ce.networks.wyt_cci_regsim import row_normalize_sparse
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.ng_kl_ot import (
    NATIVE_V7_FEATURE_BETA,
    NATIVE_V7_G_SCALE,
    NATIVE_V7_N_WEIGHT,
    build_ng_kl_cost_numpy,
    native_v7_pij_torch,
)
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import (
    balance_kernel_sinkhorn,
)
from mignet_ce.representations.coarse_input import MacroPijInputs

EPS = 1e-12


@dataclass(frozen=True)
class CompleteCombinedStage:
    units: list[str]
    cci: sp.csr_matrix
    grn_genes: list[str]
    expression_grn: np.ndarray
    grn_adjacency: sp.csr_matrix
    projection_reg: np.ndarray
    projection_tar: np.ndarray
    g_raw: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class CompleteCombinedPair:
    n_t: np.ndarray
    n_tp: np.ndarray
    g_t: np.ndarray
    g_tp: np.ndarray
    encoder_t: np.ndarray
    encoder_tp: np.ndarray
    micro_features_t: np.ndarray
    micro_features_tp: np.ndarray
    micro_pij: np.ndarray
    micro_ei: float
    n_metadata: dict[str, object]
    v7_metadata: dict[str, object]


def _pairwise_zscore_torch(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([source, target], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    values = (combined - mean) / std
    return values[: source.shape[0]], values[source.shape[0] :]


def build_macro_pij_builder(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
):
    """Build the differentiable macro PIJ callback used by the trainer."""
    cache: dict[tuple[str, str], torch.Tensor] = {}

    def torch_assets(stage: CompleteCombinedStage, label: str, device: torch.device):
        device_key = str(device)
        adjacency_key = (f"{label}_adjacency", device_key)
        if adjacency_key not in cache:
            coo = stage.grn_adjacency.tocoo()
            indices = torch.tensor(
                np.vstack([coo.row, coo.col]),
                dtype=torch.long,
                device=device,
            )
            values = torch.tensor(coo.data, dtype=torch.float32, device=device)
            cache[adjacency_key] = torch.sparse_coo_tensor(
                indices,
                values,
                size=coo.shape,
                dtype=torch.float32,
                device=device,
                check_invariants=True,
            ).coalesce()
            cache[(f"{label}_projection_reg", device_key)] = torch.tensor(
                stage.projection_reg,
                dtype=torch.float32,
                device=device,
            )
            cache[(f"{label}_projection_tar", device_key)] = torch.tensor(
                stage.projection_tar,
                dtype=torch.float32,
                device=device,
            )
        return (
            cache[adjacency_key],
            cache[(f"{label}_projection_reg", device_key)],
            cache[(f"{label}_projection_tar", device_key)],
        )

    def project(
        expression: torch.Tensor,
        stage: CompleteCombinedStage,
        label: str,
    ) -> torch.Tensor:
        values = torch.clamp(
            torch.nan_to_num(expression, nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        adjacency, projection_reg, projection_tar = torch_assets(
            stage,
            label,
            values.device,
        )
        regulator_program = torch.sparse.mm(adjacency, values.T).T
        target_program = torch.sparse.mm(adjacency.transpose(0, 1), values.T).T
        return (
            (values * regulator_program) @ projection_reg
            + (values * target_program) @ projection_tar
        )

    def build(inputs: MacroPijInputs) -> torch.Tensor:
        n_t, n_tp = _pairwise_zscore_torch(
            inputs.feature_blocks_t["N"],
            inputs.feature_blocks_tp["N"],
        )
        g_t, g_tp = _pairwise_zscore_torch(
            project(inputs.feature_blocks_t["X"], stage_t, "t"),
            project(inputs.feature_blocks_tp["X"], stage_tp, "tp"),
        )
        return native_v7_pij_torch(n_t, n_tp, g_t, g_tp)

    return build


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _dataframe_index(group: h5py.Group) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return _decode_strings(np.asarray(group[str(key)]))


def _read_h5_matrix(node: h5py.Dataset | h5py.Group) -> sp.csr_matrix:
    if isinstance(node, h5py.Dataset):
        return sp.csr_matrix(np.asarray(node))
    shape = tuple(int(value) for value in node.attrs["shape"])
    encoding = node.attrs.get("encoding-type", "csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    payload = (
        np.asarray(node["data"]),
        np.asarray(node["indices"], dtype=np.int64),
        np.asarray(node["indptr"], dtype=np.int64),
    )
    if str(encoding) == "csc_matrix":
        return sp.csc_matrix(payload, shape=shape).tocsr()
    return sp.csr_matrix(payload, shape=shape)


def read_h5ad_counts(path: Path) -> tuple[sp.csr_matrix, list[str], list[str], str]:
    """Read count/counts/raw.X/X without loading unrelated AnnData payloads."""
    with h5py.File(path, "r") as handle:
        units = _dataframe_index(handle["obs"])
        if "layers" in handle and "count" in handle["layers"]:
            matrix_node = handle["layers/count"]
            genes = _dataframe_index(handle["var"])
            source = "layers/count"
        elif "layers" in handle and "counts" in handle["layers"]:
            matrix_node = handle["layers/counts"]
            genes = _dataframe_index(handle["var"])
            source = "layers/counts"
        elif "raw" in handle and "X" in handle["raw"] and "var" in handle["raw"]:
            matrix_node = handle["raw/X"]
            genes = _dataframe_index(handle["raw/var"])
            source = "raw.X"
        else:
            matrix_node = handle["X"]
            genes = _dataframe_index(handle["var"])
            source = "X"
        matrix = _read_h5_matrix(matrix_node).astype(np.float32)
    if matrix.shape != (len(units), len(genes)):
        raise ValueError(
            f"H5AD count matrix {matrix.shape} disagrees with "
            f"obs/var dimensions {(len(units), len(genes))}: {path}"
        )
    if matrix.nnz:
        matrix.data = np.maximum(
            np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        matrix.eliminate_zeros()
    return matrix, units, genes, source


def project_grn_state(
    expression: np.ndarray,
    adjacency: sp.spmatrix,
    projection_reg: np.ndarray,
    projection_tar: np.ndarray,
) -> np.ndarray:
    values = np.maximum(
        np.nan_to_num(np.asarray(expression, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    matrix = adjacency.tocsr().astype(np.float32)
    regulator_program = np.asarray(matrix @ values.T, dtype=np.float32).T
    target_program = np.asarray(matrix.T @ values.T, dtype=np.float32).T
    projected = (
        (values * regulator_program) @ np.asarray(projection_reg, dtype=np.float32)
        + (values * target_program) @ np.asarray(projection_tar, dtype=np.float32)
    )
    return np.nan_to_num(
        projected,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def prepare_complete_stage(
    *,
    h5ad_path: Path,
    grn_path: Path,
    units: Sequence[str],
    cci: sp.csr_matrix,
    top_k_targets: int,
    state_dim: int,
    projection_seed: int,
) -> CompleteCombinedStage:
    counts, h5_units, genes, count_source = read_h5ad_counts(h5ad_path)
    if len(h5_units) != len(set(h5_units)):
        raise ValueError(f"H5AD contains duplicate observation IDs: {h5ad_path}")
    lookup = {unit: index for index, unit in enumerate(h5_units)}
    aligned_units = list(map(str, units))
    missing = [unit for unit in aligned_units if unit not in lookup]
    if missing:
        raise ValueError(
            f"CCI units are missing from H5AD {h5ad_path}: {missing[:10]}"
        )
    order = np.asarray([lookup[unit] for unit in aligned_units], dtype=np.int64)
    aligned_counts = counts[order, :].tocsr()

    edges = read_grn_edges(grn_path, top_k_targets_per_regulator=None)
    gene_lookup = {str(gene): index for index, gene in enumerate(genes)}
    grn_candidates = sorted(
        (
            set(edges["regulator"].astype(str))
            | set(edges["target"].astype(str))
        )
        & set(gene_lookup)
    )
    if not grn_candidates:
        raise ValueError(f"No GRN genes overlap H5AD genes for {grn_path}")
    columns = np.asarray([gene_lookup[gene] for gene in grn_candidates], dtype=np.int64)
    expression = pd.DataFrame(
        aligned_counts[:, columns].toarray().astype(np.float64),
        index=aligned_units,
        columns=grn_candidates,
    )
    prepared = prepare_grn_inputs(
        expression,
        aligned_units,
        edges,
        top_k_targets=top_k_targets,
    )
    projection_reg = deterministic_projection_matrix(
        prepared.genes,
        role="reg",
        output_dim=state_dim,
        seed=projection_seed,
    ).astype(np.float32)
    projection_tar = deterministic_projection_matrix(
        prepared.genes,
        role="tar",
        output_dim=state_dim,
        seed=projection_seed,
    ).astype(np.float32)
    g_raw = project_grn_state(
        prepared.expression,
        prepared.adjacency,
        projection_reg,
        projection_tar,
    )
    return CompleteCombinedStage(
        units=aligned_units,
        cci=cci.tocsr().astype(np.float32),
        grn_genes=list(prepared.genes),
        expression_grn=np.asarray(prepared.expression, dtype=np.float32),
        grn_adjacency=prepared.adjacency.tocsr().astype(np.float32),
        projection_reg=projection_reg,
        projection_tar=projection_tar,
        g_raw=g_raw,
        metadata={
            "h5ad_path": str(Path(h5ad_path).resolve()),
            "count_matrix_source": count_source,
            "grn_path": str(Path(grn_path).resolve()),
            "grn_source": "original_grn_edges_csv",
            "uses_true_grn": True,
            "uses_true_commot_cci": True,
            "unit_alignment": "CCI index order; H5AD rows reordered by exact ID",
            "missing_cci_units_in_h5ad": 0,
            "h5ad_grn_candidate_genes": int(len(grn_candidates)),
            **prepared.metadata,
            "grn_projection_seed": int(projection_seed),
            "grn_state_dim": int(state_dim),
            "grn_state_shape": list(g_raw.shape),
        },
    )


def pairwise_zscore(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(source, dtype=np.float32)
    right = np.asarray(target, dtype=np.float32)
    combined = np.vstack([left, right])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    values = (combined - mean) / std
    return values[: len(left)].astype(np.float32), values[len(left) :].astype(np.float32)


def sparse_shared_core_directed_nmf(
    source: sp.spmatrix,
    target: sp.spmatrix,
    *,
    components: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Shared-core directed NMF using sparse multiplications without densifying CCI."""
    if components <= 0 or max_iter < 0:
        raise ValueError("NMF components must be positive and max_iter non-negative.")
    left = source.tocsr().astype(np.float64)
    right = target.tocsr().astype(np.float64)
    if left.shape[0] != left.shape[1] or right.shape[0] != right.shape[1]:
        raise ValueError("Directed NMF requires square source and target matrices.")
    for values in (left, right):
        if values.nnz:
            values.data = np.maximum(
                np.nan_to_num(values.data, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )
            values.eliminate_zeros()

    eps = 1e-10
    rng = np.random.default_rng(seed)
    u_t = rng.random((left.shape[0], components), dtype=float) + eps
    v_t = rng.random((left.shape[0], components), dtype=float) + eps
    u_tp = rng.random((right.shape[0], components), dtype=float) + eps
    v_tp = rng.random((right.shape[0], components), dtype=float) + eps
    core = rng.random((components, components), dtype=float) + eps

    for _ in range(max_iter):
        vtv_t = v_t.T @ v_t
        utu_t = u_t.T @ u_t
        vtv_tp = v_tp.T @ v_tp
        utu_tp = u_tp.T @ u_tp

        u_t *= ((left @ v_t) @ core.T) / (
            u_t @ core @ vtv_t @ core.T + eps
        )
        u_tp *= ((right @ v_tp) @ core.T) / (
            u_tp @ core @ vtv_tp @ core.T + eps
        )
        u_t = np.maximum(np.nan_to_num(u_t, nan=eps, posinf=eps, neginf=eps), eps)
        u_tp = np.maximum(np.nan_to_num(u_tp, nan=eps, posinf=eps, neginf=eps), eps)

        utu_t = u_t.T @ u_t
        utu_tp = u_tp.T @ u_tp
        v_t *= ((left.T @ u_t) @ core) / (
            v_t @ core.T @ utu_t @ core + eps
        )
        v_tp *= ((right.T @ u_tp) @ core) / (
            v_tp @ core.T @ utu_tp @ core + eps
        )
        v_t = np.maximum(np.nan_to_num(v_t, nan=eps, posinf=eps, neginf=eps), eps)
        v_tp = np.maximum(np.nan_to_num(v_tp, nan=eps, posinf=eps, neginf=eps), eps)

        vtv_t = v_t.T @ v_t
        vtv_tp = v_tp.T @ v_tp
        numerator = u_t.T @ (left @ v_t) + u_tp.T @ (right @ v_tp)
        denominator = utu_t @ core @ vtv_t + utu_tp @ core @ vtv_tp + eps
        core *= numerator / denominator
        core = np.maximum(
            np.nan_to_num(core, nan=eps, posinf=eps, neginf=eps),
            eps,
        )

    n_t, n_tp = pairwise_zscore(
        np.hstack([u_t, v_t]),
        np.hstack([u_tp, v_tp]),
    )
    return n_t, n_tp, {
        "definition": "pairwise_shared_core_directed_nmf_concat_outgoing_U_incoming_V",
        "components": int(components),
        "max_iter": int(max_iter),
        "seed": int(seed),
        "core_shape": list(core.shape),
        "linear_algebra_backend": "sparse_exact_multiplicative_updates",
        "shape_t": list(n_t.shape),
        "shape_tp": list(n_tp.shape),
    }


def _log_domain_sinkhorn(
    cost: np.ndarray,
    *,
    iterations: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.asarray(cost, dtype=np.float64)
    source_count, target_count = values.shape
    log_kernel = -values
    log_source = np.full(source_count, -np.log(source_count), dtype=np.float64)
    log_target = np.full(target_count, -np.log(target_count), dtype=np.float64)
    source_potential = np.zeros(source_count, dtype=np.float64)
    target_potential = np.zeros(target_count, dtype=np.float64)
    for _ in range(iterations):
        source_potential = log_source - logsumexp(
            log_kernel + target_potential[None, :],
            axis=1,
        )
        target_potential = log_target - logsumexp(
            log_kernel + source_potential[:, None],
            axis=0,
        )
    joint = np.exp(
        log_kernel + source_potential[:, None] + target_potential[None, :]
    )
    conditional = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)
    source_residual = float(
        np.max(np.abs(joint.sum(axis=1) - 1.0 / source_count))
    )
    target_residual = float(
        np.max(np.abs(joint.sum(axis=0) - 1.0 / target_count))
    )
    return joint, conditional, {
        "mode": "log_domain_balanced_sinkhorn_uniform_marginals",
        "iterations": int(iterations),
        "source_residual": source_residual,
        "target_residual": target_residual,
        "converged_at_reporting_tolerance_1e-6": (
            max(source_residual, target_residual) <= 1e-6
        ),
    }


def native_v7_pij_numpy(
    n_t: np.ndarray,
    n_tp: np.ndarray,
    g_t: np.ndarray,
    g_tp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cost, cost_metadata = build_ng_kl_cost_numpy(
        n_t,
        n_tp,
        g_t,
        g_tp,
        beta_n=NATIVE_V7_FEATURE_BETA,
        beta_g=NATIVE_V7_FEATURE_BETA,
        g_scale=NATIVE_V7_G_SCALE,
        n_weight=NATIVE_V7_N_WEIGHT,
    )
    cost_metadata.update(
        {
            "compatibility_formula": "frozen_Native_V7",
            "parameter_selection_split": "adjacent_time_pairs_development_only",
            "heldout_split_observed": False,
        }
    )
    kernel, prebalanced = row_normalized_kernel_from_cost(cost, tau=1.0)
    try:
        joint, pij, sinkhorn_metadata = balance_kernel_sinkhorn(kernel)
        sinkhorn_metadata["log_domain_fallback_used"] = False
    except RuntimeError as error:
        joint, pij, sinkhorn_metadata = _log_domain_sinkhorn(cost)
        sinkhorn_metadata["log_domain_fallback_used"] = True
        sinkhorn_metadata["original_error"] = str(error)
    return joint, pij, {
        "cost": cost_metadata,
        "sinkhorn": sinkhorn_metadata,
        "prebalanced_ei": float(effective_information(prebalanced.copy())),
    }


def prepare_complete_pair(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    mid_dim: int,
) -> CompleteCombinedPair:
    from mignet_ce.representations.wyt_network80 import joint_fixed_pca

    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        stage_t.cci,
        stage_tp.cci,
        components=nmf_components,
        max_iter=nmf_max_iter,
        seed=seed,
    )
    g_t, g_tp = pairwise_zscore(stage_t.g_raw, stage_tp.g_raw)
    _, micro_pij, v7_metadata = native_v7_pij_numpy(n_t, n_tp, g_t, g_tp)
    encoder_t, encoder_tp = pairwise_zscore(
        np.hstack([n_t, g_t]),
        np.hstack([n_tp, g_tp]),
    )
    micro_t, micro_tp = joint_fixed_pca(
        encoder_t,
        encoder_tp,
        output_dim=mid_dim,
    )
    return CompleteCombinedPair(
        n_t=n_t,
        n_tp=n_tp,
        g_t=g_t,
        g_tp=g_tp,
        encoder_t=encoder_t,
        encoder_tp=encoder_tp,
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=float(effective_information(micro_pij.copy())),
        n_metadata=n_metadata,
        v7_metadata=v7_metadata,
    )


def pool_features(values: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    mass = np.asarray(assignment, dtype=np.float64).sum(axis=0) + EPS
    return (
        np.asarray(assignment, dtype=np.float64).T
        @ np.asarray(values, dtype=np.float64)
    ) / mass[:, None]


def project_macro_cci_raw(
    network: sp.csr_matrix,
    assignment: np.ndarray,
) -> sp.csr_matrix:
    macro = np.asarray(assignment).T @ (network @ np.asarray(assignment))
    result = sp.csr_matrix(np.asarray(macro, dtype=np.float32))
    if result.nnz:
        result.data = np.maximum(
            np.nan_to_num(result.data, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        result.eliminate_zeros()
    return result


def _assignment_stats(assignment: np.ndarray) -> dict[str, object]:
    values = np.asarray(assignment, dtype=np.float64)
    hard = np.argmax(values, axis=1)
    counts = np.bincount(hard, minlength=values.shape[1])
    usage = np.clip(values.mean(axis=0), EPS, None)
    usage /= usage.sum()
    positive = counts[counts > 0]
    return {
        "K_requested": int(values.shape[1]),
        "hardK": int(np.count_nonzero(counts)),
        "Keff": float(np.exp(-(usage * np.log(usage)).sum())),
        "max_cluster_fraction": float(counts.max() / len(hard)),
        "min_active_cluster_size": int(positive.min()) if positive.size else 0,
        "max_active_cluster_size": int(positive.max()) if positive.size else 0,
        "mean_assignment_confidence": float(values.max(axis=1).mean()),
    }


def _evaluate_exact_macro(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    assignment_t: np.ndarray,
    assignment_tp: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    normalize_macro_cci: bool,
) -> dict[str, object]:
    cci_t = project_macro_cci_raw(stage_t.cci, assignment_t)
    cci_tp = project_macro_cci_raw(stage_tp.cci, assignment_tp)
    if normalize_macro_cci:
        cci_t = row_normalize_sparse(cci_t)
        cci_tp = row_normalize_sparse(cci_tp)
    expression_t = pool_features(stage_t.expression_grn, assignment_t)
    expression_tp = pool_features(stage_tp.expression_grn, assignment_tp)
    g_t_raw = project_grn_state(
        expression_t,
        stage_t.grn_adjacency,
        stage_t.projection_reg,
        stage_t.projection_tar,
    )
    g_tp_raw = project_grn_state(
        expression_tp,
        stage_tp.grn_adjacency,
        stage_tp.projection_reg,
        stage_tp.projection_tar,
    )
    g_t, g_tp = pairwise_zscore(g_t_raw, g_tp_raw)
    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        cci_t,
        cci_tp,
        components=nmf_components,
        max_iter=nmf_max_iter,
        seed=seed,
    )
    _, pij, v7_metadata = native_v7_pij_numpy(n_t, n_tp, g_t, g_tp)
    return {
        "EI_macro": float(effective_information(pij.copy())),
        "macro_cci_normalization": (
            "row_normalized_after_projection"
            if normalize_macro_cci
            else "raw_S_transpose_A_S"
        ),
        "macro_cci_shape_t": list(cci_t.shape),
        "macro_cci_shape_tp": list(cci_tp.shape),
        "macro_cci_nnz_t": int(cci_t.nnz),
        "macro_cci_nnz_tp": int(cci_tp.nnz),
        "N_metadata": n_metadata,
        "V7_metadata": v7_metadata,
    }


def strict_complete_combined_evaluation(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    pair: CompleteCombinedPair,
    assignment_t: np.ndarray,
    assignment_tp: np.ndarray,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
) -> dict[str, object]:
    pooled_n_t, pooled_n_tp = pairwise_zscore(
        pool_features(pair.n_t, assignment_t),
        pool_features(pair.n_tp, assignment_tp),
    )
    pooled_expression_t = pool_features(stage_t.expression_grn, assignment_t)
    pooled_expression_tp = pool_features(stage_tp.expression_grn, assignment_tp)
    recomputed_g_t, recomputed_g_tp = pairwise_zscore(
        project_grn_state(
            pooled_expression_t,
            stage_t.grn_adjacency,
            stage_t.projection_reg,
            stage_t.projection_tar,
        ),
        project_grn_state(
            pooled_expression_tp,
            stage_tp.grn_adjacency,
            stage_tp.projection_reg,
            stage_tp.projection_tar,
        ),
    )
    _, training_pij, training_metadata = native_v7_pij_numpy(
        pooled_n_t,
        pooled_n_tp,
        recomputed_g_t,
        recomputed_g_tp,
    )
    training_ei = float(effective_information(training_pij.copy()))
    exact_raw = _evaluate_exact_macro(
        stage_t,
        stage_tp,
        assignment_t,
        assignment_tp,
        nmf_components=nmf_components,
        nmf_max_iter=nmf_max_iter,
        seed=seed,
        normalize_macro_cci=False,
    )
    exact_rownorm = _evaluate_exact_macro(
        stage_t,
        stage_tp,
        assignment_t,
        assignment_tp,
        nmf_components=nmf_components,
        nmf_max_iter=nmf_max_iter,
        seed=seed,
        normalize_macro_cci=True,
    )
    micro_ei = float(pair.micro_ei)
    raw_ei = float(exact_raw["EI_macro"])
    rownorm_ei = float(exact_rownorm["EI_macro"])
    return {
        "evaluation_protocol": "complete_combined_coarse_native_v7",
        "EI_micro_native_v7": micro_ei,
        "EI_macro_training_interface_pool_N_recompute_G": training_ei,
        "deltaEI_training_interface_pool_N_recompute_G": training_ei - micro_ei,
        "EI_macro_strict_raw_projected_CCI_reextract_N_recompute_G": raw_ei,
        "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G": (
            raw_ei - micro_ei
        ),
        "EI_macro_strict_rownorm_projected_CCI_reextract_N_recompute_G": rownorm_ei,
        "deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G": (
            rownorm_ei - micro_ei
        ),
        "training_interface_metadata": training_metadata,
        "strict_raw_metadata": exact_raw,
        "strict_rownorm_metadata": exact_rownorm,
        "assignment_t": _assignment_stats(assignment_t),
        "assignment_tp": _assignment_stats(assignment_tp),
    }
