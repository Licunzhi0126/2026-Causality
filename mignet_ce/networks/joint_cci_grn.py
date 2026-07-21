from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import natural_sort
from mignet_ce.networks.light_cci_grn import LightCCIGRNNetworkBuilder


@dataclass(frozen=True)
class JointCCIGRNPairResult:
    source_features: np.ndarray
    target_features: np.ndarray
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


def _clean_nonnegative(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {values.shape}.")
    return np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _stabilize(values: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(np.nan_to_num(values, nan=eps, posinf=eps, neginf=eps), eps)


def _relative_squared_error(observed: np.ndarray, reconstructed: np.ndarray, eps: float) -> float:
    numerator = float(np.sum((observed - reconstructed) ** 2, dtype=np.float64))
    denominator = float(np.sum(observed * observed, dtype=np.float64)) + eps
    return numerator / denominator


def module_double_end_gate(
    expression: np.ndarray,
    q_regulator: np.ndarray,
    q_target: np.ndarray,
    core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = _clean_nonnegative(expression, name="expression")
    q_reg = _clean_nonnegative(q_regulator, name="q_regulator")
    q_tar = _clean_nonnegative(q_target, name="q_target")
    s_grn = _clean_nonnegative(core, name="core")
    if values.shape[1] != q_reg.shape[0] or values.shape[1] != q_tar.shape[0]:
        raise ValueError("Expression genes must match Q_regulator and Q_target rows.")
    if q_reg.shape[1] != s_grn.shape[0] or q_tar.shape[1] != s_grn.shape[1]:
        raise ValueError("GRN core dimensions must match regulator and target module counts.")
    a_reg = values @ q_reg
    a_tar = values @ q_tar
    g_reg = a_reg * (a_tar @ s_grn.T)
    g_tar = a_tar * (a_reg @ s_grn)
    return a_reg, a_tar, g_reg, g_tar


def collective_joint_nmf_pair(
    a_source: np.ndarray,
    a_target: np.ndarray,
    g_reg_source: np.ndarray,
    g_tar_source: np.ndarray,
    g_reg_target: np.ndarray,
    g_tar_target: np.ndarray,
    *,
    rank: int,
    lambda_cci: float,
    lambda_grn: float,
    max_iter: int,
    seed: int,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")
    if lambda_cci < 0 or lambda_grn < 0 or lambda_cci + lambda_grn <= 0:
        raise ValueError("lambda_cci and lambda_grn must be nonnegative with a positive sum.")
    eps = max(float(eps), 1e-12)
    cci = [
        _clean_nonnegative(a_source, name="a_source"),
        _clean_nonnegative(a_target, name="a_target"),
    ]
    g_reg = [
        _clean_nonnegative(g_reg_source, name="g_reg_source"),
        _clean_nonnegative(g_reg_target, name="g_reg_target"),
    ]
    g_tar = [
        _clean_nonnegative(g_tar_source, name="g_tar_source"),
        _clean_nonnegative(g_tar_target, name="g_tar_target"),
    ]
    for index, adjacency in enumerate(cci):
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"CCI matrix {index} must be square, got {adjacency.shape}.")
        if g_reg[index].shape[0] != adjacency.shape[0] or g_tar[index].shape[0] != adjacency.shape[0]:
            raise ValueError("CCI and GRN unit-state row counts must match.")
    grn_rank = g_reg[0].shape[1]
    if any(matrix.shape[1] != grn_rank for matrix in [*g_reg, *g_tar]):
        raise ValueError("All GRN module-state matrices must have the same column count.")

    cci_weights = [
        float(lambda_cci) / (float(np.sum(matrix * matrix, dtype=np.float64)) + eps)
        for matrix in cci
    ]
    reg_weights = [
        float(lambda_grn) / (float(np.sum(matrix * matrix, dtype=np.float64)) + eps)
        for matrix in g_reg
    ]
    tar_weights = [
        float(lambda_grn) / (float(np.sum(matrix * matrix, dtype=np.float64)) + eps)
        for matrix in g_tar
    ]
    rng = np.random.default_rng(seed)
    h_send = [rng.random((matrix.shape[0], rank)) + eps for matrix in cci]
    h_recv = [rng.random((matrix.shape[0], rank)) + eps for matrix in cci]
    s_cci = rng.random((rank, rank)) + eps
    b_reg = rng.random((rank, grn_rank)) + eps
    b_tar = rng.random((rank, grn_rank)) + eps

    for _ in range(max_iter):
        for index in range(2):
            hs = h_send[index]
            hr = h_recv[index]
            numerator_send = (
                cci_weights[index] * (cci[index] @ hr @ s_cci.T)
                + reg_weights[index] * (g_reg[index] @ b_reg.T)
            )
            denominator_send = (
                cci_weights[index] * (hs @ s_cci @ (hr.T @ hr) @ s_cci.T)
                + reg_weights[index] * (hs @ b_reg @ b_reg.T)
                + eps
            )
            h_send[index] = _stabilize(hs * numerator_send / denominator_send, eps)

            hs = h_send[index]
            numerator_recv = (
                cci_weights[index] * (cci[index].T @ hs @ s_cci)
                + tar_weights[index] * (g_tar[index] @ b_tar.T)
            )
            denominator_recv = (
                cci_weights[index] * (hr @ s_cci.T @ (hs.T @ hs) @ s_cci)
                + tar_weights[index] * (hr @ b_tar @ b_tar.T)
                + eps
            )
            h_recv[index] = _stabilize(hr * numerator_recv / denominator_recv, eps)

        numerator_s = sum(
            cci_weights[index] * (h_send[index].T @ cci[index] @ h_recv[index])
            for index in range(2)
        )
        denominator_s = sum(
            cci_weights[index]
            * ((h_send[index].T @ h_send[index]) @ s_cci @ (h_recv[index].T @ h_recv[index]))
            for index in range(2)
        ) + eps
        s_cci = _stabilize(s_cci * numerator_s / denominator_s, eps)

        numerator_b_reg = sum(
            reg_weights[index] * (h_send[index].T @ g_reg[index])
            for index in range(2)
        )
        denominator_b_reg = sum(
            reg_weights[index] * ((h_send[index].T @ h_send[index]) @ b_reg)
            for index in range(2)
        ) + eps
        b_reg = _stabilize(b_reg * numerator_b_reg / denominator_b_reg, eps)

        numerator_b_tar = sum(
            tar_weights[index] * (h_recv[index].T @ g_tar[index])
            for index in range(2)
        )
        denominator_b_tar = sum(
            tar_weights[index] * ((h_recv[index].T @ h_recv[index]) @ b_tar)
            for index in range(2)
        ) + eps
        b_tar = _stabilize(b_tar * numerator_b_tar / denominator_b_tar, eps)

    loss_rows: list[dict[str, float | str]] = []
    weighted_objective = 0.0
    for label, index in (("source", 0), ("target", 1)):
        cci_loss = _relative_squared_error(
            cci[index],
            h_send[index] @ s_cci @ h_recv[index].T,
            eps,
        )
        reg_loss = _relative_squared_error(g_reg[index], h_send[index] @ b_reg, eps)
        tar_loss = _relative_squared_error(g_tar[index], h_recv[index] @ b_tar, eps)
        weighted_objective += lambda_cci * cci_loss + lambda_grn * (reg_loss + tar_loss)
        loss_rows.append(
            {
                "time_role": label,
                "cci_relative_loss": cci_loss,
                "grn_reg_relative_loss": reg_loss,
                "grn_tar_relative_loss": tar_loss,
            }
        )
    source_features = np.hstack([h_send[0], h_recv[0]])
    target_features = np.hstack([h_send[1], h_recv[1]])
    factors = {
        "S_C": s_cci,
        "B_reg": b_reg,
        "B_tar": b_tar,
        "H_send_source": h_send[0],
        "H_recv_source": h_recv[0],
        "H_send_target": h_send[1],
        "H_recv_target": h_recv[1],
    }
    diagnostics: dict[str, object] = {
        "rank": int(rank),
        "max_iter": int(max_iter),
        "seed": int(seed),
        "lambda_cci": float(lambda_cci),
        "lambda_grn": float(lambda_grn),
        "losses": loss_rows,
        "weighted_objective": float(weighted_objective),
        "all_finite": bool(
            all(np.all(np.isfinite(value)) for value in [source_features, target_features, *factors.values()])
        ),
    }
    return source_features, target_features, factors, diagnostics


def _aligned_graph_grn_pair(
    source_graph: LayerGraph,
    target_graph: LayerGraph,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_genes = list(map(str, source_graph.metadata.get("grn_genes", [])))
    target_genes = list(map(str, target_graph.metadata.get("grn_genes", [])))
    genes = natural_sort(set(source_genes) & set(target_genes))
    if not genes:
        raise ValueError("joint_cci_grn has no shared GRN genes across the time pair.")
    source_lookup = {gene: index for index, gene in enumerate(source_genes)}
    target_lookup = {gene: index for index, gene in enumerate(target_genes)}
    source_indices = [source_lookup[gene] for gene in genes]
    target_indices = [target_lookup[gene] for gene in genes]
    source_adjacency = source_graph.metadata.get("grn_adjacency_csr")
    target_adjacency = target_graph.metadata.get("grn_adjacency_csr")
    source_expression = source_graph.metadata.get("grn_expression_csr")
    target_expression = target_graph.metadata.get("grn_expression_csr")
    if any(value is None for value in (source_adjacency, target_adjacency, source_expression, target_expression)):
        raise ValueError("joint_cci_grn graph metadata is missing GRN adjacency or expression payloads.")
    source_w = sp.csr_matrix(source_adjacency)[source_indices, :][:, source_indices].toarray()
    target_w = sp.csr_matrix(target_adjacency)[target_indices, :][:, target_indices].toarray()
    source_x = sp.csr_matrix(source_expression)[:, source_indices].toarray()
    target_x = sp.csr_matrix(target_expression)[:, target_indices].toarray()
    return genes, source_w, target_w, source_x, target_x


def build_joint_cci_grn_pair(
    source_cci: sp.spmatrix,
    target_cci: sp.spmatrix,
    source_graph: LayerGraph,
    target_graph: LayerGraph,
    *,
    grn_rank: int,
    cci_rank: int,
    lambda_cci: float,
    lambda_grn: float,
    max_iter: int,
    seed: int,
) -> JointCCIGRNPairResult:
    # Import lazily so importing mignet_ce.metrics can initialize the PIJ and
    # network packages without cycling back into this module.
    from mignet_ce.metrics import pairwise_shared_core_directed_nmf

    genes, source_w, target_w, source_x, target_x = _aligned_graph_grn_pair(source_graph, target_graph)
    q_reg_source, q_tar_source, q_reg_target, q_tar_target, s_grn = pairwise_shared_core_directed_nmf(
        source_w,
        target_w,
        n_components=grn_rank,
        max_iter=max_iter,
        seed=seed,
    )
    a_reg_source, a_tar_source, g_reg_source, g_tar_source = module_double_end_gate(
        source_x,
        q_reg_source,
        q_tar_source,
        s_grn,
    )
    a_reg_target, a_tar_target, g_reg_target, g_tar_target = module_double_end_gate(
        target_x,
        q_reg_target,
        q_tar_target,
        s_grn,
    )
    source_features, target_features, collective_factors, collective_diagnostics = collective_joint_nmf_pair(
        source_cci.toarray(),
        target_cci.toarray(),
        g_reg_source,
        g_tar_source,
        g_reg_target,
        g_tar_target,
        rank=cci_rank,
        lambda_cci=lambda_cci,
        lambda_grn=lambda_grn,
        max_iter=max_iter,
        seed=seed + 7919,
    )
    grn_source_loss = _relative_squared_error(source_w, q_reg_source @ s_grn @ q_tar_source.T, 1e-10)
    grn_target_loss = _relative_squared_error(target_w, q_reg_target @ s_grn @ q_tar_target.T, 1e-10)
    artifacts: dict[str, object] = {
        "model_type": "collective_joint_cci_grn",
        "genes": genes,
        "Q_reg_source": q_reg_source,
        "Q_tar_source": q_tar_source,
        "Q_reg_target": q_reg_target,
        "Q_tar_target": q_tar_target,
        "S_G": s_grn,
        "A_reg_source": a_reg_source,
        "A_tar_source": a_tar_source,
        "A_reg_target": a_reg_target,
        "A_tar_target": a_tar_target,
        "G_reg_source": g_reg_source,
        "G_tar_source": g_tar_source,
        "G_reg_target": g_reg_target,
        "G_tar_target": g_tar_target,
        **collective_factors,
        "features_source": source_features,
        "features_target": target_features,
        "feature_definition": "concat(H_send, H_recv)",
    }
    diagnostics = {
        "grn_gene_count": int(len(genes)),
        "grn_rank": int(grn_rank),
        "cci_rank": int(cci_rank),
        "grn_source_relative_loss": float(grn_source_loss),
        "grn_target_relative_loss": float(grn_target_loss),
        "shared_grn_core": True,
        "shared_cci_core": True,
        "module_gate_mode": "double_end",
        "collective": collective_diagnostics,
    }
    return JointCCIGRNPairResult(
        source_features=source_features,
        target_features=target_features,
        artifacts=artifacts,
        diagnostics=diagnostics,
    )


class JointCCIGRNNetworkBuilder(LightCCIGRNNetworkBuilder):
    network_method = "joint_cci_grn"
    grn_integration = "directed_grn_joint_nmf_expression_bridge_collective_cci_grn_nmf"
    build_projected_state = False
    retain_joint_inputs = True
