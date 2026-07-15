from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.loaders import natural_sort


@dataclass(frozen=True)
class PreparedGRN:
    units: list[str]
    genes: list[str]
    expression: np.ndarray
    adjacency: sp.csr_matrix
    metadata: dict[str, object]


@dataclass(frozen=True)
class GRNStateResult:
    projected: np.ndarray
    regulator_state: np.ndarray
    target_state: np.ndarray
    metadata: dict[str, object]


def prepare_grn_inputs(
    expression: pd.DataFrame,
    units: Sequence[str],
    grn_edges: pd.DataFrame,
    *,
    top_k_targets: int,
) -> PreparedGRN:
    if top_k_targets <= 0:
        raise ValueError("top_k_targets must be positive.")
    required = {"regulator", "target", "weight"}
    missing = required - set(grn_edges.columns)
    if missing:
        raise ValueError(f"GRN edge table is missing columns {sorted(missing)}.")

    units = list(map(str, units))
    expression_genes = set(map(str, expression.columns))
    work = grn_edges.loc[:, ["regulator", "target", "weight"]].copy()
    work["regulator"] = work["regulator"].astype(str)
    work["target"] = work["target"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").abs()
    raw_edge_count = int(len(work))
    work = work.dropna(subset=["weight"])
    work = work.loc[
        (work["weight"] > 0.0)
        & work["regulator"].isin(expression_genes)
        & work["target"].isin(expression_genes)
    ]
    work = (
        work.groupby(["regulator", "target"], as_index=False, sort=False)["weight"]
        .sum()
        .sort_values(
            ["regulator", "weight", "target"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    aligned_edge_count_before_topk = int(len(work))
    work = work.groupby("regulator", group_keys=False, sort=False).head(int(top_k_targets)).reset_index(drop=True)
    if work.empty:
        raise ValueError("GRN has no positive edges whose regulator and target are present in expression.")

    genes = natural_sort(set(work["regulator"]) | set(work["target"]))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    rows = work["regulator"].map(gene_index).to_numpy(dtype=int)
    cols = work["target"].map(gene_index).to_numpy(dtype=int)
    weights = work["weight"].to_numpy(dtype=float)
    adjacency = sp.coo_matrix(
        (weights, (rows, cols)),
        shape=(len(genes), len(genes)),
        dtype=float,
    ).tocsr()
    adjacency.sum_duplicates()
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0.0)
    adjacency = (sp.diags(inverse, format="csr") @ adjacency).tocsr()
    adjacency.sort_indices()

    missing_units = [unit for unit in units if unit not in expression.index]
    aligned_expression = expression.reindex(index=units, columns=genes, fill_value=0.0).to_numpy(dtype=float)
    aligned_expression = np.maximum(
        np.nan_to_num(aligned_expression, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    return PreparedGRN(
        units=units,
        genes=genes,
        expression=aligned_expression,
        adjacency=adjacency,
        metadata={
            "grn_weight_mode": "abs",
            "grn_normalization": "row_sum",
            "grn_topk_targets": int(top_k_targets),
            "raw_edge_count": raw_edge_count,
            "aligned_edge_count_before_topk": aligned_edge_count_before_topk,
            "retained_edge_count": int(adjacency.nnz),
            "gene_count": int(len(genes)),
            "unit_count": int(len(units)),
            "missing_expression_units": int(len(missing_units)),
            "expression_shape": list(aligned_expression.shape),
            "adjacency_shape": list(adjacency.shape),
            "adjacency_row_normalized": True,
        },
    )


def double_end_grn_state(
    expression: np.ndarray,
    adjacency: sp.spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(expression, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expression must be 2D, got shape {values.shape}.")
    matrix = adjacency.tocsr().astype(float)
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError(
            f"GRN adjacency shape {matrix.shape} does not match expression gene dimension {values.shape[1]}."
        )
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    regulator_program = np.asarray(matrix @ values.T, dtype=float).T
    target_program = np.asarray(matrix.T @ values.T, dtype=float).T
    return values * regulator_program, values * target_program


def deterministic_projection_matrix(
    genes: Sequence[str],
    *,
    role: str,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    if role not in {"reg", "tar"}:
        raise ValueError("role must be one of ['reg', 'tar'].")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")
    scale = 1.0 / np.sqrt(float(output_dim))
    rows: list[np.ndarray] = []
    for gene in map(str, genes):
        digest = hashlib.sha256(f"{int(seed)}\0{role}\0{gene}".encode("utf-8")).digest()
        row_seed = int.from_bytes(digest[:16], byteorder="little", signed=False)
        rows.append(np.random.default_rng(row_seed).standard_normal(output_dim) * scale)
    return np.vstack(rows) if rows else np.zeros((0, output_dim), dtype=float)


def build_projected_grn_state(
    prepared: PreparedGRN,
    *,
    output_dim: int,
    seed: int,
    gate_mode: str = "double_end",
) -> GRNStateResult:
    if gate_mode != "double_end":
        raise ValueError("gate_mode must be 'double_end'.")
    regulator_state, target_state = double_end_grn_state(prepared.expression, prepared.adjacency)
    regulator_projection = deterministic_projection_matrix(
        prepared.genes,
        role="reg",
        output_dim=output_dim,
        seed=seed,
    )
    target_projection = deterministic_projection_matrix(
        prepared.genes,
        role="tar",
        output_dim=output_dim,
        seed=seed,
    )
    projected = regulator_state @ regulator_projection + target_state @ target_projection
    projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    return GRNStateResult(
        projected=projected,
        regulator_state=regulator_state,
        target_state=target_state,
        metadata={
            **prepared.metadata,
            "grn_gate_mode": "double_end",
            "grn_state_definition": "[x*(W@x), x*(W.T@x)] projected by deterministic gene-role hashing",
            "grn_projection_method": "sha256_gene_role_seeded_gaussian",
            "grn_projection_seed": int(seed),
            "grn_state_dim": int(output_dim),
            "grn_state_shape": list(projected.shape),
        },
    )
