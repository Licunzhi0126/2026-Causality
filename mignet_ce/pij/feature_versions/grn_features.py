from __future__ import annotations

import numpy as np
import pandas as pd

from mignet_ce.networks.light_cci_grn import double_end_grn_state, deterministic_projection_matrix, prepare_grn_inputs
from mignet_ce.pij.feature_versions.sources import RawGRNInputs
from mignet_ce.pij.feature_versions.spec import FeatureRecipe


def transform_expression_new_version(expression: np.ndarray, mode: str = "log1p_gene_minmax") -> np.ndarray:
    values = np.asarray(expression, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expression must be 2D, got shape {values.shape}.")
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if mode == "nonnegative_only":
        return values
    if mode != "log1p_gene_minmax":
        raise ValueError("mode must be 'log1p_gene_minmax' or 'nonnegative_only'.")
    logged = np.log1p(values)
    minima = logged.min(axis=0, keepdims=True) if logged.shape[0] else np.zeros((1, logged.shape[1]))
    maxima = logged.max(axis=0, keepdims=True) if logged.shape[0] else np.zeros((1, logged.shape[1]))
    spans = maxima - minima
    return np.divide(logged - minima, spans, out=np.zeros_like(logged), where=spans > 0.0)


def build_split_grn_features(
    raw: RawGRNInputs,
    recipe: FeatureRecipe,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray]]:
    selected = raw.expression.reindex(index=list(raw.units), fill_value=0.0)
    transformed = transform_expression_new_version(selected.to_numpy(dtype=float), mode="log1p_gene_minmax")
    transformed_frame = pd.DataFrame(transformed, index=selected.index, columns=selected.columns)
    prepared = prepare_grn_inputs(
        transformed_frame,
        raw.units,
        raw.grn_edges,
        top_k_targets=recipe.grn_topk_targets,
    )
    regulator_state, target_state = double_end_grn_state(prepared.expression, prepared.adjacency)
    q_reg = deterministic_projection_matrix(
        prepared.genes,
        role="reg",
        output_dim=recipe.projection_dim,
        seed=recipe.projection_seed,
    )
    q_tar = deterministic_projection_matrix(
        prepared.genes,
        role="tar",
        output_dim=recipe.projection_dim,
        seed=recipe.projection_seed,
    )
    g_reg = np.nan_to_num(regulator_state @ q_reg, nan=0.0, posinf=0.0, neginf=0.0)
    g_tar = np.nan_to_num(target_state @ q_tar, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = {
        "feature_source": "raw_expression_and_grn_edges_recomputed",
        "expression_transform": "nonnegative_log1p_per_timepoint_per_gene_minmax_constant_to_zero",
        "h5ad_path": str(raw.h5ad_path),
        "grn_path": str(raw.grn_path),
        "prepared_grn": prepared.metadata,
        "regulator_state_shape": list(regulator_state.shape),
        "target_state_shape": list(target_state.shape),
        "g_reg_shape": list(g_reg.shape),
        "g_tar_shape": list(g_tar.shape),
        "projection_dim": int(recipe.projection_dim),
        "projection_seed": int(recipe.projection_seed),
        "regulator_target_summed": False,
    }
    artifacts = {"Q_reg": q_reg, "Q_tar": q_tar}
    return {"g_reg": g_reg, "g_tar": g_tar}, metadata, artifacts
