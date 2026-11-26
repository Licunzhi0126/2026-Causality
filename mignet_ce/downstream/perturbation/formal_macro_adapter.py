"""Perturbation-owned differentiable Pool(X) GRN adapter."""
from __future__ import annotations
import numpy as np
import torch
from mignet_ce.pij.compare._shared.ng_kl_ot import (
    CANONICAL_ALPHA_CCI, CANONICAL_FEATURE_BETA_N, CANONICAL_FEATURE_BETA_G,
    CANONICAL_TEMPERATURE, canonical_ng_pij_torch,
)

PERTURBATION_MACRO_SCHEMA = "formal_pool_X_plus_propagated_DG_v2"


def validate_stage_coordinates(reference, candidate):
    for name in ("units", "grn_genes", "expression_grn", "projection_reg", "projection_tar"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ValueError(f"GRN coordinate mismatch: {name}; cannot subtract or propagate states.")
    if reference.g_raw.shape != candidate.g_raw.shape:
        raise ValueError("GRN state dimensions differ.")


def paired_zscore(source, target):
    values = torch.cat([source, target], dim=0)
    std = values.std(dim=0, unbiased=False, keepdim=True)
    result = (values - values.mean(dim=0, keepdim=True)) / torch.where(std < 1e-8, torch.ones_like(std), std)
    return result[:source.shape[0]], result[source.shape[0]:]


def make_projector():
    cache = {}
    def project(expression, stage, label):
        values = torch.clamp(torch.nan_to_num(expression, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        key = (label, str(values.device))
        if key not in cache:
            coo = stage.grn_adjacency.tocoo()
            adjacency = torch.sparse_coo_tensor(
                torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=values.device),
                torch.tensor(coo.data, dtype=torch.float32, device=values.device),
                size=coo.shape, dtype=torch.float32, device=values.device, check_invariants=True).coalesce()
            cache[key] = (adjacency,
                torch.tensor(stage.projection_reg, dtype=torch.float32, device=values.device),
                torch.tensor(stage.projection_tar, dtype=torch.float32, device=values.device))
        adjacency, reg, tar = cache[key]
        regulator = torch.sparse.mm(adjacency, values.T).T
        target = torch.sparse.mm(adjacency.transpose(0, 1), values.T).T
        return (values * regulator) @ reg + (values * target) @ tar
    return project


def build_direct_state_macro_pij_builder(stage_t, stage_tp, *, delta_g_raw_tp=None,
                                         alpha_cci=CANONICAL_ALPHA_CCI, feature_method="legacy", grn_enhancer=None):
    if feature_method == "legacy_nmf":
        from wyt_deltaei_coarse_grain.complete_combined import build_macro_pij_builder
        if grn_enhancer is None:
            raise ValueError("legacy_nmf requires the fitted baseline GRN basis")
        return build_macro_pij_builder(stage_t, stage_tp, grn_enhancer=grn_enhancer,
                                       alpha_cci=alpha_cci,
                                       target_residual_block="DG_TP" if delta_g_raw_tp is not None else None)
    if feature_method != "legacy":
        raise NotImplementedError("Propagated GRN residuals currently require legacy GRN features.")
    if delta_g_raw_tp is None:
        from wyt_deltaei_coarse_grain.complete_combined import build_macro_pij_builder
        return build_macro_pij_builder(stage_t, stage_tp, alpha_cci=alpha_cci)
    project = make_projector()
    def build(inputs):
        n_t, n_tp = paired_zscore(inputs.feature_blocks_t["N"], inputs.feature_blocks_tp["N"])
        raw_t = project(inputs.feature_blocks_t["X"], stage_t, "t")
        raw_tp = project(inputs.feature_blocks_tp["X"], stage_tp, "tp")
        # The trainer pools the already propagated Spot residual with its learned
        # target assignment. The pooled expression projection uses the baseline T2 graph.
        residual = inputs.feature_blocks_tp["DG_TP"]
        if residual.shape != raw_tp.shape:
            raise ValueError("Pooled DG_TP must match raw target GRN coordinates.")
        g_t, g_tp = paired_zscore(raw_t, raw_tp + residual)
        return canonical_ng_pij_torch(n_t, n_tp, g_t, g_tp,
            beta_n=CANONICAL_FEATURE_BETA_N, beta_g=CANONICAL_FEATURE_BETA_G,
            alpha_cci=alpha_cci, temperature=CANONICAL_TEMPERATURE)
    return build
