"""Formal frozen Spot ↔ Maturity Macro GRN perturbation analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass

from mignet_ce.metrics import effective_information
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_numpy
from wyt_deltaei_coarse_grain.complete_combined import (
    pairwise_zscore, pool_features, prepare_complete_pair,
    project_grn_state,
)
from ..dynamic_closure.analysis import information_closure_budget, js_rows
from ..preparation import _load_complete_stage
from .baseline import DEFAULT_METHOD
from .model_artifact import load_baseline
from .operators import METHODS, apply_operator, derived_seed, fixed_targets
from .propagation import horizontal, path_metrics
from .validation import require_finite

def _closure(p, s_t, s_tp, q):
    observed = p @ s_tp
    predicted = s_t @ q
    budget = information_closure_budget(p, s_t, s_tp)
    return float(np.mean(js_rows(observed, predicted))), budget

@dataclass(frozen=True)
class PairContext:
    pair: str
    baseline: object
    stage_t: object
    stage_tp: object
    pair_state: object


def _context(cfg, baseline_root, pair: str, frontend_method: str) -> PairContext:
    baseline = load_baseline(baseline_root, pair, frontend_method)
    source, target = pair.split("->")
    stage_t, stage_tp = _load_complete_stage(cfg, "spot", source), _load_complete_stage(cfg, "spot", target)
    record = baseline.record
    if tuple(stage_t.units) != record.source_ids or tuple(stage_tp.units) != record.target_ids:
        raise ValueError(f"Baseline unit IDs do not match current source inputs for {pair}")
    pair_state = prepare_complete_pair(stage_t, stage_tp, nmf_components=cfg.profile.nmf_components, nmf_max_iter=cfg.profile.nmf_max_iter, seed=cfg.profile.random_seed, mid_dim=64)
    return PairContext(pair, baseline, stage_t, stage_tp, pair_state)


def _one(cfg, context: PairContext, frontend_method: str, operator_method: str, strength: float, repeat: int) -> dict[str, object]:
    pair, baseline, stage_t, stage_tp, pair_state = context.pair, context.baseline, context.stage_t, context.stage_tp, context.pair_state
    record = baseline.record
    source, target = pair.split("->")
    seed = derived_seed(pair, f"{frontend_method}:{operator_method}", strength, repeat, cfg.profile.random_seed)
    a0 = stage_t.grn_adjacency.toarray()
    perturbed, operator = apply_operator(a0, operator_method, strength, seed, fixed_targets(a0))
    perturbed_sparse = sp.csr_matrix(perturbed)
    g_source_star_raw = project_grn_state(stage_t.expression_grn, perturbed_sparse, stage_t.projection_reg, stage_t.projection_tar)
    g_source, g_target = pairwise_zscore(stage_t.g_raw, stage_tp.g_raw)
    g_source_star, _ = pairwise_zscore(g_source_star_raw, stage_tp.g_raw)
    delta_spot_t = g_source_star - g_source
    delta_spot_tp = horizontal(delta_spot_t, record.p)
    p_star = canonical_ng_pij_numpy(pair_state.n_t, pair_state.n_tp, g_source_star, g_target + delta_spot_tp)[1]
    # Strict macro response: pool expression first, then recompute with the perturbed GRN.
    x_macro_t, x_macro_tp = pool_features(stage_t.expression_grn, record.source_assignment), pool_features(stage_tp.expression_grn, record.target_assignment)
    g_macro_t0_raw = project_grn_state(x_macro_t, stage_t.grn_adjacency, stage_t.projection_reg, stage_t.projection_tar)
    g_macro_tp0_raw = project_grn_state(x_macro_tp, stage_tp.grn_adjacency, stage_tp.projection_reg, stage_tp.projection_tar)
    g_macro_t0, g_macro_tp0 = pairwise_zscore(g_macro_t0_raw, g_macro_tp0_raw)
    g_macro_t_star, _ = pairwise_zscore(project_grn_state(x_macro_t, perturbed_sparse, stage_t.projection_reg, stage_t.projection_tar), g_macro_tp0_raw)
    delta_macro_t = g_macro_t_star - g_macro_t0
    delta_macro_hv, delta_macro_vh = pool_features(delta_spot_tp, record.target_assignment), record.q_direct.T @ delta_macro_t
    n_macro_t, n_macro_tp = pairwise_zscore(pool_features(pair_state.n_t, record.source_assignment), pool_features(pair_state.n_tp, record.target_assignment))
    q_hv = canonical_ng_pij_numpy(n_macro_t, n_macro_tp, g_macro_t_star, g_macro_tp0 + delta_macro_hv)[1]
    q_vh = canonical_ng_pij_numpy(n_macro_t, n_macro_tp, g_macro_t_star, g_macro_tp0 + delta_macro_vh)[1]
    ei_spot0, ei_spot = effective_information(record.p.copy()), effective_information(p_star.copy())
    ei_macro0, ei_hv, ei_vh = effective_information(record.q_direct.copy()), effective_information(q_hv.copy()), effective_information(q_vh.copy())
    closure0, budget0 = _closure(record.p, record.source_assignment, record.target_assignment, record.q_direct)
    closure_hv, budget_hv = _closure(p_star, record.source_assignment, record.target_assignment, q_hv)
    closure_vh, _ = _closure(p_star, record.source_assignment, record.target_assignment, q_vh)
    keff_t = float(np.exp(-np.sum(record.source_assignment.mean(0) * np.log(np.maximum(record.source_assignment.mean(0), 1e-12)))))
    keff_tp = float(np.exp(-np.sum(record.target_assignment.mean(0) * np.log(np.maximum(record.target_assignment.mean(0), 1e-12)))))
    return {
        "run_id": f"{pair}:{operator_method}:{strength:.3f}:{repeat}", "organ": cfg.organ,
        "t1": source, "t2": target, "time_pair": pair, "method": operator_method,
        "strength": strength, "repeat": repeat, "seed": seed, "frontend_method": frontend_method,
        "K_requested": record.source_assignment.shape[1], "Keff_t": keff_t, "Keff_tp": keff_tp,
        "S_t_checksum": baseline.checksums["S_t.npy"], "S_tp_checksum": baseline.checksums["S_tp.npy"],
        "baseline_spot_pij_checksum": baseline.checksums["PIJ_micro_train.npy"],
        "baseline_macro_pij_checksum": baseline.checksums["PIJ_macro_train.npy"],
        "checkpoint_name": baseline.checkpoint_name,
        "checkpoint_checksum": baseline.checksums[baseline.checkpoint_name], **operator,
        "grn_edge_count_baseline": int(np.count_nonzero(a0)), "grn_edge_count_perturbed": int(np.count_nonzero(perturbed)),
        "grn_delta_nnz": int(np.count_nonzero(perturbed - a0)),
        "spot_state_delta_l1_t": float(np.abs(delta_spot_t).sum()), "spot_state_delta_l1_tp": float(np.abs(delta_spot_tp).sum()),
        "spot_pij_l1": float(np.abs(p_star-record.p).sum()), "spot_pij_frobenius": float(np.linalg.norm(p_star-record.p)),
        "EI_spot_baseline": ei_spot0, "EI_spot_perturbed": ei_spot, "delta_EI_spot": ei_spot-ei_spot0,
        "macro_state_delta_l1_t": float(np.abs(delta_macro_t).sum()), "macro_state_delta_l1_tp_HV": float(np.abs(delta_macro_hv).sum()),
        "macro_state_delta_l1_tp_VH": float(np.abs(delta_macro_vh).sum()), "macro_pij_l1_HV": float(np.abs(q_hv-record.q_direct).sum()),
        "macro_pij_l1_VH": float(np.abs(q_vh-record.q_direct).sum()), "EI_macro_baseline": ei_macro0,
        "EI_macro_perturbed_HV": ei_hv, "EI_macro_perturbed_VH": ei_vh, **path_metrics(delta_macro_hv, delta_macro_vh),
        "CE_baseline": ei_macro0-ei_spot0, "CE_perturbed_HV": ei_hv-ei_spot, "CE_perturbed_VH": ei_vh-ei_spot,
        "delta_CE_HV": (ei_hv-ei_spot)-(ei_macro0-ei_spot0), "delta_CE_VH": (ei_vh-ei_spot)-(ei_macro0-ei_spot0),
        "closure_js_baseline": closure0, "closure_js_perturbed_HV": closure_hv, "closure_js_perturbed_VH": closure_vh,
        "delta_closure_js_HV": closure_hv-closure0, "delta_closure_js_VH": closure_vh-closure0,
        "I_available_baseline": budget0["I_available_bits"], "I_retained_baseline": budget0["I_macro_retained_bits"],
        "closure_quality_baseline": budget0["closure_quality"], "I_available_perturbed": budget_hv["I_available_bits"],
        "I_retained_perturbed": budget_hv["I_macro_retained_bits"], "closure_quality_perturbed": budget_hv["closure_quality"],
        "delta_closure_quality": budget_hv["closure_quality"]-budget0["closure_quality"],
    }

def build_grn_perturbation_table(
    cfg,
    baseline_root,
    *,
    frontend_method: str = DEFAULT_METHOD,
    strengths=tuple(np.arange(.1, 1., .1)),
    repeats: int = 20,
) -> pd.DataFrame:
    contexts = {
        pair: _context(cfg, baseline_root, pair, frontend_method)
        for pair in cfg.adjacent_pairs
    }
    rows = [
        _one(cfg, contexts[pair], frontend_method, method, float(strength), repeat)
        for pair in cfg.adjacent_pairs
        for method in METHODS
        for strength in strengths
        for repeat in range(repeats)
    ]
    frame = pd.DataFrame(rows)
    require_finite(frame)
    return frame
