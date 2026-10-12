"""Build T1/T2 condition-specific GRN spot states with existing project primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest, load_spot_pair
from wyt_deltaei_coarse_grain.complete_combined import prepare_complete_stage

from .propagation_bridge import propagate_delta


@dataclass(frozen=True)
class PerturbedGRNStates:
    spot_ids_t: tuple[str, ...]
    spot_ids_tp: tuple[str, ...]
    g_raw_t_baseline: np.ndarray
    g_raw_tp_baseline: np.ndarray
    g_raw_t_perturbed: np.ndarray
    g_raw_tp_perturbed: np.ndarray
    delta_g_raw_t: np.ndarray
    delta_g_raw_tp: np.ndarray


def build_baseline_grn_states(*, baseline_inputs: dict[str, Path], state_dim: int = 64, projection_seed: int = 20260713) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned baseline spot GRN states for the direct-state reference."""
    request = CoarseFrontendRequest(
        h5ad_t=baseline_inputs["h5ad_t"], h5ad_tp=baseline_inputs["h5ad_tp"],
        cci_t=baseline_inputs["cci_t"], cci_tp=baseline_inputs["cci_tp"],
        cci_index_t=baseline_inputs["index_t"], cci_index_tp=baseline_inputs["index_tp"],
    )
    pair_data = load_spot_pair(request)
    stage_t = prepare_complete_stage(h5ad_path=baseline_inputs["h5ad_t"], grn_path=baseline_inputs["grn_t"], units=pair_data.unit_ids_t, cci=pair_data.cci_t, top_k_targets=50, state_dim=state_dim, projection_seed=projection_seed)
    stage_tp = prepare_complete_stage(h5ad_path=baseline_inputs["h5ad_tp"], grn_path=baseline_inputs["grn_tp"], units=pair_data.unit_ids_tp, cci=pair_data.cci_tp, top_k_targets=50, state_dim=state_dim, projection_seed=projection_seed)
    return stage_t.g_raw, stage_tp.g_raw


def build_propagated_grn_states(*, baseline_inputs: dict[str, Path], perturbed_grn_t: Path, baseline_spot_pij: Path, state_dim: int = 64, projection_seed: int = 20260713) -> PerturbedGRNStates:
    """Build T1/T2 perturbation states without reconstructing expression."""
    request = CoarseFrontendRequest(
        h5ad_t=baseline_inputs["h5ad_t"], h5ad_tp=baseline_inputs["h5ad_tp"],
        cci_t=baseline_inputs["cci_t"], cci_tp=baseline_inputs["cci_tp"],
        cci_index_t=baseline_inputs["index_t"], cci_index_tp=baseline_inputs["index_tp"],
    )
    pair_data = load_spot_pair(request)
    stage_t = prepare_complete_stage(h5ad_path=baseline_inputs["h5ad_t"], grn_path=baseline_inputs["grn_t"], units=pair_data.unit_ids_t, cci=pair_data.cci_t, top_k_targets=50, state_dim=state_dim, projection_seed=projection_seed)
    stage_t_star = prepare_complete_stage(h5ad_path=baseline_inputs["h5ad_t"], grn_path=perturbed_grn_t, units=pair_data.unit_ids_t, cci=pair_data.cci_t, top_k_targets=50, state_dim=state_dim, projection_seed=projection_seed)
    stage_tp = prepare_complete_stage(h5ad_path=baseline_inputs["h5ad_tp"], grn_path=baseline_inputs["grn_tp"], units=pair_data.unit_ids_tp, cci=pair_data.cci_tp, top_k_targets=50, state_dim=state_dim, projection_seed=projection_seed)
    delta_raw_t = stage_t_star.g_raw - stage_t.g_raw
    propagation = np.load(baseline_spot_pij)
    delta_raw_tp = propagate_delta(delta_raw_t, propagation)
    g_raw_tp_perturbed = stage_tp.g_raw + delta_raw_tp
    return PerturbedGRNStates(
        spot_ids_t=tuple(stage_t.units), spot_ids_tp=tuple(stage_tp.units),
        g_raw_t_baseline=stage_t.g_raw, g_raw_tp_baseline=stage_tp.g_raw,
        g_raw_t_perturbed=stage_t_star.g_raw,
        g_raw_tp_perturbed=g_raw_tp_perturbed,
        delta_g_raw_t=delta_raw_t, delta_g_raw_tp=delta_raw_tp,
    )
