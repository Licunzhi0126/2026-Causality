"""Direct macro PIJ adapter using the repository's canonical NG_KLot implementation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse


def build_direct_macro_pij(*, h5ad_t: Path, h5ad_tp: Path, cci_t: Path, cci_tp: Path, index_t: Path, index_tp: Path, grn_t: Path, grn_tp: Path, nmf_components: int = 5, nmf_max_iter: int = 300, seed: int = 42) -> dict[str, object]:
    """Prepare existing complete stages and delegate all PIJ/EI mathematics."""
    from mignet_ce.io.loaders import read_commot_index
    from mignet_ce.downstream.analysis.dynamic_closure.optimal import prepare_ngklot_pair
    from wyt_deltaei_coarse_grain.complete_combined import prepare_complete_stage

    units_t, units_tp = read_commot_index(index_t), read_commot_index(index_tp)
    matrix_t, matrix_tp = sparse.load_npz(cci_t).tocsr(), sparse.load_npz(cci_tp).tocsr()
    if matrix_t.shape != (len(units_t), len(units_t)) or matrix_tp.shape != (len(units_tp), len(units_tp)):
        raise ValueError("Macro CCI matrix dimensions do not match their index files.")
    stage_t = prepare_complete_stage(h5ad_path=h5ad_t, grn_path=grn_t, units=units_t, cci=matrix_t, top_k_targets=50, state_dim=64, projection_seed=20260713)
    stage_tp = prepare_complete_stage(h5ad_path=h5ad_tp, grn_path=grn_tp, units=units_tp, cci=matrix_tp, top_k_targets=50, state_dim=64, projection_seed=20260713)
    pair = prepare_ngklot_pair(stage_t, stage_tp, nmf_components=nmf_components, nmf_max_iter=nmf_max_iter, seed=seed)
    return {"pij": pair.micro_pij, "ei": pair.micro_ei, "source_units": units_t, "target_units": units_tp, "metadata": pair.canonical_ng_metadata}
