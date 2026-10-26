"""Adapter that reuses the authoritative two-stage trainer for one local pair."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.coarse_frontends import prepare_coarse_input
from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.metrics import effective_information
from wyt_deltaei_coarse_grain import WYTTwoStageDeltaEIConfig, train_deltaei

from .local_pair_builder import LocalPairData
from .reporting import collect_summary


GLOBAL_SEED = 42


def choose_local_k(k_global: int, n_source: int, n_target: int) -> int | None:
    """Return Plan 6's feasible local K, or None when fewer than two states exist."""
    n_min = min(int(n_source), int(n_target))
    if k_global < 2:
        raise ValueError("k_global must be at least two.")
    if n_min <= 2:
        return None
    return min(int(k_global), max(2, int(np.floor(np.sqrt(n_min)))), n_min - 1)


def pair_seed(global_seed: int, source_cluster: int, target_cluster: int) -> int:
    return int(global_seed) + int(source_cluster) * 1000 + int(target_cluster)


def run_local_optimal_cg(*, pair: LocalPairData, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, k_global: int, device: str, pair_root: Path, global_seed: int = GLOBAL_SEED) -> dict[str, object]:
    """Prepare local CCI/maturity assets and train using the existing formal core.

    The frontend reconstructs all local CCI/GRN/expression representations from
    the selected spot IDs.  Only the micro PIJ is overridden with Plan 6's
    conditional submatrix, which is the explicitly defined local dynamics.
    """
    if pair.status != "ready":
        raise ValueError(f"Cannot train a pair with status {pair.status!r}.")
    k_local = choose_local_k(k_global, pair.n_source_used, pair.n_target)
    if k_local is None:
        raise ValueError("Local pair cannot support K >= 2.")
    assets = pair_root / "local_optimal_cg" / "input_assets"
    assets.mkdir(parents=True, exist_ok=True)
    source_ids = _write_local_assets(
        matrix_path=inputs["cci_t"], index_path=inputs["index_t"], indices=pair.source_indices_used,
        matrix_output=assets / "cci_t.npz", index_output=assets / "index_t.tsv",
    )
    target_ids = _write_local_assets(
        matrix_path=inputs["cci_tp"], index_path=inputs["index_tp"], indices=pair.target_indices,
        matrix_output=assets / "cci_tp.npz", index_output=assets / "index_tp.tsv",
    )
    local_maturity_t = _write_local_maturity(maturity_t, source_ids, assets / "maturity_t.csv")
    local_maturity_tp = _write_local_maturity(maturity_tp, target_ids, assets / "maturity_tp.csv")
    seed = pair_seed(global_seed, pair.source_cluster, pair.target_cluster)
    request = CoarseFrontendRequest(
        h5ad_t=inputs["h5ad_t"], h5ad_tp=inputs["h5ad_tp"],
        cci_t=assets / "cci_t.npz", cci_tp=assets / "cci_tp.npz",
        cci_index_t=assets / "index_t.tsv", cci_index_tp=assets / "index_tp.tsv",
        grn_t=inputs["grn_t"], grn_tp=inputs["grn_tp"],
        maturity_t=local_maturity_t, maturity_tp=local_maturity_tp, seed=seed,
    )
    prepared = prepare_coarse_input("maturity_cci_grn_two_stage", request)
    prepared = replace(
        prepared,
        micro_pij=pair.pij_conditional.astype(np.float32),
        micro_ei=float(effective_information(pair.pij_conditional.copy())),
        provenance={
            **dict(prepared.provenance),
            "local_pair_source_cluster": pair.source_cluster,
            "local_pair_target_cluster": pair.target_cluster,
            "local_micro_pij": "conditional_submatrix_of_global_PIJ_micro_train",
        },
    )
    prepared.validate()
    training_root = pair_root / "local_optimal_cg"
    config = WYTTwoStageDeltaEIConfig(
        k=k_local, out_dir=training_root, seed=seed, device=device, lambda_dev=0.05,
    )
    train_deltaei(prepared, config)
    summary = collect_summary(training_root)
    return {
        "k_local": k_local, "seed": seed, "local_micro_ei": float(summary["EI_micro_fixed"]),
        "local_macro_ei": float(summary["EI_macro_best_checkpoint"]),
        "local_delta_ei": float(summary["delta_EI_best_checkpoint"]),
        "local_i_available": float(summary["I_available"]),
        "local_i_retained": float(summary["I_retained"]),
        "local_cq": float(summary["closure_quality"]), "training_root": training_root,
    }


def _read_index(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t").iloc[:, 0].astype(str).tolist()


def _write_local_assets(*, matrix_path: Path, index_path: Path, indices: np.ndarray, matrix_output: Path, index_output: Path) -> list[str]:
    matrix = sp.load_npz(matrix_path).tocsr()
    index = np.asarray(indices, dtype=int)
    if np.any(index < 0) or np.any(index >= matrix.shape[0]):
        raise IndexError("Local CCI indices are outside the source matrix.")
    units = _read_index(index_path)
    if matrix.shape != (len(units), len(units)):
        raise ValueError("CCI matrix does not match its index before local slicing.")
    sp.save_npz(matrix_output, matrix[index][:, index].tocsr())
    selected = [units[item] for item in index]
    pd.DataFrame({"spot_id": selected}).to_csv(index_output, sep="\t", index=False)
    return selected


def _write_local_maturity(source: Path, selected_ids: list[str], output: Path) -> Path:
    table = pd.read_csv(source)
    if not {"spot_id", "maturity"}.issubset(table.columns):
        raise ValueError(f"Maturity CSV must contain spot_id and maturity: {source}")
    table["spot_id"] = table["spot_id"].astype(str)
    lookup = table.set_index("spot_id", drop=False)
    missing = [item for item in selected_ids if item not in lookup.index]
    if missing:
        raise ValueError(f"Maturity CSV does not cover local spots: {missing[:10]}")
    lookup.loc[selected_ids].to_csv(output, index=False)
    return output
