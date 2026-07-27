from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends._complete_combined_core import (
    sparse_shared_core_directed_nmf,
)
from mignet_ce.coarse_frontends.complete_combined_coarse import prepare
from mignet_ce.metrics import pairwise_shared_core_directed_nmf
from mignet_ce.representations.coarse_input import MacroPijInputs


def test_sparse_shared_core_nmf_matches_existing_dense_equations() -> None:
    rng = np.random.default_rng(17)
    source = rng.random((6, 6))
    target = rng.random((7, 7))
    dense = pairwise_shared_core_directed_nmf(
        source,
        target,
        n_components=3,
        max_iter=8,
        seed=31,
    )
    n_t, n_tp, metadata = sparse_shared_core_directed_nmf(
        sp.csr_matrix(source),
        sp.csr_matrix(target),
        components=3,
        max_iter=8,
        seed=31,
    )
    dense_n_t = np.hstack([dense[0], dense[1]])
    dense_n_tp = np.hstack([dense[2], dense[3]])
    combined = np.vstack([dense_n_t, dense_n_tp])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    dense_n_t = (dense_n_t - mean) / std
    dense_n_tp = (dense_n_tp - mean) / std
    np.testing.assert_allclose(n_t, dense_n_t, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(n_tp, dense_n_tp, rtol=2e-5, atol=2e-5)
    assert metadata["linear_algebra_backend"] == "sparse_exact_multiplicative_updates"


def _write_stage(tmp_path, label: str, unit_count: int, seed: int):
    rng = np.random.default_rng(seed)
    units = [f"{label}_{index}" for index in range(unit_count)]
    genes = ["G0", "G1", "G2", "G3"]
    data = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(2.0, size=(unit_count, len(genes))).astype(np.float32)),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=genes),
    )
    data.layers["counts"] = data.X.copy()
    data.obsm["spatial"] = rng.normal(size=(unit_count, 2)).astype(np.float32)
    h5ad = tmp_path / f"{label}.h5ad"
    data.write_h5ad(h5ad)

    cci = sp.csr_matrix(rng.random((unit_count, unit_count)).astype(np.float32))
    cci_path = tmp_path / f"{label}_CCI_total.npz"
    sp.save_npz(cci_path, cci)
    pd.DataFrame({"unit_id": units}).to_csv(
        tmp_path / f"{label}_index.tsv",
        sep="\t",
        index=False,
    )
    grn = tmp_path / f"{label}_grn_edges.csv"
    edge_frame = pd.DataFrame(
        {
            "regulator": ["G0", "G0", "G1", "G2"],
            "target": ["G1", "G2", "G2", "G3"],
            "weight": [1.0, 0.7, -0.8, 0.5],
        }
    )
    if label.endswith("tp"):
        edge_frame = edge_frame.iloc[:3].copy()
    edge_frame.to_csv(grn, index=False)
    return h5ad, cci_path, grn


def test_complete_combined_frontend_and_strict_evaluation(tmp_path) -> None:
    h5ad_t, cci_t, grn_t = _write_stage(tmp_path, "stage_t", 7, 101)
    h5ad_tp, cci_tp, grn_tp = _write_stage(tmp_path, "stage_tp", 8, 102)
    prepared = prepare(
        CoarseFrontendRequest(
            h5ad_t=h5ad_t,
            h5ad_tp=h5ad_tp,
            cci_t=cci_t,
            cci_tp=cci_tp,
            grn_t=grn_t,
            grn_tp=grn_tp,
            nmf_components=2,
            nmf_max_iter=3,
            mid_dim=3,
            grn_topk_targets=3,
            grn_state_dim=4,
            grn_knn_k=2,
            seed=19,
        )
    )
    assert prepared.method == "complete_combined_coarse"
    assert prepared.feature_blocks_t["N"].shape == (7, 4)
    assert prepared.feature_blocks_t["X"].shape[0] == 7
    assert (
        prepared.feature_blocks_t["X"].shape[1]
        != prepared.feature_blocks_tp["X"].shape[1]
    )
    assert prepared.provenance["uses_true_grn"] is True
    assert prepared.provenance["macro_feature_mode"] == (
        "pool_expression_then_recompute_true_GRN_G"
    )
    assert prepared.posthoc_evaluator is not None

    assignment_t = np.eye(3, dtype=np.float32)[np.arange(7) % 3]
    assignment_tp = np.eye(3, dtype=np.float32)[np.arange(8) % 3]
    pooled_t = {
        key: torch.tensor(
            (assignment_t.T @ values)
            / np.maximum(assignment_t.sum(axis=0)[:, None], 1e-12),
            dtype=torch.float32,
        )
        for key, values in prepared.feature_blocks_t.items()
    }
    pooled_tp = {
        key: torch.tensor(
            (assignment_tp.T @ values)
            / np.maximum(assignment_tp.sum(axis=0)[:, None], 1e-12),
            dtype=torch.float32,
        )
        for key, values in prepared.feature_blocks_tp.items()
    }
    macro_pij = prepared.macro_pij_builder(
        MacroPijInputs(
            z_macro_t=torch.zeros((3, 3)),
            z_macro_tp=torch.zeros((3, 3)),
            network_macro_t=torch.eye(3),
            network_macro_tp=torch.eye(3),
            feature_blocks_t=pooled_t,
            feature_blocks_tp=pooled_tp,
        )
    )
    assert tuple(macro_pij.shape) == (3, 3)
    torch.testing.assert_close(macro_pij.sum(dim=1), torch.ones(3))

    result = prepared.posthoc_evaluator(assignment_t, assignment_tp)
    for key in (
        "deltaEI_training_interface_pool_N_recompute_G",
        "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G",
        "deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G",
    ):
        assert np.isfinite(result[key])
    assert result["assignment_t"]["hardK"] == 3
    assert result["strict_raw_metadata"]["macro_cci_normalization"] == (
        "raw_S_transpose_A_S"
    )
