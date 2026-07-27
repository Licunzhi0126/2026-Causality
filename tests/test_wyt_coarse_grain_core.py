from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mignet_ce.pij.wyt_single_kl import single_kl_pij_torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.trainer import WYTDeltaEIConfig, train_deltaei


def _macro_pij(inputs: MacroPijInputs):
    return single_kl_pij_torch(inputs.z_macro_t, inputs.z_macro_tp)


def test_deltaei_core_smoke_writes_contract(tmp_path) -> None:
    rng = np.random.default_rng(12)
    count_t, count_tp = 7, 8
    network_t = sp.csr_matrix(rng.random((count_t, count_t)))
    network_tp = sp.csr_matrix(rng.random((count_tp, count_tp)))
    encoder_t = rng.normal(size=(count_t, 6)).astype(np.float32)
    encoder_tp = rng.normal(size=(count_tp, 6)).astype(np.float32)
    micro_t = rng.normal(size=(count_t, 4)).astype(np.float32)
    micro_tp = rng.normal(size=(count_tp, 4)).astype(np.float32)
    logits = micro_t @ micro_tp.T
    micro_pij = np.exp(logits - logits.max(axis=1, keepdims=True))
    micro_pij /= micro_pij.sum(axis=1, keepdims=True)
    prepared = PreparedCoarseInput(
        method="synthetic_single_kl",
        unit_ids_t=[f"t{index}" for index in range(count_t)],
        unit_ids_tp=[f"q{index}" for index in range(count_tp)],
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=encoder_t,
        encoder_features_tp=encoder_tp,
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij,
        micro_ei=0.1,
        macro_pij_builder=_macro_pij,
    )
    result = train_deltaei(
        prepared,
        WYTDeltaEIConfig(
            k=3,
            out_dir=tmp_path,
            hidden_dim=8,
            mid_dim=4,
            epochs=3,
            knn_k=2,
            log_every=1,
        ),
    )
    assert result.best_epoch in {1, 2, 3}
    for name in (
        "config.json",
        "input_manifest.json",
        "feature_manifest.json",
        "metrics.csv",
        "train.log",
        "best_model.pt",
        "S_t.npy",
        "S_tp.npy",
        "P_macro_t.npy",
        "P_macro_tp.npy",
        "PIJ_micro_train.npy",
        "PIJ_macro_train.npy",
        "P_macro_t.npz",
        "P_macro_tp.npz",
        "PIJ_micro.npz",
        "PIJ_macro.npz",
        "assignments_t.csv",
        "assignments_tp.csv",
        "summary.json",
    ):
        assert (tmp_path / name).exists(), name
    assert np.load(tmp_path / "S_t.npy").shape == (count_t, 3)
