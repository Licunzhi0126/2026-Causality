"""Small synthetic tests; these outputs are never formal experiment results."""
from dataclasses import replace
import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mignet_ce.downstream.analysis.dynamic_closure.analysis import effective_information
from mignet_ce.downstream.formal_core_analysis.tables import inspect_optimal_run
from mignet_ce.pij.wyt_single_kl import single_kl_pij_torch
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain import SHRMethod8Config, SHRMethod10Config, train_deltaei
from wyt_deltaei_coarse_grain.shr_method_8_model import PCAKNNLinearHeadEncoder
from wyt_deltaei_coarse_grain.shr_method_8_trainer import _fit_joint_pca_graph_inputs
from wyt_deltaei_coarse_grain.shr_graph_utils import build_knn_adjacency
from wyt_deltaei_coarse_grain.shr_training_utils import preprocess_route_a, _torch_sparse


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepared_input(method="shr_method_8"):
    """Uniform micro kernel exercises the genuine low-signal checkpoint path.

    The independent differentiable toy macro builder tests trainer/I/O mechanics,
    not scientific performance. Both micro EI and saved macro EI are computed.
    """
    rng = np.random.default_rng(31)
    nt, ntp = 18, 21
    micro = np.full((nt, ntp), 1. / ntp, dtype=np.float32)

    def network(count):
        array = rng.random((count, count)).astype(np.float32)
        return sp.csr_matrix(array / array.sum(axis=1, keepdims=True))

    return PreparedCoarseInput(
        method=method, unit_ids_t=[f"t{i}" for i in range(nt)],
        unit_ids_tp=[f"q{i}" for i in range(ntp)],
        network_t=network(nt), network_tp=network(ntp),
        encoder_features_t=rng.normal(size=(nt, 36)).astype(np.float32),
        encoder_features_tp=rng.normal(size=(ntp, 36)).astype(np.float32),
        micro_features_t=(3 * rng.normal(size=(nt, 5))).astype(np.float32),
        micro_features_tp=(3 * rng.normal(size=(ntp, 5))).astype(np.float32),
        micro_pij=micro, micro_ei=effective_information(micro),
        macro_pij_builder=lambda inputs: single_kl_pij_torch(
            inputs.z_macro_t, inputs.z_macro_tp, temperature=.03),
        maturity_t=np.linspace(0., 1., nt, dtype=np.float32),
        maturity_tp=np.linspace(0., 1., ntp, dtype=np.float32),
        provenance={"synthetic_test_fixture": True, "cci_feature_method": "laplacian_hks",
                    "grn_feature_method": "legacy_nmf", "alpha_cci": .20,
                    "h5ad_t": "synthetic/spot/heart/spot_heart_11.5.h5ad",
                    "h5ad_tp": "synthetic/spot/heart/spot_heart_12.5.h5ad"},
    )


def assert_saved_run(result, method):
    out = result.out_dir
    for name in ("best_model.pt", "summary.json", "config.json", "feature_manifest.json",
                 "PIJ_micro_train.npy", "PIJ_macro_train.npy", "S_t.npy", "S_tp.npy",
                 "assignments_t.csv", "assignments_tp.csv", "training_history.csv"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(out / "best_model.pt", map_location="cpu")
    assert result.best_epoch == checkpoint["epoch"] == summary["best_epoch"]
    assert result.final_delta_ei > 0
    assert summary["method"] == method
    assert summary["objective_version"] == method + "_v1"
    for name in ("S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy"):
        values = np.load(out / name, allow_pickle=False)
        assert np.isfinite(values).all() and np.min(values) >= 0
        np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-5)
    inspected = inspect_optimal_run(out)
    assert inspected["method"] == method
    assert inspected["trained_delta_EI"] == pytest.approx(result.final_delta_ei)
    assert 0 <= inspected["closure_JS_direct_bits"] <= 1
    return summary, checkpoint


def test_pca32_graph_is_reproducible_and_uses_all_dimensions():
    prepared = prepared_input()
    left, right, metadata = _fit_joint_pca_graph_inputs(prepared, 32, 42)
    again = _fit_joint_pca_graph_inputs(prepared, 32, 42)
    np.testing.assert_array_equal(left, again[0])
    np.testing.assert_array_equal(right, again[1])
    assert left.shape == (18, 32) and right.shape == (21, 32)
    assert metadata["graph_projection_sample_count"] == 39
    graph = build_knn_adjacency(left, 3)
    assert graph.shape == (18, 18)
    np.testing.assert_allclose(np.asarray(graph.sum(axis=1)).ravel(), 1., atol=1e-6)


def test_linear_head_has_finite_gradients():
    torch.manual_seed(2)
    model = PCAKNNLinearHeadEncoder(6, 12, 3, 2)
    features = torch.randn(8, 6, requires_grad=True)
    assignments, hidden = model(features, torch.eye(8), .7, return_embed=True)
    (assignments[:, 0].square().mean() + hidden.square().mean()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_method8_short_training_restores_auditable_checkpoint(tmp_path):
    prepared = prepared_input()
    config = SHRMethod8Config(k=3, out_dir=tmp_path / "run", hidden_dim=16,
                              epochs=3, knn_k=3, temperature=.7, log_every=1)
    result = train_deltaei(prepared, config)
    summary, checkpoint = assert_saved_run(result, "shr_method_8")
    assert summary["graph_projection_dim"] == 32
    assert summary["graph_projection_rebuild_verified"] is True
    assert (result.out_dir / "graph_projection.npz").is_file()
    model = PCAKNNLinearHeadEncoder(36, config.hidden_dim, config.k, config.gnn_layers)
    model.load_state_dict(checkpoint["encoder"])
    model.eval()
    features = preprocess_route_a(prepared.encoder_features_t, prepared.encoder_features_tp, config.local_dims)
    projected = _fit_joint_pca_graph_inputs(prepared, 32, config.seed)
    with torch.no_grad():
        for label, values, points in zip(("t", "tp"), features, projected[:2]):
            graph = _torch_sparse(build_knn_adjacency(points, config.knn_k), torch.device("cpu"))
            restored = model(torch.tensor(values), graph, config.temperature).numpy()
            np.testing.assert_allclose(restored, np.load(result.out_dir / f"S_{label}.npy"), atol=1e-6)
    with pytest.raises(FileExistsError):
        train_deltaei(prepared, config)


def test_method8_rejects_insufficient_pca_dimensions(tmp_path):
    prepared = prepared_input()
    small = replace(prepared, encoder_features_t=prepared.encoder_features_t[:, :4],
                    encoder_features_tp=prepared.encoder_features_tp[:, :4])
    with pytest.raises(ValueError, match="PCA requires"):
        train_deltaei(small, SHRMethod8Config(k=3, out_dir=tmp_path / "invalid"))
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("method, config_type", [("shr_method_8", SHRMethod8Config), ("shr_method_10", SHRMethod10Config)])
def test_no_positive_delta_does_not_promote_a_checkpoint(tmp_path, method, config_type):
    prepared = prepared_input(method)
    # A deterministic injective micro kernel has more EI than any K=3 kernel.
    micro = np.eye(len(prepared.unit_ids_t), len(prepared.unit_ids_tp), dtype=np.float32)
    prepared = replace(prepared, micro_pij=micro, micro_ei=effective_information(micro))
    out = tmp_path / "no_emergence"
    config = config_type(k=3, out_dir=out, hidden_dim=16, epochs=2, temperature=.7, log_every=1)
    with pytest.raises(RuntimeError, match="no eligible checkpoint"):
        train_deltaei(prepared, config)
    assert not (out / "summary.json").exists()
    assert not (out / "best_model.pt").exists()
    assert (out / "training_history.csv").is_file()
