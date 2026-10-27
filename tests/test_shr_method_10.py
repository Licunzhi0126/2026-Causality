from dataclasses import replace

import numpy as np
import pytest
import torch

from test_shr_method_8 import prepared_input, assert_saved_run, single_cpu_thread
from wyt_deltaei_coarse_grain import SHRMethod10Config, train_deltaei
from wyt_deltaei_coarse_grain.shr_method_10_model import (
    BidirectionalCrossTimeAttentionModel, stable_cross_time_bias, reverse_row_normalized_prior,
)
from wyt_deltaei_coarse_grain.shr_method_10_trainer import _row_normalized
from wyt_deltaei_coarse_grain.shr_training_utils import preprocess_route_a, _torch_sparse


def test_bidirectional_attention_rectangular_inputs_and_gradients():
    torch.manual_seed(3)
    model = BidirectionalCrossTimeAttentionModel(7, 7, 12, 3, heads=3)
    source, target = torch.randn(6, 7), torch.randn(9, 7)
    prior = torch.softmax(torch.randn(6, 9), dim=1)
    state = model(source, torch.eye(6), target, torch.eye(9), prior, .8)
    assert state["assignment_t"].shape == (6, 3)
    assert state["assignment_tp"].shape == (9, 3)
    loss = state["assignment_t"][:, 0].square().mean() + state["assignment_tp"][:, 1].square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.cross_t_from_tp.query.weight.grad.abs().sum() > 0
    assert model.cross_tp_from_t.query.weight.grad.abs().sum() > 0


def test_prior_with_zero_target_column_is_finite():
    prior = torch.tensor([[.7, .3, 0.], [.2, .8, 0.]])
    reverse = reverse_row_normalized_prior(prior)
    torch.testing.assert_close(reverse.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(reverse[-1], torch.full((2,), .5))
    assert stable_cross_time_bias(prior).abs().max() <= 5
    with pytest.raises(ValueError):
        stable_cross_time_bias(torch.tensor([[-1., 2.]]))


def test_method10_short_training_restores_auditable_checkpoint(tmp_path):
    prepared = prepared_input("shr_method_10")
    config = SHRMethod10Config(k=3, out_dir=tmp_path / "run", hidden_dim=16,
                               epochs=3, temperature=.7, log_every=1)
    result = train_deltaei(prepared, config)
    summary, checkpoint = assert_saved_run(result, "shr_method_10")
    assert summary["cross_time_attention"] is True
    assert "scheduler" in checkpoint
    model = BidirectionalCrossTimeAttentionModel(36, 36, config.hidden_dim, config.k,
                                                 heads=config.attention_heads)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    features = preprocess_route_a(prepared.encoder_features_t, prepared.encoder_features_tp, config.local_dims)
    with torch.no_grad():
        restored = model(torch.tensor(features[0]), _torch_sparse(_row_normalized(prepared.network_t), torch.device("cpu")),
                         torch.tensor(features[1]), _torch_sparse(_row_normalized(prepared.network_tp), torch.device("cpu")),
                         torch.tensor(prepared.micro_pij), config.temperature)
    for label in ("t", "tp"):
        np.testing.assert_allclose(restored[f"assignment_{label}"].numpy(),
                                   np.load(result.out_dir / f"S_{label}.npy"), atol=1e-6)
    rates = [row["learning_rate"] for row in result.metrics]
    assert rates[-1] < rates[0]
    with pytest.raises(FileExistsError):
        train_deltaei(prepared, config)


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 13}, {"gnn_layers": 1},
                                    {"lambda_dev": 0.}, {"log_every": 0},
                                    {"local_graph_mode": "coords"}])
def test_invalid_method10_configuration_fails_before_writing(tmp_path, kwargs):
    prepared = prepared_input("shr_method_10")
    with pytest.raises(ValueError):
        train_deltaei(prepared, SHRMethod10Config(k=3, out_dir=tmp_path / "invalid", **kwargs))
    assert not (tmp_path / "invalid").exists()
