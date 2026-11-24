import numpy as np
import torch
import pytest
from mignet_ce.coarse_frontends.formal_two_stage import prepare
from mignet_ce.representations.coarse_input import MacroPijInputs
from ..formal_bridge import resolve_configs
from ..config import RandomNetworkConfig
from ..materialize import materialize


def test_actual_four_graph_frontend_and_macro(pair, tmp_path):
    inputs, maturity = pair
    controls = RandomNetworkConfig(k=2)
    options = {'frontend_options': {'nmf_components': 2, 'nmf_max_iter': 10, 'mid_dim': 4, 'grn_state_dim': 4}}
    def build(paths):
        request, config = resolve_configs(paths, maturity, controls, options, tmp_path / 'training')
        assert config.seed == request.seed == controls.train_seed
        return prepare(request)
    baseline = build(inputs)
    patched, _ = materialize(inputs, tmp_path / 'run' / 'graphs', 42, 0)
    randomized = build(patched)
    repeated = build(inputs)
    assert np.array_equal(baseline.micro_pij, repeated.micro_pij)
    assert not np.allclose(baseline.micro_pij, randomized.micro_pij)
    assert not np.allclose(baseline.encoder_features_t, randomized.encoder_features_t)
    assert not np.allclose(baseline.encoder_features_tp, randomized.encoder_features_tp)
    def macro(prepared):
        blocks = [{key: torch.tensor(value[:3], dtype=torch.float32) for key, value in values.items()}
                  for values in (prepared.feature_blocks_t, prepared.feature_blocks_tp)]
        return prepared.macro_pij_builder(MacroPijInputs(torch.ones(3, 4), torch.ones(3, 4), torch.eye(3), torch.eye(3), *blocks)).detach().numpy()
    assert not np.allclose(macro(baseline), macro(randomized))
    # Isolate each network: changing any one must affect real computed products.
    for key in ('cci_t', 'grn_t', 'cci_tp', 'grn_tp'):
        single = build({**inputs, key: patched[key]})
        assert not np.allclose(single.micro_pij, baseline.micro_pij), key


def test_prevents_baseline_checkpoint_reuse(pair, tmp_path):
    inputs, maturity = pair
    with pytest.raises(ValueError, match='from scratch'):
        resolve_configs(inputs, maturity, RandomNetworkConfig(k=2),
                        {'training_options': {'stage1_source': 'baseline'}}, tmp_path)


def test_complete_synthetic_training(pair, tmp_path):
    import h5py
    import scipy.sparse as sp
    from ..formal_bridge import run_training
    inputs, maturity = pair
    # Two stable developmental communities: macro EI can exceed noisy micro EI.
    for suffix in ('t', 'tp'):
        with h5py.File(inputs[f'h5ad_{suffix}'], 'r+') as handle:
            counts = np.ones((12, 8))
            counts[:6, :4] = 30
            counts[6:, 4:] = 30
            handle['X'][:] = counts
        adjacency = np.full((12, 12), .05)
        adjacency[:6, :6] = 1
        adjacency[6:, 6:] = 1
        np.fill_diagonal(adjacency, 0)
        sp.save_npz(inputs[f'cci_{suffix}'], sp.csr_matrix(adjacency))
    options = {'frontend_options': {'nmf_components': 2, 'nmf_max_iter': 10, 'mid_dim': 4, 'grn_state_dim': 4},
               'training_options': {'epochs': 80, 'hidden_dim': 8, 'mid_dim': 4, 'knn_k': 3, 'log_every': 80}}
    result = run_training(inputs, maturity, RandomNetworkConfig(k=2), options, tmp_path / 'full')
    assert np.isfinite(result['delta_EI_best_checkpoint'])
    assert np.isfinite(result['closure_quality'])
    assert (tmp_path / 'full/training/best_joint.pt').is_file()
