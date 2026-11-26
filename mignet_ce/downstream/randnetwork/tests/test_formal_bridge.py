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


def _make_training_pair(pair):
    import h5py
    import scipy.sparse as sp
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
    return inputs, maturity


def test_complete_synthetic_training(pair, tmp_path):
    from ..formal_bridge import run_training

    inputs, maturity = _make_training_pair(pair)
    options = {'frontend_options': {'nmf_components': 2, 'nmf_max_iter': 10, 'mid_dim': 4, 'grn_state_dim': 4},
               'training_options': {'epochs': 80, 'hidden_dim': 8, 'mid_dim': 4, 'knn_k': 3, 'log_every': 80}}
    result = run_training(inputs, maturity, RandomNetworkConfig(k=2), options, tmp_path / 'full')
    assert np.isfinite(result['delta_EI_best_checkpoint'])
    assert np.isfinite(result['closure_quality'])
    assert (tmp_path / 'full/training/best_joint.pt').is_file()


def test_zero_grn_intervention_matches_formal_at_spot_and_macro(pair, tmp_path):
    from mignet_ce.downstream.perturbation.direct_state_bridge import prepare_direct_state_two_stage
    from mignet_ce.downstream.perturbation.grn_io import read_grn
    from mignet_ce.downstream.perturbation.state_bridge import build_propagated_grn_states
    from mignet_ce.downstream.experiment_scheduler import load_prepared, save_prepared
    import pandas as pd

    inputs, maturity = pair
    request, config = resolve_configs(inputs, maturity, RandomNetworkConfig(k=2), {}, tmp_path / 'train')
    assert (request.alpha_cci, request.cci_feature_method, request.grn_feature_method) == (0.20, 'laplacian_hks', 'legacy_nmf')
    assert (config.seed, request.seed, config.knn_k, config.local_graph_mode) == (42, 42, 30, 'all_features')
    baseline = prepare(request)
    pij = tmp_path / 'PIJ_micro_train.npy'
    np.save(pij, baseline.micro_pij)
    for suffix, units in [('t', baseline.unit_ids_t), ('tp', baseline.unit_ids_tp)]:
        pd.DataFrame({'unit_id': units}).to_csv(tmp_path / f'assignments_{suffix}.csv', index=False)
    options = {'alpha_cci': 0.20, 'cci_feature_method': 'laplacian_hks', 'grn_feature_method': 'legacy_nmf',
               'maturity_id_column': 'spot_id', 'maturity_column': 'maturity'}
    graph = read_grn(inputs['grn_t'])
    states = build_propagated_grn_states(baseline_inputs=inputs, perturbed_grn_t=inputs['grn_t'], baseline_spot_pij=pij,
        frontend_options=options, baseline_graph=graph, perturbed_graph=graph,
        baseline_assets=baseline.macro_pij_builder.formal_assets)
    assert np.count_nonzero(states.delta_g_raw_t) == np.count_nonzero(states.delta_g_raw_tp) == 0
    direct = prepare_direct_state_two_stage(baseline_inputs=inputs, maturity_t=maturity / 'source.csv',
        maturity_tp=maturity / 'target.csv', frontend_options=options, states=states)
    cache = tmp_path / 'prepared.pkl'
    save_prepared(cache, direct)
    restored = load_prepared(cache)
    for key in ('encoder_features_t', 'encoder_features_tp', 'micro_features_t', 'micro_features_tp', 'micro_pij'):
        np.testing.assert_allclose(getattr(restored, key), getattr(baseline, key), rtol=1e-5, atol=1e-6)
    def macro(prepared):
        source, target = [{name: torch.tensor(values[:3], dtype=torch.float32) for name, values in block.items()}
                          for block in (prepared.feature_blocks_t, prepared.feature_blocks_tp)]
        return prepared.macro_pij_builder(MacroPijInputs(torch.ones(3, 4), torch.ones(3, 4),
            torch.eye(3), torch.eye(3), source, target)).detach().numpy()

    np.testing.assert_allclose(macro(restored), macro(baseline), rtol=1e-5, atol=1e-6)


def test_perturbation_staged_cpu_reuses_only_one_baseline(pair, tmp_path):
    import json
    from mignet_ce.downstream.experiment_scheduler import Execution
    from mignet_ce.downstream.perturbation.config import PerturbationConfig
    from mignet_ce.downstream.perturbation.runner import run_experiment

    inputs, maturity = _make_training_pair(pair)
    config = PerturbationConfig(k=2, mirror_fraction=.05, region_node_fraction=1, region_edge_fraction=1,
        frontend_options={'mid_dim': 4, 'grn_state_dim': 4},
        training_options={'epochs': 80, 'hidden_dim': 8, 'mid_dim': 4, 'log_every': 80})
    kwargs = dict(time_pair='pair', source_sample='source', target_sample='target', baseline_inputs=inputs,
                  output_root=tmp_path / 'grn', supplied_maturity_root=maturity, config=config,
                  strengths=(.25,), execution=Execution(cpu_workers=2))
    root = tmp_path / 'grn' / 'pair'
    first = run_experiment(pair_ids=('P1_degree_heterogeneity',), **kwargs)
    checkpoint = root / 'baseline' / 'training' / 'best_joint.pt'
    p0 = checkpoint.stat().st_mtime_ns
    second = run_experiment(pair_ids=('P4_assortativity',), **kwargs)
    assert checkpoint.stat().st_mtime_ns == p0
    assert all(record['training_status'] == 'COMPLETED' for record in first + second), first + second
    assert len({r['baseline_id'] for r in first + second}) == 1
    assert {r['region_seed'] for r in first + second} == {42}
    assert {r['train_seed'] for r in first + second} == {42}
    assert all(r['delta_CQ'] == r['perturbed_CQ'] - r['baseline_CQ'] for r in first + second)
    assert json.loads((root / 'progress.json').read_text(encoding='utf-8'))['P0']['reused'] == 1
    assert (root / 'figures' / 'grn_response_cq.pdf').is_file()
