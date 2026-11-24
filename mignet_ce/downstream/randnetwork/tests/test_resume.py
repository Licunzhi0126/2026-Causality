import json
import pytest
from ..config import RandomNetworkConfig
from ..runner import run_experiment
from ..audit import write_json
from ..materialize import materialize


def test_repeat_37_same_seed_and_skip_completed(pair, tmp_path):
    inputs, maturity = pair
    calls = []
    fail = [True]
    def trainer(paths, maturity, controls, options, root):
        label = root.parent.name
        calls.append(label)
        if label == 'seed_036' and fail[0]:
            raise RuntimeError('simulated interruption')
        write_json(root / 'checkpoint.json', {'seed': controls.train_seed})
        return {'delta_EI_best_checkpoint': 1., 'closure_quality': .5}
    controls = RandomNetworkConfig(n_random=38, k=2)
    with pytest.raises(RuntimeError):
        run_experiment(inputs, tmp_path / 'out', 'pair', controls, {}, maturity, fail_fast=True, trainer=trainer)
    first = json.loads((tmp_path / 'out/pair/repeats/seed_036/result.json').read_text())
    assert calls.count('seed_000') == 1
    fail[0] = False
    run_experiment(inputs, tmp_path / 'out', 'pair', controls, {}, maturity, resume=True, trainer=trainer)
    final = json.loads((tmp_path / 'out/pair/repeats/seed_036/result.json').read_text())
    assert final['network_seed'] == first['network_seed'] == [42, 36]
    assert calls.count('seed_036') == 2 and calls.count('seed_000') == 1
    assert not (tmp_path / 'out/pair/pvalues.csv').exists()
    _, expected_audit = materialize(inputs, tmp_path / 'reference/graphs', 42, 36)
    assert {k: v['sha256'] for k, v in final['graph_audit'].items()} == {k: v['sha256'] for k, v in expected_audit.items()}
    # Completed graph corruption must never be silently skipped on resume.
    graph = tmp_path / 'out/pair/repeats/seed_000/graphs/grn_t.csv'
    graph.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact mismatch'):
        run_experiment(inputs, tmp_path / 'out', 'pair', controls, {}, maturity, resume=True, trainer=trainer)
