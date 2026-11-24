import numpy as np
import scipy.sparse as sp
from ..audit import input_record, file_hash
from ..materialize import materialize
from ..runner import run_experiment
from ..config import RandomNetworkConfig
import json


def test_four_graphs_and_frozen_inputs(pair, tmp_path):
    inputs, _ = pair
    before = input_record(inputs)
    patched, audits = materialize(inputs, tmp_path / 'a' / 'graphs', 42, 0)
    again, _ = materialize(inputs, tmp_path / 'b' / 'graphs', 42, 0)
    assert input_record(inputs) == before
    assert set(audits) == {'cci_t', 'grn_t', 'cci_tp', 'grn_tp'}
    for key in inputs:
        if key not in audits:
            assert patched[key] == inputs[key]
        else:
            assert patched[key] != inputs[key]
            assert file_hash(patched[key]) == file_hash(again[key])
    for key in ('cci_t', 'cci_tp'):
        a, b = sp.load_npz(inputs[key]), sp.load_npz(patched[key])
        assert a.nnz == b.nnz and np.array_equal(np.sort(a.data), np.sort(b.data))


def test_three_seed_randomize_only(pair, tmp_path):
    inputs, _ = pair
    def forbidden_trainer(*args):
        raise AssertionError('randomize-only invoked training')
    root = run_experiment(inputs, tmp_path / 'out', 'smoke', RandomNetworkConfig(n_random=3), {},
                          randomize_only=True, trainer=forbidden_trainer)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['status'] == 'RANDOMIZED_ONLY' and manifest['n_valid'] == 3
    assert not (root / 'pvalues.csv').exists()
    assert len(list(root.glob('repeats/seed_*/graphs/*.npz'))) == 6
