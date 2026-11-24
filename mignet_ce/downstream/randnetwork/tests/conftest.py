import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def pair(tmp_path):
    data = tmp_path / 'inputs'
    data.mkdir()
    maturity = tmp_path / 'maturity'
    maturity.mkdir()
    inputs = {}
    rng = np.random.default_rng(17)
    units = [f'u{i}' for i in range(12)]
    genes = [f'g{i}' for i in range(8)]
    for suffix, name in [('t', 'source'), ('tp', 'target')]:
        h5 = data / f'{suffix}.h5ad'
        with h5py.File(h5, 'w') as handle:
            for axis, ids in [('obs', units), ('var', genes)]:
                group = handle.create_group(axis)
                group.attrs['_index'] = '_index'
                group.create_dataset('_index', data=np.array(ids, dtype='S'))
            handle.create_dataset('X', data=rng.poisson(7, (12, 8)).astype(float))
            handle.create_group('obsm').create_dataset('spatial', data=rng.random((12, 2)))
        index = data / f'{suffix}.tsv'
        pd.DataFrame({'unit_id': units}).to_csv(index, sep='\t', index=False)
        cci = rng.random((12, 12))
        cci[cci < .6] = 0
        np.fill_diagonal(cci, 0)
        cci_path = data / f'{suffix}.npz'
        sp.save_npz(cci_path, sp.csr_matrix(cci))
        # Dense bipartite roles ensure all nodes remain represented under uniform sampling.
        edges = [(a, b, float(rng.uniform(.1, 2))) for a in genes[:4] for b in genes[4:]]
        grn = data / f'{suffix}.csv'
        pd.DataFrame(edges[:-2], columns=['regulator', 'target', 'weight']).to_csv(grn, index=False)
        inputs.update({f'h5ad_{suffix}': h5, f'index_{suffix}': index, f'cci_{suffix}': cci_path, f'grn_{suffix}': grn})
        pd.DataFrame({'spot_id': units, 'maturity': np.linspace(0, 1, 12)}).to_csv(maturity / f'{name}.csv', index=False)
    return inputs, maturity
