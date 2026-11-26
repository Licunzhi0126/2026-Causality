import numpy as np
import scipy.sparse as sp
from ..io import read_cci


def test_original_self_loops_count_as_cci_edges(pair):
    inputs, _ = pair
    sp.save_npz(inputs['cci_t'], sp.eye(12, format='csr'))
    matrix, ids, zeros = read_cci(inputs['cci_t'], inputs['index_t'])
    assert matrix.nnz == len(ids) == 12
    assert np.array_equal(matrix.row, matrix.col)
    assert zeros == 0


def test_explicit_zeros_counted(pair):
    inputs, _ = pair
    sp.save_npz(inputs['cci_t'], sp.coo_matrix((np.array([0., 2.]), ([0, 1], [1, 2])), shape=(12, 12)))
    matrix, ids, zeros = read_cci(inputs['cci_t'], inputs['index_t'])
    assert matrix.nnz == 1 and zeros == 1 and len(ids) == 12
