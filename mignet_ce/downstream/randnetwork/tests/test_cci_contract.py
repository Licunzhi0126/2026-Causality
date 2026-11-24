import pytest
import numpy as np
import scipy.sparse as sp
from ..io import read_cci


def test_reject_unresolved_self_loops(pair):
    inputs, _ = pair
    sp.save_npz(inputs['cci_t'], sp.eye(12, format='csr'))
    with pytest.raises(ValueError, match='self loops'):
        read_cci(inputs['cci_t'], inputs['index_t'])


def test_explicit_zeros_counted(pair):
    inputs, _ = pair
    sp.save_npz(inputs['cci_t'], sp.coo_matrix((np.array([0., 2.]), ([0, 1], [1, 2])), shape=(12, 12)))
    matrix, ids, zeros = read_cci(inputs['cci_t'], inputs['index_t'])
    assert matrix.nnz == 1 and zeros == 1 and len(ids) == 12
