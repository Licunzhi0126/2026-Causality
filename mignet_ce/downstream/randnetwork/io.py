"""Strict input contracts, preserving original index files and GRN columns."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from mignet_ce.downstream.analysis.io import canonical_layer_stem, cci_index_path, cci_path, grn_path, layer_h5ad


def factory_inputs(root, organ, layer, source, target):
    if not Path(root).is_absolute():
        raise ValueError('data-root must be absolute')
    result = {}
    for suffix, time in [('t', source), ('tp', target)]:
        for key, resolver in [('h5ad', layer_h5ad), ('cci', cci_path), ('index', cci_index_path), ('grn', grn_path)]:
            result[f'{key}_{suffix}'] = resolver(root, layer, time, organ)
    return result


def read_cci(path, index):
    ids = pd.read_csv(index, sep='\t', dtype=str).iloc[:, 0].tolist()
    raw = sp.load_npz(path).tocoo()
    if raw.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids) or not ids:
        raise ValueError('CCI index/shape invalid')
    if len(set(zip(raw.row, raw.col))) != raw.nnz:
        raise ValueError('Duplicate CCI entries require explicit preprocessing')
    if not np.isfinite(raw.data).all() or (raw.data < 0).any():
        raise ValueError('CCI must have finite nonnegative weights (formal loader contract)')
    explicit_zeros = int(np.sum(raw.data == 0))
    raw.eliminate_zeros()
    # Keep all nonzero entries, including loops, for the random graph's edge and weight counts.
    return raw, ids, explicit_zeros


def read_grn(path):
    table = pd.read_csv(path, float_precision='round_trip')
    aliases = {'source': {'regulator', 'tf', 'source', 'src'}, 'target': {'target', 'target_gene', 'gene', 'dst'}, 'weight': {'weight', 'importance', 'score'}}
    columns = {}
    for role, names in aliases.items():
        found = [col for col in table if str(col).strip().lower() in names]
        if len(found) != 1:
            raise ValueError(f'Ambiguous/missing GRN {role} column')
        columns[role] = found[0]
    if set(table) != set(columns.values()):
        raise ValueError('Additional GRN columns have unresolved edge semantics; expected only source/target/weight')
    table = pd.read_csv(path, dtype={columns['source']: str, columns['target']: str}, float_precision='round_trip')
    for role in ('source', 'target'):
        if table[columns[role]].isna().any():
            raise ValueError('Missing GRN node ID')
        table[columns[role]] = table[columns[role]].astype(str)
    src, dst, weight = (table[columns[key]] for key in ('source', 'target', 'weight'))
    if not len(table) or not np.isfinite(weight).all() or (weight == 0).any():
        raise ValueError('GRN requires finite nonzero edge weights')
    if (src == dst).any() or table.duplicated([columns['source'], columns['target']]).any():
        raise ValueError('GRN self loops or duplicate edges require explicit preprocessing')
    return table, columns, sorted(set(src)), sorted(set(dst))
