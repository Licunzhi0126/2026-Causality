"""Materialize four graphs; all other input paths remain unchanged."""
import os
import shutil
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .io import read_cci, read_grn
from .randomize import streams, rewire, NETWORKS
from .audit import file_hash, write_json


def materialize(inputs, root, base_seed, repeat_idx):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patched, audits = dict(inputs), {}
    for name, seed in streams(base_seed, repeat_idx).items():
        seed_record = seed.state
        suffix = 'tp' if name.endswith('_tp') else 't'
        if name.startswith('cci'):
            original, ids, zeros = read_cci(inputs[name], inputs[f'index_{suffix}'])
            rows, cols, values, size = rewire(ids, ids, original.data, seed)
            matrix = sp.coo_matrix((values, (rows, cols)), shape=original.shape).tocsr()
            destination = root / f'{name}.npz'
            temp = root / f'{name}.tmp.npz'
            sp.save_npz(temp, matrix)
            os.replace(temp, destination)
            # Sidecar is a byte copy; request still uses the original frozen index path.
            shutil.copyfile(inputs[f'index_{suffix}'], root / f'{name}_index.tsv')
            old = set(zip(original.row.tolist(), original.col.tolist()))
            new = set(zip(rows.tolist(), cols.tolist()))
            if not np.array_equal(np.sort(matrix.data), np.sort(original.data)):
                raise ValueError('CCI weight invariant failed')
            original_loops = int(np.count_nonzero(original.row == original.col))
            randomized_loops = int(np.count_nonzero(rows == cols))
            if randomized_loops:
                raise ValueError('Random CCI contains self loops')
            detail = {'nodes': ids, 'explicit_zeros_removed': zeros, 'direction': 'directed (formal frontend contract)',
                      'original_self_loops': original_loops, 'randomized_self_loops': randomized_loops}
        else:
            table, columns, sources, targets = read_grn(inputs[name])
            rows, cols, values, size = rewire(sources, targets, table[columns['weight']].to_numpy(), seed)
            old = set(zip(table[columns['source']], table[columns['target']]))
            new_sources = np.asarray(sources)[rows]
            new_targets = np.asarray(targets)[cols]
            if set(new_sources) | set(new_targets) != set(sources) | set(targets):
                raise ValueError('Random GRN loses an endpoint-defined node; upstream cannot represent isolated genes. Same seed must be retried after protocol repair.')
            new = set(zip(new_sources, new_targets))
            table[columns['source']] = new_sources
            table[columns['target']] = new_targets
            table[columns['weight']] = values
            destination = root / f'{name}.csv'
            temp = root / f'{name}.tmp.csv'
            table.to_csv(temp, index=False)
            check, _, _, _ = read_grn(temp)
            if not np.array_equal(np.sort(check[columns['weight']]), np.sort(values)):
                raise ValueError('GRN serialization changed weights')
            os.replace(temp, destination)
            detail = {'sources': sources, 'targets': targets, 'direction': 'regulator -> target', 'weight_consumption': 'formal frontend uses abs(weight) and fixed top-k'}
        if len(new) != len(old):
            raise ValueError('Edge count invariant failed')
        patched[name] = destination.resolve()
        audits[name] = {**detail, 'seed': seed_record, 'legal_pairs': size, 'edges': len(new), 'overlap': len(old & new) / max(1, len(old)), 'sha256': file_hash(destination)}
    write_json(root.parent / 'audit.json', audits)
    return patched, audits
