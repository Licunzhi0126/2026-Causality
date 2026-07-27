#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import scipy.sparse as sp


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def candidates(roots: Iterable[Path], patterns: Iterable[str]) -> list[Path]:
    out: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            out.update(p.resolve() for p in root.rglob(pattern) if p.is_file())
    return sorted(out)


def select_unique(label: str, paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError(f'No candidate found for {label}.')
    if len(paths) > 1:
        pretty = '\n'.join(f'  - {p}' for p in paths)
        raise RuntimeError(f'Ambiguous candidates for {label}:\n{pretty}')
    return paths[0]


def infer_index(cci: Path) -> Path:
    stem = cci.name[:-len('_CCI_total.npz')] if cci.name.endswith('_CCI_total.npz') else cci.stem
    variants = [
        cci.with_name(stem + '_index.tsv'),
        cci.with_name(stem + '_index.csv'),
        cci.with_name(stem + '_units.csv'),
    ]
    for path in variants:
        if path.exists():
            return path
    raise FileNotFoundError(f'No index sidecar beside {cci}; tried {variants}.')


def read_count(path: Path) -> int:
    frame = pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',', dtype=str)
    return int(len(frame))


def main() -> None:
    p = argparse.ArgumentParser(description='Discover and validate true CCI/GRN inputs for native V7 + WYT.')
    p.add_argument('--roots', nargs='+', type=Path, required=True)
    p.add_argument('--organ', default='heart')
    p.add_argument('--stage-t', default='11.5')
    p.add_argument('--stage-tp', default='12.5')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    roots = [r.resolve() for r in args.roots]
    result: dict[str, object] = {'roots': [str(r) for r in roots], 'organ': args.organ, 'stages': {}}
    for stage in [args.stage_t, args.stage_tp]:
        compact = stage.replace('.', 'p')
        sample = f'spot_{args.organ}_{stage}'
        cci = select_unique(
            f'{stage} CCI_total',
            candidates(roots, [f'{sample}_CCI_total.npz', f'*{args.organ}*{stage}*CCI_total.npz', f'*{args.organ}*{compact}*CCI_total.npz']),
        )
        grn = select_unique(
            f'{stage} grn_edges.csv',
            candidates(roots, [f'**/{sample}/grn_edges.csv', f'*{args.organ}*{stage}*/grn_edges.csv', f'*{args.organ}*{compact}*/grn_edges.csv']),
        )
        index = infer_index(cci)
        matrix = sp.load_npz(cci)
        index_rows = read_count(index)
        if matrix.shape != (index_rows, index_rows):
            raise ValueError(f'{stage}: CCI shape {matrix.shape} != index rows {index_rows}.')
        result['stages'][stage] = {
            'cci_total': str(cci),
            'cci_index': str(index),
            'grn_edges': str(grn),
            'cci_shape': list(matrix.shape),
            'cci_nnz': int(matrix.nnz),
            'index_rows': index_rows,
            'files': {
                'cci_sha256': sha256(cci),
                'index_sha256': sha256(index),
                'grn_sha256': sha256(grn),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
