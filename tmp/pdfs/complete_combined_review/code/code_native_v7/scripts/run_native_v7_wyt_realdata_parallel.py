#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'run_native_v7_wyt_realdata.py'


def run_one(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, str(log_path)


def main() -> None:
    p = argparse.ArgumentParser(description='Parallel true-data native V7 + WYT experiment launcher.')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--h5ad-t', type=Path, required=True)
    p.add_argument('--h5ad-tp', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--k', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--graph-modes', nargs='+', default=['cci_only', 'cci_g_integrated'])
    p.add_argument('--local-modes', nargs='+', default=['legacy_features', 'coords', 'all_features'])
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    stages = list(manifest['stages'])
    if len(stages) != 2:
        raise ValueError('Manifest must contain exactly two stages.')
    stage_t, stage_tp = stages
    left = manifest['stages'][stage_t]
    right = manifest['stages'][stage_tp]
    args.out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
    })

    jobs: list[tuple[str, list[str], Path]] = []
    for graph in args.graph_modes:
        for local in args.local_modes:
            label = f'{graph}__local_{local}'
            out = args.out_root / label
            cmd = [
                sys.executable, str(RUNNER),
                '--stage-t', stage_t,
                '--stage-tp', stage_tp,
                '--h5ad-t', str(args.h5ad_t),
                '--h5ad-tp', str(args.h5ad_tp),
                '--cci-t', left['cci_total'],
                '--cci-tp', right['cci_total'],
                '--cci-index-t', left['cci_index'],
                '--cci-index-tp', right['cci_index'],
                '--grn-t', left['grn_edges'],
                '--grn-tp', right['grn_edges'],
                '--out-root', str(out),
                '--k', str(args.k),
                '--epochs', str(args.epochs),
                '--seeds', str(args.seed),
                '--graph-modes', graph,
                '--local-graph-modes', local,
                '--device', args.device,
            ]
            jobs.append((label, cmd, args.out_root / 'launcher_logs' / f'{label}.log'))

    statuses: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(run_one, cmd, log, env): label for label, cmd, log in jobs
        }
        for future in as_completed(future_map):
            label = future_map[future]
            code, log = future.result()
            row = {'job': label, 'returncode': code, 'log': log}
            statuses.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    (args.out_root / 'parallel_status.json').write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    failed = [row for row in statuses if row['returncode'] != 0]
    frames = []
    for label, _, _ in jobs:
        result = args.out_root / label / 'all_results.csv'
        if result.exists():
            frame = pd.read_csv(result)
            frame.insert(0, 'launcher_job', label)
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(args.out_root / 'combined_results.csv', index=False)
    if failed:
        raise SystemExit(f'{len(failed)} jobs failed; see parallel_status.json and launcher_logs.')


if __name__ == '__main__':
    main()
