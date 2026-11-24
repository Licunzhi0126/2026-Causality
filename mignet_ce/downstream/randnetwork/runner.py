"""Frozen baseline, predetermined repeats, atomic completion and same-seed retry."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import traceback
import platform
import importlib.metadata
from .audit import digest, input_record, write_json, file_hash
from .materialize import materialize
from .formal_bridge import run_training, resolve_configs
from .reporting import report


def reusable(path, fingerprint, resume):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if not resume:
        raise FileExistsError(f'{path} exists; use --resume')
    if record.get('fingerprint') != fingerprint:
        raise ValueError(f'Resume input/config mismatch: {path}')
    if record.get('status') == 'COMPLETED':
        for file, checksum in record['artifacts'].items():
            if file_hash(file) != checksum:
                raise ValueError(f'Resume artifact mismatch: {file}')
        return record
    return None


def run_experiment(inputs, output_root, time_pair, controls, options, maturity_root=None,
                   randomize_only=False, resume=False, fail_fast=False, trainer=None):
    trainer = trainer or run_training
    root = Path(output_root).resolve() / time_pair
    # Never write inside a factory or maturity input directory.
    protected = [Path(path).resolve().parent for path in inputs.values()]
    if maturity_root is not None:
        protected.append(Path(maturity_root).resolve())
    if any(root == path or path in root.parents for path in protected):
        raise ValueError('Output overlaps an input directory')
    frozen = dict(inputs)
    if maturity_root is not None:
        frozen.update(maturity_t=Path(maturity_root) / 'source.csv', maturity_tp=Path(maturity_root) / 'target.csv')
    initial = input_record(frozen)
    resolved = {}
    if not randomize_only:
        if maturity_root is None:
            raise ValueError('maturity-root required for training')
        request, config = resolve_configs(inputs, maturity_root, controls, options, root / 'baseline' / 'training')
        resolved = {'frontend': asdict(request), 'training': {k: v for k, v in asdict(config).items() if k != 'out_dir'}}
    # Include source hashes so a changed implementation cannot silently reuse old results.
    source_root = Path(__file__).resolve().parents[3]
    code = {str(p.relative_to(source_root)): file_hash(p) for folder in ('mignet_ce', 'wyt_deltaei_coarse_grain')
            for p in (source_root / folder).rglob('*.py') if '__pycache__' not in str(p)}
    software = {'python': platform.python_version(), 'platform': platform.platform()}
    for package in ('numpy', 'scipy', 'pandas', 'torch', 'h5py', 'scikit-learn'):
        software[package] = importlib.metadata.version(package)
    locked = {'controls': asdict(controls), 'options': options, 'resolved': resolved, 'inputs': initial, 'software': software,
              'randomize_only': randomize_only, 'code': code}
    fingerprint = digest(locked)
    lock_path = root / 'config_locked.json'
    if lock_path.exists():
        if not resume or json.loads(lock_path.read_text(encoding='utf-8')) != json.loads(json.dumps(locked, default=str)):
            raise ValueError('Output already exists or frozen configuration mismatch')
    else:
        write_json(lock_path, locked)
    baseline, repeats, failures = None, [], []
    jobs = ([] if randomize_only else [('baseline', None)]) + [(f'repeats/seed_{i:03d}', i) for i in range(controls.n_random)]
    for label, idx in jobs:
        destination = root / label
        result_path = destination / 'result.json'
        record = reusable(result_path, fingerprint, resume)
        if record is None:
            for attempt in range(controls.max_retries + 1):
                start = perf_counter()
                try:
                    if input_record(frozen) != initial:
                        raise ValueError('Frozen input changed during experiment')
                    patched, audits = (inputs, {}) if idx is None else materialize(inputs, destination / 'graphs', controls.base_seed, idx)
                    # Each attempt has a fresh training directory; failed artifacts remain inspectable.
                    attempt_root = destination / f'attempt_{len(list(destination.glob("attempt_*"))):03d}'
                    attempt_root.mkdir(parents=True, exist_ok=False)
                    summary = {} if randomize_only else trainer(patched, maturity_root, controls, options, attempt_root)
                    if input_record(frozen) != initial:
                        raise ValueError('Trainer modified frozen input')
                    artifacts = {str(p.resolve()): file_hash(p) for p in attempt_root.rglob('*') if p.is_file()}
                    artifacts.update({str(Path(patched[key]).resolve()): audits[key]['sha256'] for key in audits})
                    record = {'status': 'COMPLETED', 'fingerprint': fingerprint, 'repeat_idx': idx,
                              'network_seed': None if idx is None else [controls.base_seed, idx],
                              'graph_audit': audits, 'training_config_hash': digest(resolved.get('training', {})),
                              'elapsed_seconds': perf_counter() - start, 'artifacts': artifacts, **summary}
                    write_json(result_path, record)
                    if idx is None:
                        write_json(destination / 'summary.json', summary)
                    break
                except Exception:
                    record = {'status': 'FAILED', 'fingerprint': fingerprint, 'repeat_idx': idx,
                              'network_seed': None if idx is None else [controls.base_seed, idx], 'traceback': traceback.format_exc()}
                    write_json(destination / f'failure_{len(list(destination.glob("failure_*"))):03d}.json', record)
                    write_json(result_path, record)
            if record['status'] != 'COMPLETED':
                failures.append(label)
                write_json(root / 'run_manifest.json', {'status': 'INCOMPLETE', 'failures': failures})
                if fail_fast or idx is None:
                    raise RuntimeError(f'Failed {label}; see {result_path}')
                continue
        if idx is None:
            baseline = record
        else:
            repeats.append(record)
        print(f'{time_pair}: {label} COMPLETED', flush=True)
    if not randomize_only:
        report(root, baseline, repeats, controls.n_random, time_pair)
    write_json(root / 'run_manifest.json', {'status': 'INCOMPLETE' if failures else ('RANDOMIZED_ONLY' if randomize_only else ('COMPLETED' if len(repeats) == 100 else 'SMOKE_COMPLETED')),
               'n_valid': len(repeats), 'requested': controls.n_random, 'failures': failures, 'fingerprint': fingerprint})
    if failures:
        raise RuntimeError(f'Incomplete experiment: {failures}')
    return root
