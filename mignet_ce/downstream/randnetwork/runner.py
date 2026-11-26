"""Frozen baseline, predetermined repeats, atomic completion and same-seed retry."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import traceback
import platform
import importlib.metadata
from .audit import digest, input_record, write_json, file_hash
from .materialize import materialize
from .formal_bridge import run_training, resolve_configs, prepare_training, finish_training
from .reporting import report
from ..experiment_scheduler import Job, Training, load_prepared, run_jobs, save_prepared
from ..formal_protocol import FORMAL_SEED, resolve_options
from ..post_perbutation.scheduler import run_lock


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
                   randomize_only=False, resume=False, fail_fast=False, trainer=None, execution=None):
    root = Path(output_root).resolve() / time_pair
    # Never write inside a factory or maturity input directory.
    protected = [Path(path).resolve().parent for path in inputs.values()]
    if maturity_root is not None:
        protected.append(Path(maturity_root).resolve())
    if any(root == path or path in root.parents for path in protected):
        raise ValueError('Output overlaps an input directory')
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        return _run_locked(inputs, root, time_pair, controls, options, maturity_root, randomize_only,
                           resume, fail_fast, trainer, execution)


def _jobs_for(label, idx, root, inputs, controls, options, maturity_root, randomize_only):
    return Job(label, _prepare_job, (idx, root / label, inputs, controls, options, maturity_root, randomize_only))


def _prepare_job(staging, idx, destination, inputs, controls, options, maturity_root, randomize_only):
    start = perf_counter()
    patched, audits = (inputs, {}) if idx is None else materialize(inputs, destination / 'graphs', controls.base_seed, idx)
    attempt = destination / f'attempt_{len(list(destination.glob("attempt_*"))):03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    if randomize_only:
        return _completed({}, attempt, patched, audits, start)
    request, config, prepared = prepare_training(patched, maturity_root, controls, options, attempt)
    cache = staging / 'prepared.pkl'
    save_prepared(cache, prepared)
    return Training(_train_job, (request, config, cache, attempt, patched, audits, start), (config.out_dir,))


def _train_job(device, request, config, cache, attempt, patched, audits, start):
    summary = finish_training(request, replace(config, device=device), load_prepared(cache), attempt)
    return _completed(summary, attempt, patched, audits, start)


def _completed(summary, attempt, patched, audits, start):
    artifacts = {str(p.resolve()): file_hash(p) for p in attempt.rglob('*') if p.is_file()}
    artifacts.update({str(Path(patched[key]).resolve()): audits[key]['sha256'] for key in audits})
    return {**summary, 'graph_audit': audits, 'elapsed_seconds': perf_counter() - start, 'artifacts': artifacts}


def _input_stats(inputs):
    return {key: (state.st_size, state.st_mtime_ns, state.st_ctime_ns)
            for key, state in ((key, Path(path).stat()) for key, path in inputs.items())}


def _run_locked(inputs, root, time_pair, controls, options, maturity_root, randomize_only,
                resume, fail_fast, trainer, execution):
    if controls.train_seed != FORMAL_SEED or controls.base_seed != FORMAL_SEED:
        raise ValueError(f'Formal Core requires training and network master seed {FORMAL_SEED}')
    if not randomize_only:
        resolve_options(options.get('frontend_options'), options.get('training_options'), strict=True)
    frozen = dict(inputs)
    if maturity_root is not None:
        frozen.update(maturity_t=Path(maturity_root) / 'source.csv', maturity_tp=Path(maturity_root) / 'target.csv')
    initial = input_record(frozen)
    initial_stats = _input_stats(frozen)
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
    staged = {}
    if execution is not None and trainer is None:
        if randomize_only:
            execution = replace(execution, devices=('cpu',))
        groups = ([jobs[:1], jobs[1:]] if not randomize_only else [jobs])
        for group in groups:
            if not group:
                continue
            pending = [(label, idx) for label, idx in group
                       if reusable(root / label / 'result.json', fingerprint, resume) is None]
            group_name = 'P0' if group[0][1] is None else 'Random networks'
            cached = len(group) - len(pending)
            for _ in range(controls.max_retries + 1):
                def save_result(index, outcome):
                    label, idx = pending[index]
                    if _input_stats(frozen) != initial_stats:
                        raise ValueError('Frozen input changed during experiment')
                    record = {**outcome, 'status': 'FAILED' if 'error' in outcome else 'COMPLETED',
                              'fingerprint': fingerprint, 'repeat_idx': idx,
                              'network_seed': None if idx is None else [controls.base_seed, idx],
                              'training_config_hash': digest(resolved.get('training', {}))}
                    write_json(root / label / 'result.json', record)
                    if record['status'] == 'FAILED':
                        attempt_root = root / label
                        write_json(attempt_root / f'failure_{len(list(attempt_root.glob("failure_*"))):03d}.json', record)
                    elif idx is None:
                        write_json(root / label / 'summary.json', outcome)
                    staged[label] = record
                outcomes = run_jobs(root, group_name,
                    [_jobs_for(label, idx, root, inputs, controls, options, maturity_root, randomize_only)
                     for label, idx in pending], execution, cached=cached, fail_fast=fail_fast, on_result=save_result)
                for index, outcome in enumerate(outcomes):
                    if pending[index][0] not in staged:
                        save_result(index, outcome)
                pending = [(label, idx) for label, idx in pending if staged[label]['status'] == 'FAILED']
                cached = len(group) - len(pending)
                if not pending:
                    break
            if group_name == 'P0' and pending:
                write_json(root / 'run_manifest.json', {'status': 'INCOMPLETE', 'failures': ['baseline']})
                raise RuntimeError(f'Failed baseline; see {root / "baseline" / "result.json"}')
            if input_record(frozen) != initial:
                raise ValueError('Frozen input changed during experiment')
    trainer = trainer or run_training
    for label, idx in jobs:
        destination = root / label
        result_path = destination / 'result.json'
        record = staged[label] if label in staged else reusable(result_path, fingerprint, resume)
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
                    record = {'status': 'COMPLETED', 'fingerprint': fingerprint, 'repeat_idx': idx,
                              'network_seed': None if idx is None else [controls.base_seed, idx],
                              'training_config_hash': digest(resolved.get('training', {})),
                              **_completed(summary, attempt_root, patched, audits, start)}
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
