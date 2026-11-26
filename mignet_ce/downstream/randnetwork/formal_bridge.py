"""Fresh standard formal frontend and training, never direct-state propagation."""
from dataclasses import asdict
import platform
from pathlib import Path
import numpy as np
from .audit import digest, write_json, array_hash
from ..formal_protocol import maturity_columns, reset_seed, resolve_options


def resolve_configs(inputs, maturity_root, controls, options, training):
    from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
    from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig
    frontend = dict(options.get('frontend_options', {}))
    train = dict(options.get('training_options', {}))
    forbidden = set(inputs) | {'cci_index_t', 'cci_index_tp', 'maturity_t', 'maturity_tp'}
    if forbidden & frontend.keys() or {'out_dir', 'k', 'device'} & train.keys():
        raise ValueError('Formal config must not override input/output paths, k or device')
    if train.get('stage1_source') is not None or train.get('stage1_only', False):
        raise ValueError('Every repeat requires both stages from scratch; stage1_source/stage1_only forbidden')
    frontend, train = resolve_options(frontend, train, seed=controls.train_seed)
    maturity = tuple(Path(maturity_root) / name for name in ('source.csv', 'target.csv'))
    for key, value in maturity_columns(maturity).items():
        frontend.setdefault(key, value)
    request = CoarseFrontendRequest(
        **{key: value for key, value in inputs.items() if not key.startswith('index_')},
        cci_index_t=inputs['index_t'], cci_index_tp=inputs['index_tp'],
        maturity_t=maturity[0], maturity_tp=maturity[1], **frontend)
    config = WYTFormalTwoStageConfig(k=controls.k, device=controls.device, out_dir=Path(training), **train)
    request.validate()
    return request, config


def prepare_training(inputs, maturity_root, controls, options, root):
    from mignet_ce.coarse_frontends.formal_two_stage import prepare
    request, config = resolve_configs(inputs, maturity_root, controls, options, Path(root) / 'training')
    reset_seed(config.seed)
    return request, config, prepare(request)


def finish_training(request, config, prepared, root):
    import torch
    from wyt_deltaei_coarse_grain import train_deltaei
    from mignet_ce.downstream.perturbation.reporting import collect_summary

    reset_seed(config.seed)
    record = asdict(config)
    record.pop('out_dir')
    write_json(Path(root) / 'frontend_audit.json', {
        'request': asdict(request), 'training_config': record, 'training_config_hash': digest(record),
        'provenance': prepared.provenance,
        'products': {name: array_hash(getattr(prepared, name)) for name in
                     ('encoder_features_t', 'encoder_features_tp', 'micro_features_t', 'micro_features_tp', 'micro_pij')},
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'device': config.device, 'platform': platform.platform()},
        'deterministic_algorithms': True,
        'limitation': 'Reproducibility is scoped to fixed software/hardware; unsupported deterministic GPU operations fail.'})
    result = train_deltaei(prepared, config)
    summary = collect_summary(Path(root) / 'training')
    for field in ('delta_EI_best_checkpoint', 'closure_quality', 'Keff_source', 'Keff_target', 'hardK_t', 'hardK_tp'):
        if not np.isfinite(float(summary[field])):
            raise ValueError(f'Nonfinite metric: {field}')
    checkpoint = Path(root) / 'training' / 'best_joint.pt'
    if not checkpoint.is_file():
        raise ValueError('Formal best_joint checkpoint missing')
    return {**summary, 'best_checkpoint': str(checkpoint), 'best_epoch': result.best_epoch}


def run_training(inputs, maturity_root, controls, options, root):
    return finish_training(*prepare_training(inputs, maturity_root, controls, options, root), root)
