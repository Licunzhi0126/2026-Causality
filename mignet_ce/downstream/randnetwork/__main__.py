"""Both-time, both-network experiment CLI."""
import argparse
from pathlib import Path
from .config import RandomNetworkConfig
from .io import factory_inputs
from .runner import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('organ', 'source-time', 'target-time'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--layer', default='spot')
    parser.add_argument('--maturity-root', type=Path)
    for name, default in [('n-random', 100), ('base-seed', 42), ('train-seed', 20260930), ('k', 40), ('max-retries', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'cuda', 'auto'))
    for name in ('randomize-only', 'resume', 'fail-fast'):
        parser.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv)
    if not args.randomize_only and args.maturity_root is None:
        parser.error('Full training requires --maturity-root')
    controls = RandomNetworkConfig(**{key: getattr(args, key) for key in ('n_random', 'base_seed', 'train_seed', 'k', 'device', 'max_retries')})
    inputs = factory_inputs(args.data_root, args.organ, args.layer, args.source_time, args.target_time)
    run_experiment(inputs, args.output_root, f'{args.organ}_{args.layer}_{args.source_time}_to_{args.target_time}',
                   controls, {}, args.maturity_root, args.randomize_only, args.resume, args.fail_fast)


if __name__ == '__main__':
    main()
