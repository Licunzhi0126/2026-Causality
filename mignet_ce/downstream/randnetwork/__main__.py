"""Both-time, both-network experiment CLI."""
import argparse
from pathlib import Path
from .config import RandomNetworkConfig
from .io import factory_inputs
from .runner import run_experiment
from ..experiment_scheduler import add_execution_arguments, execution_from_args
from ..formal_protocol import FORMAL_ALPHA, FORMAL_KNN_K, FORMAL_LOCAL_GRAPH_MODE, FORMAL_SEED


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('organ', 'source-time', 'target-time'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--layer', default='spot')
    parser.add_argument('--maturity-root', type=Path)
    for name, default in [('n-random', 100), ('base-seed', FORMAL_SEED), ('train-seed', FORMAL_SEED), ('k', 40), ('max-retries', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'auto'))
    parser.add_argument('--local-graph-mode', choices=(FORMAL_LOCAL_GRAPH_MODE,), default=FORMAL_LOCAL_GRAPH_MODE)
    parser.add_argument('--knn-k', type=int, choices=(FORMAL_KNN_K,), default=FORMAL_KNN_K)
    parser.add_argument('--alpha', type=float, choices=(FORMAL_ALPHA,), default=FORMAL_ALPHA)
    parser.add_argument('--epochs', type=int, default=1500)
    add_execution_arguments(parser)
    for name in ('randomize-only', 'resume', 'fail-fast'):
        parser.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv)
    if not args.randomize_only and args.maturity_root is None:
        parser.error('Full training requires --maturity-root')
    if (args.base_seed, args.train_seed) != (FORMAL_SEED, FORMAL_SEED) or args.epochs < 1:
        parser.error('Formal Core requires seed 42 and a positive epoch count')
    try:
        execution = execution_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if execution:
        args.device = 'cpu' if execution.devices == ('cpu',) else 'cuda'
    controls = RandomNetworkConfig(**{key: getattr(args, key) for key in ('n_random', 'base_seed', 'train_seed', 'k', 'device', 'max_retries')})
    inputs = factory_inputs(args.data_root, args.organ, args.layer, args.source_time, args.target_time)
    options = {'frontend_options': {'alpha_cci': args.alpha},
               'training_options': {'local_graph_mode': args.local_graph_mode, 'knn_k': args.knn_k,
                                    'epochs': args.epochs}}
    run_experiment(inputs, args.output_root, f'{args.organ}_{args.layer}_{args.source_time}_to_{args.target_time}',
                   controls, options, args.maturity_root, args.randomize_only, args.resume, args.fail_fast,
                   execution=execution)


if __name__ == '__main__':
    main()
