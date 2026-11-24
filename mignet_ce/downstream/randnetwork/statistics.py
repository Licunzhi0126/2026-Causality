"""Monte Carlo global-network-null statistics, with ties and add-one correction."""
import numpy as np

METRICS = ('delta_EI_best_checkpoint', 'closure_quality')


def statistics(baseline, repeats, expected=100):
    if expected != 100 or len(repeats) != 100:
        raise ValueError('Formal p-values require exactly 100 valid predetermined repeats')
    output = []
    for metric in METRICS:
        real = float(baseline[metric])
        values = np.asarray([row[metric] for row in repeats], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(real):
            raise ValueError('Nonfinite statistics')
        count = int(np.sum(values >= real))
        output.append({'metric': metric, 'real': real, 'random_mean': float(values.mean()),
                       'random_std': float(values.std(ddof=1)), 'random_median': float(np.median(values)),
                       'random_q025': float(np.quantile(values, .025)), 'random_q975': float(np.quantile(values, .975)),
                       'effect': float(real - values.mean()), 'count_rand_ge_real': count,
                       'n_valid': 100, 'p_one_sided': (1 + count) / 101})
    return output
