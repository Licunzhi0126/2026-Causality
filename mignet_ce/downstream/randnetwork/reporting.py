"""Machine-readable reports and null histograms."""
import os
import pandas as pd
from .audit import write_json
from .statistics import statistics


def csv(path, rows):
    temp = path.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def report(root, baseline, repeats, expected, time_pair):
    csv(root / 'repetitions.csv', repeats)
    if expected != 100 or len(repeats) != 100 or baseline is None:
        write_json(root / 'statistics.json', {'status': 'DIAGNOSTIC_ONLY', 'n_valid': len(repeats), 'pvalues_published': False})
        return
    stats = [{**row, 'time_pair': time_pair} for row in statistics(baseline, repeats)]
    csv(root / 'pvalues.csv', stats)
    write_json(root / 'statistics.json', {'status': 'COMPLETED', 'results': stats,
               'interpretation': 'Monte Carlo global-random-network null; not biological replicate significance. Across time pairs apply Holm before confirmatory claims.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for row, filename in zip(stats, ('null_delta_ei.png', 'null_cq.png')):
        fig, ax = plt.subplots()
        ax.hist([r[row['metric']] for r in repeats], bins=15)
        ax.axvline(row['real'], color='red', label=f"Real; p={row['p_one_sided']:.6g}")
        ax.set(xlabel=row['metric'], ylabel='Random-network count', title=time_pair)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=150)
        plt.close(fig)
