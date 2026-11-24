"""Calculate Group Micro EI using existing global training arrays only."""
import argparse
from pathlib import Path

from ..optimal_cg.group_micro_analysis import run_group_micro_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--no-export-matrices", action="store_true")
    args = parser.parse_args(argv)
    result = run_group_micro_analysis(training_root=args.training_root, output_root=args.output_root,
                                      eps=args.eps,
                                      export_matrices=not args.no_export_matrices)
    print(f"Group Micro EI: {result['status']}; completed={result['completed_count']} "
          f"failed={result['failed_count']} skipped={result['skipped_count']}")


if __name__ == "__main__":
    main()
