from __future__ import annotations

import argparse
from pathlib import Path

from .config import SensitivityConfig, default_alphas
from .workflow import run_sensitivity_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRN–CCI alpha sensitivity across natural hierarchy levels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument(
        "--time-pairs",
        nargs="+",
        default=["11.5->12.5", "12.5->13.5", "11.5->13.5"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(default_alphas()))
    parser.add_argument("--beta-n", type=float, default=SensitivityConfig.__dataclass_fields__["beta_n"].default)
    parser.add_argument("--beta-g", type=float, default=SensitivityConfig.__dataclass_fields__["beta_g"].default)
    parser.add_argument("--tau", type=float, default=SensitivityConfig.__dataclass_fields__["tau"].default)
    parser.add_argument("--canonical-alpha", type=float, default=SensitivityConfig.__dataclass_fields__["canonical_alpha"].default)
    parser.add_argument("--nmf-components", type=int, default=SensitivityConfig.__dataclass_fields__["nmf_components"].default)
    parser.add_argument("--nmf-max-iter", type=int, default=SensitivityConfig.__dataclass_fields__["nmf_max_iter"].default)
    parser.add_argument("--random-seed", type=int, default=SensitivityConfig.__dataclass_fields__["random_seed"].default)
    return parser


def main() -> None:
    args = _parser().parse_args()
    outputs = run_sensitivity_analysis(
        SensitivityConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            organ=args.organ,
            time_pairs=tuple(args.time_pairs),
            alphas=tuple(args.alphas),
            beta_n=args.beta_n,
            beta_g=args.beta_g,
            tau=args.tau,
            canonical_alpha=args.canonical_alpha,
            nmf_components=args.nmf_components,
            nmf_max_iter=args.nmf_max_iter,
            random_seed=args.random_seed,
        )
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
