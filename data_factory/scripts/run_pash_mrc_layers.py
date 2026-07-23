#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from factory_common import FACTORY_OUTPUT_ROOT
from pash_mrc import PASHMRCConfig
from pash_mrc_layer_runner import run_pash_mrc_layers


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly build prospective PASH-MRC K40/K150 layers from each "
            "single-timepoint spot h5ad."
        )
    )
    parser.add_argument("--spot-root", type=Path, default=FACTORY_OUTPUT_ROOT / "spot")
    parser.add_argument("--factory-root", type=Path, default=FACTORY_OUTPUT_ROOT)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--n-hvg", type=int, default=1200)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-states", type=int, default=16)
    parser.add_argument("--ring-ends", nargs=3, type=int, default=(6, 18, 36))
    parser.add_argument("--composition-dim", type=int, default=24)
    parser.add_argument("--clustering-knn", type=int, default=10)
    parser.add_argument("--diagnostic-knn", type=int, default=6)
    parser.add_argument("--weight-expression", type=float, default=0.75)
    parser.add_argument("--weight-composition", type=float, default=1.00)
    parser.add_argument("--weight-neighborhood", type=float, default=0.30)
    parser.add_argument("--weight-spatial", type=float, default=0.16)
    parser.add_argument("--max-detached-piece", type=int, default=2)
    parser.add_argument("--icm-lambda", type=float, default=0.35)
    parser.add_argument("--icm-balance", type=float, default=0.03)
    parser.add_argument("--icm-passes", type=int, default=3)
    parser.add_argument("--min-domain-size-during-icm", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260723)
    return parser


def config_from_args(args: argparse.Namespace) -> PASHMRCConfig:
    return PASHMRCConfig(
        n_hvg=int(args.n_hvg),
        n_pca=int(args.n_pca),
        n_states=int(args.n_states),
        ring_ends=tuple(map(int, args.ring_ends)),
        composition_dim=int(args.composition_dim),
        k40=40,
        k150=150,
        clustering_knn=int(args.clustering_knn),
        diagnostic_knn=int(args.diagnostic_knn),
        weight_expression=float(args.weight_expression),
        weight_composition=float(args.weight_composition),
        weight_neighborhood=float(args.weight_neighborhood),
        weight_spatial=float(args.weight_spatial),
        max_detached_piece=int(args.max_detached_piece),
        icm_lambda=float(args.icm_lambda),
        icm_balance=float(args.icm_balance),
        icm_passes=int(args.icm_passes),
        min_domain_size_during_icm=int(args.min_domain_size_during_icm),
        random_state=int(args.random_state),
    )


def main() -> None:
    args = build_argparser().parse_args()
    run_pash_mrc_layers(
        spot_root=args.spot_root,
        factory_root=args.factory_root,
        config=config_from_args(args),
        sample_names=args.sample_names,
    )


if __name__ == "__main__":
    main()
