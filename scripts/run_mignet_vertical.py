#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_LEVEL_PAIRS, TemporalRunConfig, VerticalPairSpec
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vertical MIGNet EI/DI/TE for configured domain layer pairs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument(
        "--level-pairs",
        nargs="+",
        default=[pair.label().replace("_to_", ":") for pair in DEFAULT_LEVEL_PAIRS],
        help="Pairs like spot:louvain_less_than5 louvain_less_than5:louvain_k150 louvain_k150:seurat_k40.",
    )
    parser.add_argument("--expr-threshold", type=float, default=0.0)
    parser.add_argument("--cci-min", type=float, default=0.0)
    parser.add_argument("--top-k-targets-per-regulator", type=int, default=20)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-seed", type=int, default=42)
    parser.add_argument("--embedding-method", choices=["joint_nmf", "laplacian"], default="joint_nmf")
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--no-laplacian-normalized", action="store_true")
    parser.add_argument("--kraskov-k", type=int, default=3)
    parser.add_argument("--no-feature-log1p", action="store_true")
    parser.add_argument("--no-export-features", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TemporalRunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=args.organs,
        time_points=args.time_points,
        level_pairs=[VerticalPairSpec.parse(value) for value in args.level_pairs],
        expr_threshold=args.expr_threshold,
        cci_min=args.cci_min,
        top_k_targets_per_regulator=args.top_k_targets_per_regulator,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        nmf_seed=args.nmf_seed,
        embedding_method=args.embedding_method,
        laplacian_components=args.laplacian_components,
        laplacian_normalized=not args.no_laplacian_normalized,
        kraskov_k=args.kraskov_k,
        feature_log1p=not args.no_feature_log1p,
        export_features=not args.no_export_features,
    )
    metrics = VerticalMIGNetPipeline(cfg).run()
    if metrics.empty:
        print(f"No metrics were produced. Inspect {cfg.output_root / 'run_summary.csv'}")
    else:
        print(metrics.loc[:, ["organ", "lower_layer", "upper_layer", "time_pair", "EI_lower", "EI_upper", "DI", "TE"]].to_string(index=False))
        print(f"Wrote metrics: {cfg.output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
