#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT
from mignet_ce.pipelines.organ import DEFAULT_ORGAN_OUTPUT_ROOT, OrganPipelineConfig, OrganTemporalPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run organ-level MIGNet EI/DI/TE from cross-organ seurat_k40/louvain_k150.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ORGAN_OUTPUT_ROOT)
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5", "13.5"])
    parser.add_argument("--expected-organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--lower-layer", default="seurat_k40")
    parser.add_argument("--upper-layer", default="louvain_k150")
    parser.add_argument("--macro-group-column", default="organ")
    parser.add_argument("--cci-scope", choices=["all", "intra", "inter"], default="all")
    parser.add_argument("--allow-incomplete-organs", action="store_true")
    parser.add_argument("--expr-threshold", type=float, default=0.0)
    parser.add_argument("--cci-min", type=float, default=0.0)
    parser.add_argument("--top-k-targets-per-regulator", type=int, default=20)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-seed", type=int, default=42)
    parser.add_argument("--kraskov-k", type=int, default=3)
    parser.add_argument("--no-feature-log1p", action="store_true")
    parser.add_argument("--no-export-features", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OrganPipelineConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        lower_layer=args.lower_layer,
        upper_layer=args.upper_layer,
        macro_group_column=args.macro_group_column,
        expected_organs=args.expected_organs,
        time_points=args.time_points,
        strict_complete_organs=not args.allow_incomplete_organs,
        cci_scope=args.cci_scope,
        expr_threshold=args.expr_threshold,
        cci_min=args.cci_min,
        top_k_targets_per_regulator=args.top_k_targets_per_regulator,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        nmf_seed=args.nmf_seed,
        kraskov_k=args.kraskov_k,
        feature_log1p=not args.no_feature_log1p,
        export_features=not args.no_export_features,
    )
    metrics = OrganTemporalPipeline(cfg).run()
    print(metrics.loc[:, ["organ_scope", "lower_layer", "upper_layer", "cci_scope", "time_pair", "EI_lower", "EI_upper", "DI", "TE"]].to_string(index=False))
    print(f"Wrote metrics: {cfg.output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
