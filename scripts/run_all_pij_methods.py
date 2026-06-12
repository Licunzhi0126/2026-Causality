#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, DEFAULT_LEVEL_PAIRS, PIJ_METHODS, TemporalRunConfig, VerticalPairSpec
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vertical MIGNet for multiple Pij methods.")
    parser.add_argument("--methods", nargs="+", choices=sorted(PIJ_METHODS), default=["joint_nmf", "laplacian", "3dot", "slat"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument(
        "--level-pairs",
        nargs="+",
        default=[pair.label().replace("_to_", ":") for pair in DEFAULT_LEVEL_PAIRS],
    )
    parser.add_argument("--export-pij", action="store_true")
    parser.add_argument("--pij-feature-components", type=int, default=30)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-gamma", type=float, default=1.0)
    parser.add_argument("--ot-max-iter", type=int, default=100)
    parser.add_argument("--ot-sim-k", type=int, default=10)
    parser.add_argument("--ot-dist-k", type=int, default=50)
    parser.add_argument("--slat-k-neighbors", type=int, default=20)
    parser.add_argument("--slat-hidden-dim", type=int, default=2048)
    parser.add_argument("--slat-epochs", type=int, default=6)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    base_cfg = TemporalRunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=args.organs,
        time_points=args.time_points,
        level_pairs=[VerticalPairSpec.parse(value) for value in args.level_pairs],
        export_pij=args.export_pij,
        pij_feature_components=args.pij_feature_components,
        ot_epsilon=args.ot_epsilon,
        ot_gamma=args.ot_gamma,
        ot_max_iter=args.ot_max_iter,
        ot_sim_k=args.ot_sim_k,
        ot_dist_k=args.ot_dist_k,
        slat_k_neighbors=args.slat_k_neighbors,
        slat_hidden_dim=args.slat_hidden_dim,
        slat_epochs=args.slat_epochs,
    )

    all_metrics = []
    all_summaries = []
    for method in args.methods:
        embedding_method = method if method in {"joint_nmf", "laplacian"} else "joint_nmf"
        cfg = replace(
            base_cfg,
            output_root=args.output_root / method,
            pij_method=method,
            embedding_method=embedding_method,
        )
        metrics = VerticalMIGNetPipeline(cfg).run()
        if not metrics.empty:
            all_metrics.append(metrics)
        summary_path = cfg.output_root / "run_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary["pij_method"] = method
            all_summaries.append(summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(args.output_root / "all_methods_metrics.csv", index=False)
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(args.output_root / "all_methods_run_summary.csv", index=False)


if __name__ == "__main__":
    main()
