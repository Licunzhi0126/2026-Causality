#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import (
    DEFAULT_ABLATION_OUTPUT_ROOT,
    DEFAULT_DATA_ROOT,
    NETWORK_METHODS,
    PAIR_PRESETS,
    PIJ_METHODS,
    PIJ_METHOD_PRESETS,
    TemporalRunConfig,
    VerticalPairSpec,
)
from mignet_ce.pipelines.vertical_ablation import VerticalAblationPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vertical MIGNet network_method x pij_method ablation matrix.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ABLATION_OUTPUT_ROOT)
    parser.add_argument("--network-methods", nargs="+", choices=sorted(NETWORK_METHODS), default=["legacy_mixed_grn_cci", "cross_cell_multilayer"])
    parser.add_argument("--pij-methods", nargs="+", choices=sorted(PIJ_METHODS), default=None)
    parser.add_argument("--pij-method-preset", choices=sorted(PIJ_METHOD_PRESETS), default=None)
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument("--pair-preset", choices=sorted(PAIR_PRESETS), default="legacy_mixed_adjacent")
    parser.add_argument("--level-pairs", nargs="+", default=None)
    parser.add_argument("--expr-threshold", type=float, default=0.0)
    parser.add_argument("--cci-min", type=float, default=0.0)
    parser.add_argument("--top-k-targets-per-regulator", type=int, default=20)
    parser.add_argument("--cross-cell-lr-use-grn-gate", action="store_true")
    parser.add_argument("--cross-cell-top-k-edges", type=int, default=1000)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-seed", type=int, default=42)
    parser.add_argument("--export-pij", action="store_true")
    parser.add_argument("--export-pij-topk", type=int, default=10)
    parser.add_argument(
        "--pij-archive-root",
        type=Path,
        default=None,
        help="PIJ CSV archive root. Defaults to DATA_ROOT/pij.",
    )
    parser.add_argument("--export-pair-artifacts", action="store_true")
    parser.add_argument("--development-feature-root", type=Path, default=None)
    parser.add_argument("--pij-feature-aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--pij-missing-feature-policy", choices=["error", "impute_mean", "ignore"], default="impute_mean")
    parser.add_argument("--pij-feature-components", type=int, default=30)
    parser.add_argument("--pij-temperature", type=float, default=1.0)
    parser.add_argument("--pij-expr-weight", type=float, default=1.0)
    parser.add_argument("--pij-spatial-weight", type=float, default=0.2)
    parser.add_argument("--pij-graph-energy-weight", type=float, default=0.2)
    parser.add_argument("--pij-pseudotime-weight", type=float, default=0.5)
    parser.add_argument("--pij-sr-weight", type=float, default=0.5)
    parser.add_argument("--pij-potency-weight", type=float, default=0.5)
    parser.add_argument("--pij-velocity-weight", type=float, default=0.5)
    parser.add_argument("--pij-backward-pseudotime-weight", type=float, default=0.0)
    parser.add_argument("--pij-reverse-potency-weight", type=float, default=0.0)
    parser.add_argument("--pij-entropy-epsilon", type=float, default=0.05)
    parser.add_argument("--pij-use-unbalanced-ot", action="store_true")
    parser.add_argument("--pij-unbalanced-mass", type=float, default=1.0)
    parser.add_argument("--pij-cost-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-gamma", type=float, default=1.0)
    parser.add_argument("--ot-max-iter", type=int, default=100)
    parser.add_argument("--ot-sim-k", type=int, default=10)
    parser.add_argument("--ot-dist-k", type=int, default=50)
    parser.add_argument("--slat-k-neighbors", type=int, default=20)
    parser.add_argument("--slat-hidden-dim", type=int, default=2048)
    parser.add_argument("--slat-mlp-hidden", type=int, default=256)
    parser.add_argument("--slat-layers", type=int, default=1)
    parser.add_argument("--slat-epochs", type=int, default=6)
    parser.add_argument("--slat-alpha", type=float, default=0.01)
    parser.add_argument("--slat-temperature", type=float, default=0.1)
    parser.add_argument("--slat-seed", type=int, default=42)
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--no-laplacian-normalized", action="store_true")
    parser.add_argument("--kraskov-k", type=int, default=3)
    parser.add_argument("--no-feature-log1p", action="store_true")
    parser.add_argument("--no-export-features", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    level_pairs = (
        [VerticalPairSpec.parse(value) for value in args.level_pairs]
        if args.level_pairs is not None
        else list(PAIR_PRESETS[args.pair_preset])
    )
    if args.pij_methods is not None:
        pij_methods = args.pij_methods
    elif args.pij_method_preset is not None:
        pij_methods = list(PIJ_METHOD_PRESETS[args.pij_method_preset])
    else:
        pij_methods = list(PIJ_METHOD_PRESETS["core"])
    base_cfg = TemporalRunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=args.organs,
        time_points=args.time_points,
        level_pairs=level_pairs,
        expr_threshold=args.expr_threshold,
        cci_min=args.cci_min,
        top_k_targets_per_regulator=args.top_k_targets_per_regulator,
        cross_cell_lr_use_grn_gate=args.cross_cell_lr_use_grn_gate,
        cross_cell_top_k_edges=args.cross_cell_top_k_edges,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        nmf_seed=args.nmf_seed,
        export_pij=args.export_pij,
        export_pij_topk=args.export_pij_topk,
        pij_archive_root=args.pij_archive_root,
        export_pair_artifacts=args.export_pair_artifacts,
        development_feature_root=args.development_feature_root,
        pij_feature_aggregation=args.pij_feature_aggregation,
        pij_missing_feature_policy=args.pij_missing_feature_policy,
        pij_feature_components=args.pij_feature_components,
        pij_temperature=args.pij_temperature,
        pij_expr_weight=args.pij_expr_weight,
        pij_spatial_weight=args.pij_spatial_weight,
        pij_graph_energy_weight=args.pij_graph_energy_weight,
        pij_pseudotime_weight=args.pij_pseudotime_weight,
        pij_sr_weight=args.pij_sr_weight,
        pij_potency_weight=args.pij_potency_weight,
        pij_velocity_weight=args.pij_velocity_weight,
        pij_backward_pseudotime_weight=args.pij_backward_pseudotime_weight,
        pij_reverse_potency_weight=args.pij_reverse_potency_weight,
        pij_entropy_epsilon=args.pij_entropy_epsilon,
        pij_use_unbalanced_ot=args.pij_use_unbalanced_ot,
        pij_unbalanced_mass=args.pij_unbalanced_mass,
        pij_cost_metric=args.pij_cost_metric,
        ot_epsilon=args.ot_epsilon,
        ot_gamma=args.ot_gamma,
        ot_max_iter=args.ot_max_iter,
        ot_sim_k=args.ot_sim_k,
        ot_dist_k=args.ot_dist_k,
        slat_k_neighbors=args.slat_k_neighbors,
        slat_hidden_dim=args.slat_hidden_dim,
        slat_mlp_hidden=args.slat_mlp_hidden,
        slat_layers=args.slat_layers,
        slat_epochs=args.slat_epochs,
        slat_alpha=args.slat_alpha,
        slat_temperature=args.slat_temperature,
        slat_seed=args.slat_seed,
        laplacian_components=args.laplacian_components,
        laplacian_normalized=not args.no_laplacian_normalized,
        kraskov_k=args.kraskov_k,
        feature_log1p=not args.no_feature_log1p,
        export_features=not args.no_export_features,
    )
    metrics = VerticalAblationPipeline(
        base_cfg=base_cfg,
        network_methods=args.network_methods,
        pij_methods=pij_methods,
        output_root=args.output_root,
        fail_fast=args.fail_fast,
    ).run()
    print(f"Wrote manifest: {args.output_root / 'ablation_manifest.csv'}")
    print(f"Wrote all metrics: {args.output_root / 'all_metrics.csv'}")
    if not metrics.empty:
        print(metrics.loc[:, ["network_method", "pij_method", "organ", "lower_layer", "upper_layer", "time_pair", "EI_gain", "DI", "TE", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
