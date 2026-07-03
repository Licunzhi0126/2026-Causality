#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, NETWORK_METHODS, PAIR_PRESETS, PIJ_METHODS, TemporalRunConfig, VerticalPairSpec
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vertical MIGNet EI/DI/TE for configured domain layer pairs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument(
        "--pair-preset",
        choices=sorted(PAIR_PRESETS),
        default="legacy_mixed_adjacent",
        help="Named vertical pair set. Ignored when --level-pairs is provided.",
    )
    parser.add_argument(
        "--level-pairs",
        nargs="+",
        default=None,
        help="Pairs like spot:louvain_less_than5 louvain_less_than5:louvain_k150 louvain_k150:seurat_k40.",
    )
    parser.add_argument("--expr-threshold", type=float, default=0.0)
    parser.add_argument("--cci-min", type=float, default=0.0)
    parser.add_argument("--top-k-targets-per-regulator", type=int, default=20)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-seed", type=int, default=42)
    parser.add_argument("--pij-method", choices=sorted(PIJ_METHODS), default=None)
    parser.add_argument("--embedding-method", choices=["joint_nmf", "laplacian"], default=None, help="Deprecated alias for --pij-method.")
    parser.add_argument("--network-method", choices=sorted(NETWORK_METHODS), default="legacy_mixed_grn_cci")
    parser.add_argument(
        "--grn-expression-weight-mode",
        choices=["none", "geometric_mean", "product", "min"],
        default="geometric_mean",
    )
    parser.add_argument(
        "--grn-expression-transform",
        choices=["log1p_minmax", "log1p_zscore", "none"],
        default="log1p_minmax",
    )
    parser.add_argument("--grn-expression-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--unit-grn-fallback",
        choices=["error", "sample_grn_masked", "sample_grn_expression_weighted", "skip_unit_intra"],
        default="sample_grn_expression_weighted",
    )
    parser.add_argument("--cross-cell-lr-use-grn-gate", action="store_true")
    parser.add_argument("--cross-cell-top-k-edges", type=int, default=1000)
    parser.add_argument(
        "--export-pij",
        action="store_true",
        help="Export full PIJ transition matrices as sparse .npz files with units.csv and metadata.",
    )
    parser.add_argument(
        "--export-pij-topk",
        type=int,
        default=0,
        help="Optional top-k CSV for inspection. 0 means only export full sparse .npz matrices.",
    )
    parser.add_argument(
        "--pij-archive-root",
        type=Path,
        default=None,
        help="PIJ sparse archive root. Defaults to DATA_ROOT/pij.",
    )
    parser.add_argument(
        "--export-pair-artifacts",
        action="store_true",
        help="Export pair-level debugging artifacts. This does not control PIJ export.",
    )
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
    parser.add_argument("--no-pure-expression-normalize", action="store_true")
    parser.add_argument("--no-pure-expression-log1p", action="store_true")
    parser.add_argument("--pure-expression-scale-factor", type=float, default=10000.0)
    parser.add_argument("--pure-expression-max-genes", type=int, default=2000)
    parser.add_argument("--pure-expression-gene-selection", choices=["variance", "all"], default="variance")
    parser.add_argument("--pure-expression-pca-components", type=int, default=None)
    parser.add_argument("--pure-expression-scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--pure-expression-cosine-eps", type=float, default=1e-8)
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
    parser.add_argument("--export-raw-native-features", action="store_true")
    parser.add_argument("--export-graphs", action="store_true")
    parser.add_argument("--export-feature-diagnostics", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Global worker budget inside one organ/pair. Pair parallelism remains disabled.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.pij_method and args.embedding_method and args.pij_method != args.embedding_method:
        parser.error("--pij-method and --embedding-method were both provided with different values.")
    pij_method = args.pij_method or args.embedding_method or "joint_nmf"
    embedding_method = pij_method if pij_method in {"joint_nmf", "laplacian"} else "joint_nmf"
    level_pairs = (
        [VerticalPairSpec.parse(value) for value in args.level_pairs]
        if args.level_pairs is not None
        else list(PAIR_PRESETS[args.pair_preset])
    )
    cfg = TemporalRunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=args.organs,
        time_points=args.time_points,
        level_pairs=level_pairs,
        expr_threshold=args.expr_threshold,
        cci_min=args.cci_min,
        top_k_targets_per_regulator=args.top_k_targets_per_regulator,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        nmf_seed=args.nmf_seed,
        embedding_method=embedding_method,
        pij_method=pij_method,
        network_method=args.network_method,
        grn_expression_weight_mode=args.grn_expression_weight_mode,
        grn_expression_transform=args.grn_expression_transform,
        grn_expression_weight_floor=args.grn_expression_weight_floor,
        unit_grn_fallback=args.unit_grn_fallback,
        cross_cell_lr_use_grn_gate=args.cross_cell_lr_use_grn_gate,
        cross_cell_top_k_edges=args.cross_cell_top_k_edges,
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
        pure_expression_normalize=not args.no_pure_expression_normalize,
        pure_expression_log1p=not args.no_pure_expression_log1p,
        pure_expression_scale_factor=args.pure_expression_scale_factor,
        pure_expression_max_genes=args.pure_expression_max_genes,
        pure_expression_gene_selection=args.pure_expression_gene_selection,
        pure_expression_pca_components=args.pure_expression_pca_components,
        pure_expression_scaler=args.pure_expression_scaler,
        pure_expression_cosine_eps=args.pure_expression_cosine_eps,
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
        export_graphs=args.export_graphs,
        export_raw_native_features=args.export_raw_native_features,
        export_feature_diagnostics=args.export_feature_diagnostics,
        max_workers=args.max_workers,
    )
    metrics = VerticalMIGNetPipeline(cfg).run()
    if metrics.empty:
        print(f"No metrics were produced. Inspect {cfg.output_root / 'run_summary.csv'}")
    else:
        print(metrics.loc[:, ["network_method", "pij_method", "organ", "lower_layer", "upper_layer", "time_pair", "EI_lower", "EI_upper", "EI_gain", "DI", "TE"]].to_string(index=False))
        print(f"Wrote metrics: {cfg.output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
