#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.visualization.downstream import (
    DownstreamConfig,
    render_downstream_figures,
    run_downstream_analysis,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1_domain_factory"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "downstream_six_panel"
DEFAULT_TIMES = ("11.5", "12.5", "13.5", "14.5")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or re-render the causal-emergence downstream six-panel figure suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Compute tables and render all ten six-panel figures.")
    analyze.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    analyze.add_argument("--metrics-csv", type=Path, required=True)
    analyze.add_argument("--pair-archive", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    analyze.add_argument("--organ", default="heart")
    analyze.add_argument("--time-points", nargs=4, default=list(DEFAULT_TIMES))
    analyze.add_argument("--network-method", default="light_cci_grn")
    analyze.add_argument("--pij-method", default="NG_KLot")
    analyze.add_argument("--lower-layer", default="seurat_k150")
    analyze.add_argument("--upper-layer", default="seurat_k40")
    analyze.add_argument("--random-repeats", type=int, default=500)
    analyze.add_argument("--random-seed", type=int, default=20260731)
    analyze.add_argument("--spatial-knn", type=int, default=6)
    analyze.add_argument("--perturb-random-repeats", type=int, default=200)

    render = subparsers.add_parser("render", help="Re-render all figures from existing result tables.")
    render.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    render.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    render.add_argument("--organ", default="heart")
    render.add_argument("--time-points", nargs=4, default=list(DEFAULT_TIMES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "analyze":
        outputs = run_downstream_analysis(
            DownstreamConfig(
                data_root=args.data_root,
                metrics_csv=args.metrics_csv,
                pair_archive=args.pair_archive,
                output_dir=args.output_dir,
                organ=args.organ,
                times=tuple(map(str, args.time_points)),
                network_method=args.network_method,
                pij_method=args.pij_method,
                lower_layer=args.lower_layer,
                upper_layer=args.upper_layer,
                random_repeats=args.random_repeats,
                random_seed=args.random_seed,
                spatial_knn=args.spatial_knn,
                perturb_random_repeats=args.perturb_random_repeats,
            )
        )
    else:
        outputs = render_downstream_figures(
            args.results_dir,
            args.data_root,
            organ=args.organ,
            times=tuple(map(str, args.time_points)),
        )
    print(f"Downstream figures: {outputs['figures_dir']}")
    print(f"Manifest: {outputs['manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
