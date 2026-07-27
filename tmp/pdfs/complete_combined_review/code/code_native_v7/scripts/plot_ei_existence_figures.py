#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.visualization.ei_existence import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LEVEL_PAIRS,
    DEFAULT_NETWORK_METHOD,
    DEFAULT_ORGAN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIJ_METHOD,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SLICE_ROOT,
    DEFAULT_TIME_POINTS,
    SpatialOrientation,
    generate_ei_existence_figures,
    parse_level_pairs,
    print_paths,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot EI existence figures from local MIGNet vertical ablation data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--slice-root", type=Path, default=DEFAULT_SLICE_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--organ", default=DEFAULT_ORGAN)
    parser.add_argument("--network-method", default=DEFAULT_NETWORK_METHOD)
    parser.add_argument("--pij-method", default=DEFAULT_PIJ_METHOD)
    parser.add_argument("--cluster-method", default="seurat", help="Reserved for output metadata and compatibility.")
    parser.add_argument("--time-points", nargs="+", default=list(DEFAULT_TIME_POINTS))
    parser.add_argument("--level-pairs", nargs="+", default=list(DEFAULT_LEVEL_PAIRS))
    parser.add_argument("--timeline-layer", default="seurat_k40", help="Reserved for backward-compatible CLI calls.")
    parser.add_argument("--swap-xy", action="store_true", help="Swap x/y spatial coordinates before plotting.")
    parser.add_argument("--invert-x", action="store_true", help="Reflect x coordinates before plotting.")
    parser.add_argument("--invert-y", action="store_true", help="Reflect y coordinates before plotting.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = generate_ei_existence_figures(
        data_root=args.data_root,
        slice_root=args.slice_root,
        result_root=args.result_root,
        output_dir=args.output_dir,
        organ=args.organ,
        network_method=args.network_method,
        pij_method=args.pij_method,
        time_points=args.time_points,
        level_pairs=parse_level_pairs(args.level_pairs),
        orientation=SpatialOrientation(
            swap_xy=args.swap_xy,
            invert_x=args.invert_x,
            invert_y=args.invert_y,
        ),
        dpi=args.dpi,
    )
    print_paths(paths)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
