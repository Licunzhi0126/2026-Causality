#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from grn_layer_runner import DEFAULT_N_TREES, DEFAULT_THREADS, DEFAULT_TOP_EDGE_COUNT, DEFAULT_TOP_HVG
from spot_local_grn_runner import run_spot_local_grn_pilot


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected center-cell local GRNs using pure spatial KNN neighborhoods."
    )
    parser.add_argument("--spot-h5ad", type=Path, required=True)
    parser.add_argument("--selected-units", nargs="+", required=True)
    parser.add_argument("--neighbor-mode", choices=["spatial"], default="spatial")
    parser.add_argument("--k-neighbors", type=int, default=50)
    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument("--include-center", dest="include_center", action="store_true")
    center_group.add_argument("--exclude-center", dest="include_center", action="store_false")
    parser.set_defaults(include_center=True)
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--n-trees", type=int, default=DEFAULT_N_TREES)
    parser.add_argument("--top-hvg", type=int, default=DEFAULT_TOP_HVG)
    parser.add_argument("--top-edge-count", type=int, default=DEFAULT_TOP_EDGE_COUNT)
    parser.add_argument("--tf-list", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_spot_local_grn_pilot(
        spot_h5ad=args.spot_h5ad,
        selected_units=args.selected_units,
        output_root=args.output_root,
        k_neighbors=args.k_neighbors,
        include_center=args.include_center,
        min_cells=args.min_cells,
        threads=args.threads,
        n_trees=args.n_trees,
        top_hvg=args.top_hvg,
        top_edge_count=args.top_edge_count,
        tf_list=args.tf_list,
    )


if __name__ == "__main__":
    main()
