#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from factory_common import FACTORY_OUTPUT_ROOT
from grn_layer_runner import DEFAULT_N_TREES, DEFAULT_THREADS, DEFAULT_TOP_EDGE_COUNT, DEFAULT_TOP_HVG
from layer_specs import DOMAIN_LAYER_SPECS, get_domain_layer_spec
from unit_grn_layer_runner import run_spot_unit_grn_layer, run_unit_grn_layer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unit-specific GENIE3 GRNs for spot or domain layers."
    )
    parser.add_argument("--layer", choices=sorted([*DOMAIN_LAYER_SPECS, "spot"]), required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--unit-column", default="domain_id")
    parser.add_argument("--min-cells-per-unit", type=int, default=30)
    parser.add_argument("--spot-k-neighbors", type=int, default=50)
    parser.add_argument("--spot-neighbor-mode", choices=["spatial"], default="spatial")
    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument("--include-center", dest="include_center", action="store_true")
    center_group.add_argument("--exclude-center", dest="include_center", action="store_false")
    parser.set_defaults(include_center=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--n-trees", type=int, default=DEFAULT_N_TREES)
    parser.add_argument("--top-hvg", type=int, default=DEFAULT_TOP_HVG)
    parser.add_argument("--top-edge-count", type=int, default=DEFAULT_TOP_EDGE_COUNT)
    parser.add_argument("--tf-list", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.layer == "spot":
        run_spot_unit_grn_layer(
            input_root=args.input_root or FACTORY_OUTPUT_ROOT / "spot",
            output_root=args.output_root or FACTORY_OUTPUT_ROOT / "grn_unit_specific" / "spot",
            manifest_name=args.manifest_name or "unit_grn_manifest_spot.csv",
            sample_names=args.sample_names,
            min_cells_per_unit=args.min_cells_per_unit,
            k_neighbors=args.spot_k_neighbors,
            include_center=args.include_center,
            threads=args.threads,
            n_trees=args.n_trees,
            top_hvg=args.top_hvg,
            top_edge_count=args.top_edge_count,
            tf_list=args.tf_list,
        )
        return

    spec = get_domain_layer_spec(args.layer)
    run_unit_grn_layer(
        input_root=args.input_root or FACTORY_OUTPUT_ROOT / spec.output_name,
        output_root=args.output_root or FACTORY_OUTPUT_ROOT / "grn_unit_specific" / spec.output_name,
        manifest_name=args.manifest_name or f"unit_grn_manifest_{spec.output_name}.csv",
        layer=spec.output_name,
        sample_names=args.sample_names,
        unit_column=args.unit_column,
        min_cells_per_unit=args.min_cells_per_unit,
        threads=args.threads,
        n_trees=args.n_trees,
        top_hvg=args.top_hvg,
        top_edge_count=args.top_edge_count,
        tf_list=args.tf_list,
    )


if __name__ == "__main__":
    main()
