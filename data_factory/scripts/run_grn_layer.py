#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from factory_common import FACTORY_OUTPUT_ROOT
from grn_layer_runner import DEFAULT_N_TREES, DEFAULT_THREADS, DEFAULT_TOP_EDGE_COUNT, DEFAULT_TOP_HVG, run_grn_layer
from layer_specs import DOMAIN_LAYER_SPECS, get_domain_layer_spec


def build_argparser(default_layer: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GENIE3-style GRN on a configured domain layer.")
    parser.add_argument("--layer", choices=sorted(DOMAIN_LAYER_SPECS), default=default_layer, required=default_layer is None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--n-trees", type=int, default=DEFAULT_N_TREES)
    parser.add_argument("--top-hvg", type=int, default=DEFAULT_TOP_HVG)
    parser.add_argument("--top-edge-count", type=int, default=DEFAULT_TOP_EDGE_COUNT)
    parser.add_argument("--tf-list", type=Path, default=None)
    return parser


def run_layer(args: argparse.Namespace) -> None:
    spec = get_domain_layer_spec(args.layer)
    input_root = args.input_root or FACTORY_OUTPUT_ROOT / spec.output_name
    output_root = args.output_root or FACTORY_OUTPUT_ROOT / "grn" / spec.output_name
    manifest_name = args.manifest_name or spec.grn_manifest
    run_grn_layer(
        input_root=input_root,
        output_root=output_root,
        manifest_name=manifest_name,
        min_units=args.min_units,
        sample_names=args.sample_names,
        threads=args.threads,
        n_trees=args.n_trees,
        top_hvg=args.top_hvg,
        top_edge_count=args.top_edge_count,
        tf_list=args.tf_list,
    )


def main() -> None:
    run_layer(build_argparser().parse_args())


if __name__ == "__main__":
    main()
