#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cross_organ_layer_builder import CROSS_ORGAN_LAYER_SPECS
from factory_common import FACTORY_OUTPUT_ROOT
from grn_layer_runner import build_argparser as build_grn_argparser
from grn_layer_runner import run_grn_layer


def build_argparser() -> argparse.ArgumentParser:
    parser = build_grn_argparser(
        description="Run GENIE3-style GRN on cross-organ domain h5ad files.",
        default_input=FACTORY_OUTPUT_ROOT / "cross_organ",
        default_output=FACTORY_OUTPUT_ROOT / "grn" / "cross_organ",
    )
    parser.add_argument("--layers", nargs="+", default=["seurat_k40", "louvain_k150"], choices=sorted(CROSS_ORGAN_LAYER_SPECS))
    parser.add_argument("--manifest-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_root = args.manifest_root if args.manifest_root is not None else args.input_root.parent / "manifests"
    for layer in args.layers:
        run_grn_layer(
            input_root=args.input_root / layer,
            output_root=args.output_root / layer,
            manifest_name=f"grn_manifest_cross_organ_{layer}.csv",
            min_units=args.min_units,
            sample_names=args.sample_names,
            threads=args.threads,
            n_trees=args.n_trees,
            top_hvg=args.top_hvg,
            top_edge_count=args.top_edge_count,
            tf_list=args.tf_list,
            manifest_root=manifest_root,
        )


if __name__ == "__main__":
    main()
