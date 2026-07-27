#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from commot_layer_runner import build_argparser as build_commot_argparser
from commot_layer_runner import run_commot_layer
from cross_organ_layer_builder import CROSS_ORGAN_LAYER_SPECS
from factory_common import FACTORY_OUTPUT_ROOT


def build_argparser() -> argparse.ArgumentParser:
    parser = build_commot_argparser(
        description="Run official COMMOT CCI on cross-organ domain h5ad files.",
        default_input=FACTORY_OUTPUT_ROOT / "cross_organ",
        default_output=FACTORY_OUTPUT_ROOT / "cci" / "cross_organ",
        unit_kind="domain",
    )
    parser.add_argument("--layers", nargs="+", default=["seurat_k40", "louvain_k150"], choices=sorted(CROSS_ORGAN_LAYER_SPECS))
    parser.add_argument("--manifest-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest_root = args.manifest_root if args.manifest_root is not None else args.input_root.parent / "manifests"
    for layer in args.layers:
        run_commot_layer(
            input_root=args.input_root / layer,
            output_root=args.output_root / layer,
            unit_kind=args.unit_kind,
            manifest_name=f"cci_manifest_cross_organ_{layer}.csv",
            commot_reference_dir=args.commot_reference_dir,
            min_units=args.min_units,
            dis_thr=args.dis_thr,
            sample_names=args.sample_names,
            workers=args.workers,
            lr_chunk_size=args.lr_chunk_size,
            heartbeat_seconds=args.heartbeat_seconds,
            manifest_root=manifest_root,
        )


if __name__ == "__main__":
    main()
