#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from commot_layer_runner import (
    DEFAULT_COMMOT_WORKERS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_LR_CHUNK_SIZE,
    run_commot_layer,
)
from factory_common import COMMOT_REFERENCE_DIR, FACTORY_OUTPUT_ROOT
from layer_specs import DOMAIN_LAYER_SPECS, get_domain_layer_spec


def build_argparser(default_layer: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official COMMOT CCI on a configured domain layer.")
    parser.add_argument("--layer", choices=sorted(DOMAIN_LAYER_SPECS), default=default_layer, required=default_layer is None)
    parser.add_argument("--input-root", "--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--commot-reference-dir", type=Path, default=COMMOT_REFERENCE_DIR)
    parser.add_argument("--unit-kind", choices=("spot", "domain"), default=None)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--dis-thr", type=float, default=200.0)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=DEFAULT_COMMOT_WORKERS)
    parser.add_argument("--lr-chunk-size", type=int, default=DEFAULT_LR_CHUNK_SIZE)
    parser.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    return parser


def run_layer(args: argparse.Namespace) -> None:
    spec = get_domain_layer_spec(args.layer)
    input_root = args.input_root or FACTORY_OUTPUT_ROOT / spec.output_name
    output_root = args.output_root or FACTORY_OUTPUT_ROOT / "cci" / spec.output_name
    manifest_name = args.manifest_name or spec.cci_manifest
    unit_kind = args.unit_kind or spec.unit_kind
    run_commot_layer(
        input_root=input_root,
        output_root=output_root,
        unit_kind=unit_kind,
        manifest_name=manifest_name,
        commot_reference_dir=args.commot_reference_dir,
        min_units=args.min_units,
        dis_thr=args.dis_thr,
        sample_names=args.sample_names,
        workers=args.workers,
        lr_chunk_size=args.lr_chunk_size,
        heartbeat_seconds=args.heartbeat_seconds,
    )


def main() -> None:
    run_layer(build_argparser().parse_args())


if __name__ == "__main__":
    main()
