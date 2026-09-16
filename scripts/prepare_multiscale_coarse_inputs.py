#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.io.multiscale_coarse_inputs import (  # noqa: E402
    DEFAULT_SCALES,
    DEFAULT_TIME_POINTS,
    InputRoots,
    prepare_multiscale_inputs,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate multiscale coarse inputs and stage missing CCI indexes.")
    parser.add_argument("--spot-root", type=Path, required=True)
    parser.add_argument("--seurat-k40-root", type=Path, required=True)
    parser.add_argument("--seurat-k150-root", type=Path, required=True)
    parser.add_argument("--cci-root", type=Path, required=True)
    parser.add_argument("--grn-root", type=Path, required=True)
    parser.add_argument("--developmental-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--time-points", nargs="+", default=list(DEFAULT_TIME_POINTS))
    parser.add_argument("--scales", nargs="+", choices=list(DEFAULT_SCALES), default=list(DEFAULT_SCALES))
    parser.add_argument("--overwrite-prepared", action="store_true")
    return parser


def roots_from_args(args: argparse.Namespace) -> InputRoots:
    return InputRoots(
        spot_root=args.spot_root,
        seurat_k40_root=args.seurat_k40_root,
        seurat_k150_root=args.seurat_k150_root,
        cci_root=args.cci_root,
        grn_root=args.grn_root,
        developmental_root=args.developmental_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    prepared = prepare_multiscale_inputs(
        roots_from_args(args),
        out_root=args.out_root,
        organ=args.organ,
        time_points=args.time_points,
        scales=args.scales,
        overwrite=args.overwrite_prepared,
    )
    print(f"Prepared and validated {len(prepared)} scale/stage inputs under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
