#!/usr/bin/env python3
from __future__ import annotations

"""Formal four-representation downstream entry with locked full DeltaEI runs."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.visualization.downstream.config import UnifiedDownstreamConfig
from mignet_ce.visualization.downstream.workflow import run_unified_downstream_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal unified downstream benchmark. DeltaEI is always 2 methods x "
            "3 adjacent pairs at K=40, 1500 epochs, and NMF max_iter=300 for every pair; "
            "reduced preview parameters are not accepted."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--developmental-feature-root", type=Path, default=None)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--time-points", nargs=4, default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--spatial-knn", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = UnifiedDownstreamConfig(
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        developmental_feature_root=args.developmental_feature_root,
        organ=args.organ,
        times=tuple(map(str, args.time_points)),
        device=args.device,
        spatial_knn=args.spatial_knn,
    )
    outputs = run_unified_downstream_analysis(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
