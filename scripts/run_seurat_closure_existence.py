#!/usr/bin/env python3
from __future__ import annotations

"""Create Seurat K150/K40/K10 dynamical-closure evidence tables."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.analysis.dynamic_closure.seurat_existence import (  # noqa: E402
    SEURAT_LAYERS,
    SeuratClosureConfig,
    run_seurat_closure_existence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Seurat K150/K40/K10 closure quality for all six time pairs."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pij-archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--time-points", nargs=4, default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument(
        "--micro-reference-pair",
        choices=[f"spot:{layer}" for layer in SEURAT_LAYERS],
        default="spot:seurat_k150",
    )
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SeuratClosureConfig(
        data_root=args.data_root,
        pij_archive_root=args.pij_archive_root,
        output_root=args.output_root,
        organ=args.organ,
        times=tuple(map(str, args.time_points)),
        micro_reference_layer=args.micro_reference_pair.split(":", 1)[1],
        null_repeats=args.null_repeats,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    outputs = run_seurat_closure_existence(cfg)
    print(json.dumps({name: str(path.resolve()) for name, path in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
