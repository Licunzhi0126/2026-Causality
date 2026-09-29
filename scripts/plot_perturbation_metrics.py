#!/usr/bin/env python3
"""Render GRN perturbation figures from an existing result table only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.analysis.visualization.grn_perturbation.plots import plot_grn_perturbation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot an existing GRN perturbation CSV without recomputation.")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before = _sha256(args.table)
    plot_grn_perturbation(pd.read_csv(args.table), args.output)
    if _sha256(args.table) != before:
        raise RuntimeError("Plotting modified the input analysis table")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
