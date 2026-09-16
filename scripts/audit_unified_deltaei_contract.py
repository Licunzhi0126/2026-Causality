#!/usr/bin/env python3
from __future__ import annotations

"""Audit that formal downstream DeltaEI exactly reproduces full-model training."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 3 methods x 3 pairs against PIJ_micro_train.npy, "
            "PIJ_macro_train.npy, summary.json, and formal metrics.csv."
        )
    )
    parser.add_argument("--full-cache-root", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from mignet_ce.downstream.analysis.deltaei_contract import audit_full_cache_root

    table = audit_full_cache_root(
        args.full_cache_root,
        args.metrics_csv,
        tolerance=args.tolerance,
    )
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
    columns = [
        "mapping",
        "time_pair",
        "epochs",
        "best_epoch",
        "K",
        "nmf_max_iter_used",
        "cache_protocol",
        "hardK_t",
        "hardK_tp",
        "Keff_t",
        "Keff_tp",
        "summary_EI_micro",
        "recomputed_EI_micro",
        "summary_EI_macro",
        "recomputed_EI_macro",
        "summary_delta_EI",
        "downstream_delta_EI",
        "EI_micro_abs_error",
        "EI_macro_abs_error",
        "delta_EI_abs_error",
        "downstream_delta_EI_abs_error",
        "passed",
    ]
    print(table[columns].to_string(index=False))
    return 0 if bool(table["passed"].all()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
