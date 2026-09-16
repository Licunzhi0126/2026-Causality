#!/usr/bin/env python3
"""Run the frozen-assignment GRN perturbation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.analysis.config import UnifiedDownstreamConfig
PERTURBATION_METHOD_CHOICES = (
    "complete_combined_coarse_maturity_cci_grn",
    "maturity_cci_grn_two_stage",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Spot-to-Macro GRN perturbations with a method-specific baseline."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--developmental-feature-root", type=Path, default=None)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--time-points", nargs=4, default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", choices=PERTURBATION_METHOD_CHOICES, default="maturity_cci_grn_two_stage")
    parser.add_argument("--strengths", type=float, nargs="+", default=list(np.arange(0.1, 1.0, 0.1)))
    parser.add_argument("--repeats", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from mignet_ce.downstream.analysis.grn_perturbation.analysis import build_grn_perturbation_table
    from mignet_ce.downstream.analysis.grn_perturbation.baseline import ensure_grn_perturbation_baselines
    from mignet_ce.downstream.analysis.visualization.grn_perturbation.plots import plot_grn_perturbation

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not args.strengths or any(not 0.0 <= value <= 1.0 for value in args.strengths):
        raise ValueError("--strengths values must lie in [0, 1]")
    cfg = UnifiedDownstreamConfig(
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        developmental_feature_root=args.developmental_feature_root,
        organ=args.organ,
        times=tuple(map(str, args.time_points)),
        device=args.device,
    ).normalized()
    cfg.validate()
    root = cfg.output_dir
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Perturbation output directory is not empty: {root}")
    baseline_dir = (
        Path(args.baseline_dir).resolve()
        if args.baseline_dir is not None
        else cfg.full_cache_root / "grn_perturbation_baseline"
    )
    ensure_grn_perturbation_baselines(cfg, baseline_dir, args.method)
    table = build_grn_perturbation_table(
        cfg,
        baseline_dir,
        frontend_method=args.method,
        strengths=tuple(args.strengths),
        repeats=args.repeats,
    )
    summary = (
        table.groupby(["frontend_method", "time_pair", "method", "strength"], as_index=False)
        .agg(
            delta_CE_HV_mean=("delta_CE_HV", "mean"),
            delta_CE_HV_std=("delta_CE_HV", "std"),
            delta_CE_VH_mean=("delta_CE_VH", "mean"),
            path_discrepancy_mean=("path_delta_relative_l1", "mean"),
            closure_response_mean=("delta_closure_js_HV", "mean"),
            repeat_count=("repeat", "nunique"),
        )
    )
    tables_dir, figures_dir, audit_dir = root / "tables", root / "figures", root / "audit"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tables_dir / "12_grn_perturbation_distribution.csv"
    summary_path = tables_dir / "13_grn_perturbation_summary.csv"
    table.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_path = figures_dir / "fig09_grn_perturbation.png"
    plot_grn_perturbation(table, figure_path)
    expected_rows = len(cfg.adjacent_pairs) * 12 * len(args.strengths) * args.repeats
    checks = {
        "selected_frontend_method": bool((table["frontend_method"] == args.method).all()),
        "expected_rows": len(table) == expected_rows,
        "twelve_operators": table["method"].nunique() == 12,
        "finite_numeric_results": bool(np.isfinite(table.select_dtypes(include="number")).all().all()),
        "figure_outputs": figure_path.exists() and figure_path.with_suffix(".pdf").exists(),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"check": name, "passed": passed} for name, passed in checks.items()]
    ).to_csv(audit_dir / "validation_checks.csv", index=False)
    manifest = {
        "frontend_method": args.method,
        "baseline_dir": str(baseline_dir),
        "distribution_table": str(raw_path.relative_to(root)),
        "summary_table": str(summary_path.relative_to(root)),
        "figures": [
            str(figure_path.relative_to(root)),
            str(figure_path.with_suffix(".pdf").relative_to(root)),
        ],
        "strengths": list(map(float, args.strengths)),
        "repeat_count": args.repeats,
        "rows": len(table),
        "validation_passed": all(checks.values()),
    }
    manifest_path = audit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"GRN perturbation audit failed: {checks}")
    print(json.dumps({"table": str(raw_path), "figure": str(figure_path), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
