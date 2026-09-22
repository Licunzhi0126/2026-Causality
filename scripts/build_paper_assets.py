#!/usr/bin/env python3
from __future__ import annotations

"""The single active entry point for paper tables and figures."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.paper_assets import AssetConfig, build_paper_assets  # noqa: E402
from mignet_ce.downstream.paper_assets.demo import build_demo_assets  # noqa: E402
from mignet_ce.downstream.paper_assets.registry import ASSET_NAMES, normalize_asset_names  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and render publication assets from saved scientific results.")
    parser.add_argument("--stage", choices=("prepare", "render", "all"), default="all")
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument(
        "--feature-ablation-root",
        type=Path,
        help=(
            "Controlled feature-ablation output containing feature_ablation_long.csv; "
            "when omitted, Table 1 uses the legacy ablation fallback."
        ),
    )
    parser.add_argument("--k10-ablation-root", type=Path)
    parser.add_argument("--ei-metrics", type=Path)
    parser.add_argument("--seurat-cg-metrics", type=Path)
    parser.add_argument("--legacy-coarse-root", type=Path)
    parser.add_argument("--multiscale-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--optimal-method",
        action="append",
        default=[],
        help="Repeat to generate independent multiscale optimal assets for multiple methods.",
    )
    parser.add_argument(
        "--primary-coarse-method",
        default="complete_combined_coarse_maturity_cci_grn",
        help="Legacy Table 3/Figure 2 method and target of backward-compatible optimal aliases.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--slice-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--figure-pair", default="12.5->13.5")
    parser.add_argument("--legacy-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", action="append", choices=(*ASSET_NAMES, "all"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--demo", action="store_true", help="Style preview with marked mock values.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        result = build_demo_assets(args.output_root, dpi=args.dpi, data_root=args.data_root)
    else:
        requested = normalize_asset_names(args.asset)
        preparing = args.stage in {"prepare", "all"}
        rendering = args.stage in {"render", "all"}
        required = {
            "--ablation-root": preparing and (
                (
                    args.feature_ablation_root is None
                    and bool(set(requested) & {"table1", "table1_k10"})
                )
                or (args.ei_metrics is None and bool(set(requested) & {"table2", "table2_k10", "figure1", "figure1_k10"}))
            ),
            "--legacy-coarse-root": preparing and bool(set(requested) & {"table3", "figure2"}),
            "--multiscale-root": preparing and bool(
                set(requested) & {"table4", "table5", "figure2_optimal"}
            ),
            "--data-root": rendering and bool(
                set(requested) & {"figure1", "figure1_k10", "figure2", "figure2_optimal"}
            ),
        }
        values = {
            "--ablation-root": args.ablation_root,
            "--legacy-coarse-root": args.legacy_coarse_root,
            "--multiscale-root": args.multiscale_root,
            "--data-root": args.data_root,
        }
        missing = [name for name, needed in required.items() if needed and not values[name]]
        if missing:
            raise SystemExit(f"Missing required paths: {', '.join(missing)}")
        cfg = AssetConfig(
            vertical_ablation_root=args.ablation_root or Path("."),
            feature_ablation_root=args.feature_ablation_root,
            k10_ablation_root=args.k10_ablation_root,
            ei_metrics_path=args.ei_metrics,
            seurat_cg_metrics_path=args.seurat_cg_metrics,
            coarse_root=args.legacy_coarse_root or Path("."),
            multiscale_roots=tuple(args.multiscale_root),
            optimal_coarse_methods=tuple(args.optimal_method),
            data_root=args.data_root or Path("."),
            slice_root=args.slice_root,
            output_root=args.output_root,
            organ=args.organ,
            figure_pair=args.figure_pair,
            primary_coarse_method=args.primary_coarse_method,
            coarse_k=args.legacy_k,
            coarse_seed=args.seed,
            dpi=args.dpi,
        )
        result = build_paper_assets(cfg, assets=args.asset, stage=args.stage)
    if "prepare" in result:
        prepared = result["prepare"]
        result["prepare"] = {
            "tables": prepared["tables"],
            "optimal_methods": prepared.get("optimal_methods", {}),
            "manifest": str(args.output_root / "audit" / "paper_assets_prepare_manifest.json"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
