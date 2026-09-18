#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.asset import AssetConfig, build_paper_assets
from mignet_ce.downstream.asset.demo import build_demo_assets
from mignet_ce.downstream.asset.registry import ASSET_NAMES


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper tables/figures from persisted PIJ and coarse-graining outputs."
    )
    parser.add_argument("--vertical-ablation-root", type=Path)
    parser.add_argument("--coarse-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--slice-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--network-method", default="light_cci_grn")
    parser.add_argument("--primary-pij-method", default="NG_KLot")
    parser.add_argument("--figure-pair", default="12.5->13.5")
    parser.add_argument("--coarse-scale", default="spot", choices=["spot", "seurat_k150", "seurat_k40"])
    parser.add_argument("--coarse-k", type=int, default=40)
    parser.add_argument("--coarse-seed", type=int, default=42)
    parser.add_argument("--primary-coarse-method", default="complete_combined_coarse_maturity_cci_grn")
    parser.add_argument("--asset", action="append", choices=[*ASSET_NAMES, "all"], default=None)
    parser.add_argument("--allow-missing", action="store_true", help="Render missing table cells as '-' instead of failing.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--demo", action="store_true", help="Generate style-only preview with deterministic mock values.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.demo:
        result = build_demo_assets(args.output_root, dpi=args.dpi)
    else:
        required = {
            "--vertical-ablation-root": args.vertical_ablation_root,
            "--coarse-root": args.coarse_root,
            "--data-root": args.data_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"Missing required arguments outside --demo: {', '.join(missing)}")
        cfg = AssetConfig(
            vertical_ablation_root=args.vertical_ablation_root,
            coarse_root=args.coarse_root,
            data_root=args.data_root,
            slice_root=args.slice_root,
            output_root=args.output_root,
            organ=args.organ,
            network_method=args.network_method,
            primary_pij_method=args.primary_pij_method,
            figure_pair=args.figure_pair,
            coarse_scale=args.coarse_scale,
            coarse_k=args.coarse_k,
            coarse_seed=args.coarse_seed,
            primary_coarse_method=args.primary_coarse_method,
            strict=not args.allow_missing,
            dpi=args.dpi,
        )
        result = build_paper_assets(cfg, assets=args.asset)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
