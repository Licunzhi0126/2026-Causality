#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, DEFAULT_WORK_ROOT
from mignet_ce.io.developmental_feature_builder import (
    DevelopmentalFeatureBuildConfig,
    build_developmental_features,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build spot-level proxy developmental feature CSVs from factory h5ad files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_WORK_ROOT / "output" / "developmental_features")
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument("--mode", choices=["factory_proxy"], default="factory_proxy")
    parser.add_argument("--velocity-components", type=int, default=30)
    parser.add_argument("--pseudotime-within-stage-weight", type=float, default=0.15)
    parser.add_argument("--sr-source", choices=["auto", "obs", "module", "regulon", "expression"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DevelopmentalFeatureBuildConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=args.organs,
        time_points=args.time_points,
        mode=args.mode,
        velocity_components=args.velocity_components,
        pseudotime_within_stage_weight=args.pseudotime_within_stage_weight,
        sr_source=args.sr_source,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
        seed=args.seed,
    )
    result = build_developmental_features(cfg)
    manifest_path = result.output_root / "manifest" / "developmental_features_manifest.csv"
    status_counts = result.manifest["status"].value_counts().to_dict() if not result.manifest.empty else {}
    print(f"Wrote developmental features under {result.output_root}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
