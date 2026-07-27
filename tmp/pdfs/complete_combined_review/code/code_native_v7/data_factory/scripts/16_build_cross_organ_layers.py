#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cross_organ_layer_builder import CROSS_ORGAN_LAYER_SPECS, build_cross_organ_layers
from factory_common import FACTORY_OUTPUT_ROOT, ORGANS, STAGES


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage-level cross-organ domain h5ad files.")
    parser.add_argument("--input-root", type=Path, default=FACTORY_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=FACTORY_OUTPUT_ROOT / "cross_organ")
    parser.add_argument("--layers", nargs="+", default=["seurat_k40", "louvain_k150"], choices=sorted(CROSS_ORGAN_LAYER_SPECS))
    parser.add_argument("--organs", nargs="+", default=list(ORGANS))
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = build_cross_organ_layers(
        input_root=args.input_root,
        output_root=args.output_root,
        layers=args.layers,
        organs=args.organs,
        stages=args.stages,
        overwrite=args.overwrite,
        manifest_root=args.manifest_root,
    )
    written = sum(1 for row in rows if row.get("status") == "written")
    skipped = sum(1 for row in rows if str(row.get("status", "")).endswith("_skipped"))
    print(f"[CrossOrgan] rows={len(rows)} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
