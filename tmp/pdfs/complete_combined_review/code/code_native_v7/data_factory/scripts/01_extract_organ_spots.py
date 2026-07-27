#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from factory_common import (
    FACTORY_OUTPUT_ROOT,
    ORGAN_LABELS,
    ORGANS,
    RAW_E1S1_ROOT,
    STAGES,
    ensure_dir,
    normalize_organ,
    raw_stage_path,
    write_csv,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Heart/Brain/Lung spot h5ad files from raw E1S1 data.")
    parser.add_argument("--input-root", type=Path, default=RAW_E1S1_ROOT)
    parser.add_argument("--output-root", type=Path, default=FACTORY_OUTPUT_ROOT)
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--organs", nargs="+", default=list(ORGANS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    organs = [normalize_organ(organ) for organ in args.organs]
    rows = []

    for stage in args.stages:
        path = raw_stage_path(args.input_root, stage)
        if not path.exists():
            rows.append({"stage": stage, "status": "missing_input", "input_file": str(path)})
            continue

        adata = ad.read_h5ad(path, backed="r")
        try:
            if "annotation" not in adata.obs.columns:
                raise KeyError(f"{path} is missing obs['annotation'].")
            labels = adata.obs["annotation"].astype(str)

            for organ in organs:
                accepted = ORGAN_LABELS[organ]
                mask = labels.isin(accepted).to_numpy()
                out_dir = args.output_root / "spot" / organ
                ensure_dir(out_dir)
                out_path = out_dir / f"spot_{organ}_{stage}.h5ad"

                row = {
                    "stage": stage,
                    "organ": organ,
                    "accepted_annotations": "|".join(accepted),
                    "input_file": str(path),
                    "output_file": str(out_path),
                    "n_spots": int(mask.sum()),
                    "status": "planned",
                }
                if out_path.exists() and not args.overwrite:
                    row["status"] = "exists_skipped"
                    rows.append(row)
                    continue
                if int(mask.sum()) == 0:
                    row["status"] = "empty_skipped"
                    rows.append(row)
                    continue

                sub = adata[mask].to_memory(copy=True)
                sub.obs["factory_organ"] = organ
                sub.obs["factory_stage"] = stage
                sub.obs["factory_annotation_normalized"] = organ
                sub.write_h5ad(out_path)
                row["status"] = "written"
                rows.append(row)
                print(f"Wrote {out_path} ({sub.n_obs} spots)")
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()

    manifest = args.output_root / "manifests" / "extraction_manifest.csv"
    write_csv(manifest, rows)
    print(f"Wrote extraction manifest: {manifest}")


if __name__ == "__main__":
    main()
