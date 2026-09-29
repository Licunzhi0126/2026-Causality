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
from factory_common import FACTORY_OUTPUT_ROOT, ORGAN_LABELS, ORGANS, RAW_E1S1_ROOT, STAGES, raw_stage_path, write_csv


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check raw E1S1 h5ad inputs before building organ/domain data.")
    parser.add_argument("--input-root", type=Path, default=RAW_E1S1_ROOT)
    parser.add_argument("--output-root", type=Path, default=FACTORY_OUTPUT_ROOT)
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = []

    for stage in args.stages:
        path = raw_stage_path(args.input_root, stage)
        row = {
            "stage": stage,
            "input_file": str(path),
            "exists": path.exists(),
            "n_obs": "",
            "n_vars": "",
            "has_annotation": False,
            "has_spatial": False,
            "has_count_layer": False,
            "error": "",
        }
        for organ in ORGANS:
            row[f"{organ}_spots"] = ""

        if not path.exists():
            row["error"] = "missing input file"
            rows.append(row)
            continue

        adata = ad.read_h5ad(path, backed="r")
        try:
            row["n_obs"] = int(adata.n_obs)
            row["n_vars"] = int(adata.n_vars)
            row["has_annotation"] = "annotation" in adata.obs.columns
            row["has_spatial"] = "spatial" in adata.obsm
            row["has_count_layer"] = "count" in adata.layers or "counts" in adata.layers

            if "annotation" not in adata.obs.columns:
                row["error"] = "missing obs['annotation']"
            else:
                labels = adata.obs["annotation"].astype(str)
                for organ, accepted in ORGAN_LABELS.items():
                    row[f"{organ}_spots"] = int(labels.isin(accepted).sum())
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()

        rows.append(row)

    out_path = args.output_root / "manifests" / "input_check.csv"
    write_csv(out_path, rows)
    print(f"Wrote input check manifest: {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
