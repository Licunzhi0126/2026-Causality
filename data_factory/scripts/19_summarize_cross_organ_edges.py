#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cross_organ_layer_builder import CROSS_ORGAN_LAYER_SPECS
from factory_common import FACTORY_OUTPUT_ROOT, ensure_dir, write_csv


def _read_index(path: Path) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return []
    return df.iloc[:, 0].astype(str).tolist()


def summarize_one(h5ad_path: Path, cci_root: Path, layer: str) -> Dict[str, object]:
    sample = h5ad_path.stem
    matrix_path = cci_root / layer / f"{sample}_CCI_total.npz"
    index_path = cci_root / layer / f"{sample}_index.tsv"
    row: Dict[str, object] = {
        "layer": layer,
        "sample": sample,
        "h5ad_file": str(h5ad_path),
        "cci_total": str(matrix_path),
        "status": "planned",
    }
    if not matrix_path.exists() or not index_path.exists():
        row["status"] = "missing_cci"
        row["reason"] = "Missing CCI total matrix or index file."
        return row

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        organ_by_unit = adata.obs["organ"].astype(str).to_dict()
        n_units = int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    index_names = _read_index(index_path)
    mat = sp.load_npz(matrix_path).tocoo()
    if mat.shape[0] != len(index_names) or mat.shape[1] != len(index_names):
        row["status"] = "shape_mismatch"
        row["reason"] = f"matrix_shape={mat.shape}, index_len={len(index_names)}"
        return row

    src_organs = np.array([organ_by_unit.get(index_names[int(i)], "UNKNOWN") for i in mat.row], dtype=object)
    dst_organs = np.array([organ_by_unit.get(index_names[int(i)], "UNKNOWN") for i in mat.col], dtype=object)
    same_unit = mat.row == mat.col
    same_organ = src_organs == dst_organs
    inter = ~same_organ

    row.update(
        {
            "status": "summarized",
            "n_units": n_units,
            "matrix_shape": f"{mat.shape[0]}x{mat.shape[1]}",
            "total_nnz": int(mat.nnz),
            "self_unit_nnz": int(same_unit.sum()),
            "intra_organ_nnz": int(same_organ.sum()),
            "inter_organ_nnz": int(inter.sum()),
            "inter_organ_fraction": float(inter.sum() / mat.nnz) if mat.nnz else 0.0,
            "total_score": float(mat.data.sum()) if mat.nnz else 0.0,
            "intra_organ_score": float(mat.data[same_organ].sum()) if mat.nnz else 0.0,
            "inter_organ_score": float(mat.data[inter].sum()) if mat.nnz else 0.0,
        }
    )
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize intra-organ and inter-organ CCI in cross-organ COMMOT outputs.")
    parser.add_argument("--cross-organ-root", type=Path, default=FACTORY_OUTPUT_ROOT / "cross_organ")
    parser.add_argument("--cci-root", type=Path, default=FACTORY_OUTPUT_ROOT / "cci" / "cross_organ")
    parser.add_argument("--layers", nargs="+", default=["seurat_k40", "louvain_k150"], choices=sorted(CROSS_ORGAN_LAYER_SPECS))
    parser.add_argument("--output", type=Path, default=FACTORY_OUTPUT_ROOT / "manifests" / "cross_organ_cci_intra_inter_summary.csv")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows: List[Dict[str, object]] = []
    for layer in args.layers:
        for h5ad_path in sorted((args.cross_organ_root / layer).glob("*.h5ad")):
            rows.append(summarize_one(h5ad_path, args.cci_root, layer))
    ensure_dir(args.output.parent)
    write_csv(args.output, rows)
    print(f"[CrossOrgan] Wrote CCI intra/inter summary: {args.output}")


if __name__ == "__main__":
    main()
