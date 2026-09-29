#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from factory_common import FACTORY_OUTPUT_ROOT, ensure_dir, iter_h5ad_files, parse_sample_stem, write_csv


def choose_count_matrix(adata: ad.AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def aggregate_one(path: Path, output_root: Path, overwrite: bool) -> dict:
    organ, stage = parse_sample_stem(path.stem, path.parent.name)
    out_dir = output_root / organ
    ensure_dir(out_dir)
    out_path = out_dir / f"organ_{organ}_{stage}.h5ad"
    row = {
        "input_file": str(path),
        "output_file": str(out_path),
        "organ": organ,
        "stage": stage,
        "status": "planned",
    }
    if out_path.exists() and not overwrite:
        row["status"] = "exists_skipped"
        return row

    spot = ad.read_h5ad(path)
    if spot.n_obs == 0:
        row["status"] = "empty_skipped"
        return row
    counts = choose_count_matrix(spot)
    if sp.issparse(counts):
        aggregated = counts.sum(axis=0)
        aggregated = sp.csr_matrix(aggregated)
    else:
        aggregated = np.asarray(counts).sum(axis=0, keepdims=True)

    obs = pd.DataFrame(
        {
            "domain_id": ["domain_001"],
            "domain_label": ["organ_domain"],
            "organ": [organ],
            "stage": [stage],
            "spot_count": [int(spot.n_obs)],
        },
        index=pd.Index(["domain_001"], name="domain_id"),
    )
    domain = ad.AnnData(X=aggregated, obs=obs, var=spot.var.copy())
    domain.layers["count"] = aggregated.copy() if hasattr(aggregated, "copy") else aggregated
    if "spatial" in spot.obsm:
        domain.obsm["spatial"] = np.asarray(spot.obsm["spatial"], dtype=np.float32).mean(axis=0, keepdims=True)
    domain.uns["X_name"] = "counts"
    domain.uns["domain_label"] = ["organ_domain"]
    domain.uns["source_spot_file"] = str(path)
    domain.write_h5ad(out_path)

    assignments = pd.DataFrame(
        {
            "spot_id": spot.obs_names.astype(str),
            "domain_id": "domain_001",
            "domain_label": "organ_domain",
            "organ": organ,
            "stage": stage,
        }
    )
    if "annotation" in spot.obs.columns:
        assignments["annotation"] = spot.obs["annotation"].astype(str).to_numpy()
    if "spatial" in spot.obsm:
        spatial = np.asarray(spot.obsm["spatial"])
        assignments["x"] = spatial[:, 0]
        assignments["y"] = spatial[:, 1]
    assignments.to_csv(out_dir / f"organ_{organ}_{stage}_spot_domain_map.csv", index=False)

    row.update({"status": "written", "n_spots": int(spot.n_obs), "n_domains": 1})
    print(f"Wrote {out_path}")
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one-organ-one-domain h5ad files.")
    parser.add_argument("--spot-root", type=Path, default=FACTORY_OUTPUT_ROOT / "spot")
    parser.add_argument("--output-root", type=Path, default=FACTORY_OUTPUT_ROOT / "organ")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = [aggregate_one(path, args.output_root, args.overwrite) for path in iter_h5ad_files(args.spot_root)]
    manifest = args.output_root.parent / "manifests" / "domain_manifest_organ.csv"
    write_csv(manifest, rows)
    print(f"Wrote organ-domain manifest: {manifest}")


if __name__ == "__main__":
    main()
