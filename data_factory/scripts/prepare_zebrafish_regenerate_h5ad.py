"""Convert published zebrafish regeneration counts and metadata to stage H5ADs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "data_factory" / "lib")]
from species_adapter import ad, sha256, write_new_json

STAGES = {
    "uninjured": "0",
    "6 hpa": "0.25",
    "12 hpa": "0.5",
    "1 dpa": "1",
    "3 dpa": "3",
    "7 dpa": "7",
    "14 dpa": "14",
    "28 dpa": "28",
}
STAGE_DIR = Path("h5ad/by_stage")
QC_PATH = Path("manifests/section_qc.tsv")
MANIFEST_PATH = Path("manifests/preprocessing_manifest.json")


def read_metadata(path):
    obs = pd.read_csv(path, sep="\t", index_col=0, keep_default_na=False, float_precision="round_trip")
    obs.index = pd.Index(obs.index.astype(str), name="spot_id")
    required = {"orig.ident", "nCount_Spatial", "nFeature_Spatial", "x", "y",
                "grid_x", "grid_y", "isolate", "cid", "time_points", "annotation"}
    if not required <= set(obs):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(obs))}")
    if not obs.index.is_unique or (obs.index.str.strip() == "").any():
        raise ValueError("Metadata requires unique, nonempty spot IDs")
    if set(obs["time_points"]) != set(STAGES):
        raise ValueError(f"Unexpected time-point labels: {sorted(set(obs['time_points']))}")
    if not np.isfinite(obs[["x", "y", "grid_x", "grid_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Metadata contains non-finite spatial coordinates")
    section_keys = ["orig.ident", "isolate", "cid", "time_points"]
    if obs[section_keys].apply(lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("Missing section, isolate, chip, or time-point identifier")
    if (obs.groupby("orig.ident")[section_keys[1:]].nunique() != 1).any().any():
        raise ValueError("Each section must belong to a single isolate, chip and time point")
    obs["stage_day"] = obs["time_points"].map(STAGES).astype(float)
    return obs


def read_sparse_counts(directory, obs):
    spots = (directory / "spots.txt").read_text(encoding="utf-8").splitlines()
    genes = pd.Index((directory / "genes.txt").read_text(encoding="utf-8").splitlines(), name="gene_id")
    if not genes.is_unique or not len(genes) or (genes.str.strip() == "").any():
        raise ValueError("Spatial counts require unique, nonempty gene IDs")
    if not pd.Index(spots).equals(obs.index):
        raise ValueError("Spatial counts and metadata spot IDs differ in identity or order")
    data = np.memmap(directory / "counts.i32", mode="r", dtype="<i4")
    indices = np.memmap(directory / "indices.i32", mode="r", dtype="<i4")
    indptr = np.memmap(directory / "indptr.i32", mode="r", dtype="<i4")
    counts = sp.csr_matrix((data, indices, indptr), shape=(len(spots), len(genes)))
    counts.check_format(full_check=True)
    if not counts.has_canonical_format or len(data) != counts.nnz or len(indices) != counts.nnz:
        raise ValueError("Spatial counts contain invalid or duplicate sparse entries")
    return counts, pd.DataFrame({"Gene": genes.to_numpy()}, index=genes)


def verify_stage(path, counts, obs, var, coords):
    restored = ad.read_h5ad(path)
    pd.testing.assert_frame_equal(restored.obs, obs, check_dtype=False, check_categorical=False, check_exact=True)
    pd.testing.assert_frame_equal(restored.var, var, check_dtype=False, check_categorical=False, check_exact=True)
    np.testing.assert_array_equal(restored.obsm["spatial"], coords)
    for matrix in (restored.X, restored.layers["counts"]):
        if not sp.isspmatrix_csr(matrix) or matrix.shape != counts.shape or matrix.dtype != counts.dtype:
            raise ValueError(f"Counts representation changed on H5AD round trip: {path}")
        np.testing.assert_array_equal(matrix.indptr, counts.indptr)
        np.testing.assert_array_equal(matrix.indices, counts.indices)
        np.testing.assert_array_equal(matrix.data, counts.data)
    unit_counts = np.asarray(restored.layers["counts"].sum(axis=1)).ravel()
    np.testing.assert_array_equal(unit_counts, obs["nCount_Spatial"].to_numpy())
    np.testing.assert_array_equal(restored.layers["counts"].getnnz(axis=1), obs["nFeature_Spatial"].to_numpy())
    if np.any(unit_counts <= 0):
        raise ValueError(f"Zero-expression spot in {path}")
    return int(unit_counts.sum())


def write_stages(work, sparse_dir, obs):
    counts, var = read_sparse_counts(sparse_dir, obs)
    (work / STAGE_DIR).mkdir(parents=True)
    records = []
    for label, stage in STAGES.items():
        mask = obs["time_points"].eq(label).to_numpy()
        stage_obs = obs.loc[mask].copy()
        stage_counts = counts[mask]
        coords = stage_obs[["x", "y"]].to_numpy()
        sample = ad.AnnData(X=stage_counts, obs=stage_obs, var=var.copy(),
                           layers={"counts": stage_counts}, obsm={"spatial": coords})
        relpath = STAGE_DIR / f"{stage}.h5ad"
        path = work / relpath
        sample.write_h5ad(path, compression="gzip")
        umi = verify_stage(path, stage_counts, stage_obs, var, coords)
        records.append({"stage": stage, "time_point": label, "path": relpath.as_posix(),
                        "n_spots": sample.n_obs, "n_genes": sample.n_vars,
                        "n_sections": int(stage_obs["orig.ident"].nunique()),
                        "n_isolates": int(stage_obs["isolate"].nunique()),
                        "stored_counts": int(stage_counts.nnz), "total_umis": umi,
                        "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        print(f"Verified {label}: {sample.n_obs:,} spots, {stage_counts.nnz:,} stored counts", flush=True)
    return records


def write_section_qc(path, obs):
    rows = []
    for section, group in obs.groupby("orig.ident", sort=True):
        annotated = group.loc[group["annotation"].str.strip().ne(""), "annotation"]
        rows.append({"section_id": section, "isolate": group["isolate"].iloc[0], "cid": group["cid"].iloc[0],
                     "time_point": group["time_points"].iloc[0], "stage_day": group["stage_day"].iloc[0],
                     "n_spots": len(group), "median_umi": group["nCount_Spatial"].median(),
                     "median_detected_genes": group["nFeature_Spatial"].median(),
                     "x_min": group["x"].min(), "x_max": group["x"].max(),
                     "y_min": group["y"].min(), "y_max": group["y"].max(),
                     "n_annotated_spots": len(annotated), "n_annotations": annotated.nunique(),
                     "annotation_spot_counts": json.dumps(annotated.value_counts().to_dict(), sort_keys=True)})
    frame = pd.DataFrame(rows).sort_values(["stage_day", "section_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return len(frame)


def preprocess(rds, metadata, root, rscript):
    root = root.resolve()
    outputs = (STAGE_DIR, QC_PATH, MANIFEST_PATH)
    for relative in outputs:
        output = root / relative
        if not output.resolve().is_relative_to(root):
            raise ValueError(f"Output resolves outside the requested directory: {output}")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")

    obs = read_metadata(metadata)
    sources = {name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for name, path in (("rds", rds), ("metadata", metadata))}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".zebrafish_preprocessing_", dir=root) as tmp:
        work = Path(tmp)
        sparse_dir = work / "sparse"
        sparse_dir.mkdir()
        extractor = Path(__file__).with_name("extract_zebrafish_regenerate_counts.R")
        subprocess.run([rscript, "--vanilla", str(extractor), str(rds.resolve()), str(sparse_dir)], check=True)
        stages = write_stages(work, sparse_dir, obs)
        n_sections = write_section_qc(work / QC_PATH, obs)
        if sum(stage["n_spots"] for stage in stages) != len(obs) or sum(stage["n_sections"] for stage in stages) != n_sections:
            raise ValueError("The H5ADs do not cover all metadata spots and sections exactly once")
        manifest = {"dataset_id": "zebrafish_regenerate", "status": "complete", "sources": sources,
                    "counts_source": "Spatial@counts", "coordinate_source": ["x", "y"],
                    "time_unit": "day", "n_spots": len(obs), "n_genes": stages[0]["n_genes"],
                    "n_sections": n_sections, "stages": stages,
                    "section_qc": {"path": QC_PATH.as_posix(), "sha256": sha256(work / QC_PATH)},
                    "verification": "Exact sparse counts, ordered IDs, metadata and coordinates after H5AD reload"}
        write_new_json(work / MANIFEST_PATH, manifest)
        for relative in outputs:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (work / relative).rename(root / relative)
    print(f"Completed {len(stages)} H5ADs covering {len(obs):,} spots in {n_sections} sections", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    preprocess(args.rds, args.metadata, args.output_root, args.rscript)


if __name__ == "__main__":
    main()
