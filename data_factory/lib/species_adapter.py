"""Non-destructive count/coordinate adapters and provenance for species inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
import anndata as ad
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.organ_paths import sample_stem


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_counts(path, spec):
    a = ad.read_h5ad(path)
    if "counts" not in a.layers:
        raise ValueError(f"Raw counts layer required: {path}")
    x = sp.csr_matrix(a.layers["counts"], dtype=np.float64)
    if not np.isfinite(x.data).all() or np.any(x.data < 0) or not np.allclose(x.data, np.rint(x.data)):
        raise ValueError(f"Expected finite nonnegative integer counts: {path}")
    if np.any(np.asarray(x.sum(axis=1)).ravel() <= 0):
        raise ValueError(f"Zero-expression spatial units: {path}")
    if not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError(f"Duplicate raw axis IDs: {path}")
    if spec.gene_column in a.var:
        genes = a.var[spec.gene_column]
    elif a.var.index.name == spec.gene_column:
        # AnnData consumes the H5AD index field rather than retaining a column.
        genes = a.var_names.to_series()
    else:
        raise ValueError(f"Missing canonical gene field {spec.gene_column}: {path}")
    if genes.isna().any() or genes.astype(str).str.strip().eq("").any() or genes.astype(str).duplicated().any():
        raise ValueError(f"Missing/duplicate canonical gene IDs: {path}")
    if "spatial" in a.obsm:
        xy = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]
    elif {"spatial_x", "spatial_y"} <= set(a.obs):
        xy = a.obs[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    else:
        raise ValueError(f"No measured coordinates: {path}")
    if xy.shape != (a.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid measured coordinates: {path}")
    a.var["original_gene_name"] = a.var_names.astype(str)
    a.var_names = pd.Index(genes.astype(str).to_numpy())
    a.X = x.copy()
    a.layers["counts"] = x
    a.obsm["spatial"] = xy
    return a


def prepare_inputs(spec, data_root, *, resume=False, dry_run=False):
    spec.validate()
    if len(spec.organs) != 1 or not spec.raw_files:
        raise ValueError("Species adapter requires one scope and explicit raw_files")
    root = Path(data_root)
    organ = spec.organs[0]
    manifest_path = root / "manifests" / "species_input_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if previous and (not resume or previous["experiment"] != spec.metadata()):
        raise ValueError("Input manifest already exists or profile changed; use matching --resume or a new data-root")
    rows = []
    for stage in spec.time_points:
        raw = root / spec.raw_files[stage]
        stem = sample_stem("spot", organ, stage)
        output = root / "spot" / (organ if spec.output_layout == "nested" else "") / (stem + ".h5ad")
        a = read_counts(raw, spec)
        raw_hash = sha256(raw)
        row = {"stage": stage, "source": str(raw.resolve()), "source_sha256": raw_hash,
               "output": str(output.resolve()), "n_spots": a.n_obs, "n_gene_columns": a.n_vars,
               "n_detected_genes": int(np.sum(np.asarray((a.X > 0).sum(axis=0)).ravel() > 0)),
               "expression_source": "layers/counts", "coordinate_transform": "none"}
        old = next((r for r in previous["samples"] if r["stage"] == stage), None) if previous else None
        if output.exists():
            if not resume or old is None or old["source_sha256"] != raw_hash or old["output_sha256"] != sha256(output):
                raise ValueError(f"Existing output cannot be verified: {output}")
        elif previous:
            raise ValueError(f"Incomplete materialized input manifest: {output}")
        elif not dry_run:
            a.obs["organ"] = organ
            a.obs["stage"] = stage
            a.obs["spot_id"] = a.obs_names.astype(str)
            if "annotation" not in a.obs:
                for key in ("bin_annotation", "Annotation", "layer_annotation"):
                    if key in a.obs:
                        a.obs["annotation"] = a.obs[key].astype(str)
                        break
            a.uns["species_input_provenance"] = {
                "dataset_id": spec.dataset_id, "profile_sha256": spec.fingerprint,
                "raw_sha256": raw_hash, "raw_path": str(raw.resolve()),
                "expression_source": "layers/counts", "coordinate_transform": "none"}
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.stem + ".partial." + uuid.uuid4().hex + ".h5ad")
            a.write_h5ad(partial, compression="gzip")
            partial.rename(output)
        row["output_sha256"] = sha256(output) if output.is_file() else None
        rows.append(row)
    report = {"experiment": spec.metadata(), "status": "dry_run" if dry_run else "complete", "samples": rows}
    if not dry_run and not previous:
        write_new_json(manifest_path, report)
    return previous if previous and not dry_run else report
