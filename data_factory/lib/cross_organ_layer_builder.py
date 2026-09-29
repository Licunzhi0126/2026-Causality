from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from factory_common import FACTORY_OUTPUT_ROOT, ORGANS, STAGES, ensure_dir, write_csv


@dataclass(frozen=True)
class CrossOrganLayerSpec:
    layer: str
    sample_prefix: str
    output_prefix: str


CROSS_ORGAN_LAYER_SPECS: Dict[str, CrossOrganLayerSpec] = {
    "seurat_k40": CrossOrganLayerSpec(
        layer="seurat_k40",
        sample_prefix="seurat",
        output_prefix="seurat_k40_all_organs",
    ),
    "louvain_k150": CrossOrganLayerSpec(
        layer="louvain_k150",
        sample_prefix="louvain150",
        output_prefix="louvain_k150_all_organs",
    ),
}


def _choose_count_matrix(adata: ad.AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def _sample_stem(spec: CrossOrganLayerSpec, organ: str, stage: str) -> str:
    return f"{spec.sample_prefix}_{organ}_{stage}"


def _input_h5ad(input_root: Path, spec: CrossOrganLayerSpec, organ: str, stage: str) -> Path:
    return input_root / spec.layer / organ / f"{_sample_stem(spec, organ, stage)}.h5ad"


def _input_spot_map(input_root: Path, spec: CrossOrganLayerSpec, organ: str, stage: str) -> Path:
    stem = _sample_stem(spec, organ, stage)
    return input_root / spec.layer / organ / f"{stem}_spot_domain_map.csv"


def _output_stem(spec: CrossOrganLayerSpec, stage: str) -> str:
    return f"{spec.output_prefix}_{stage}"


def _as_count_layer(adata: ad.AnnData) -> ad.AnnData:
    work = adata.copy()
    counts = _choose_count_matrix(work)
    work.X = counts.copy() if hasattr(counts, "copy") else counts
    work.layers["count"] = work.X.copy() if hasattr(work.X, "copy") else work.X
    return work


def _prepare_part(
    input_root: Path,
    spec: CrossOrganLayerSpec,
    organ: str,
    stage: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    path = _input_h5ad(input_root, spec, organ, stage)
    sample = path.stem
    part = _as_count_layer(ad.read_h5ad(path))
    source_ids = part.obs_names.astype(str).tolist()
    unique_ids = [f"{organ}__{unit}" for unit in source_ids]

    obs = part.obs.copy()
    obs["domain_id"] = unique_ids
    obs["organ"] = organ
    obs["stage"] = stage
    obs["source_domain_id"] = source_ids
    if "domain_label" in obs.columns:
        obs["source_domain_label"] = obs["domain_label"].astype(str).to_numpy()
    else:
        obs["source_domain_label"] = source_ids
    obs["source_sample"] = sample
    if "spot_count" not in obs.columns:
        obs["spot_count"] = np.nan
    obs.index = pd.Index(unique_ids, name="domain_id")
    part.obs = obs[
        [
            "domain_id",
            "organ",
            "stage",
            "source_domain_id",
            "source_domain_label",
            "source_sample",
            "spot_count",
        ]
    ].copy()
    part.obs_names = unique_ids
    part.uns["source_sample"] = sample
    part.uns["source_layer"] = spec.layer

    map_path = _input_spot_map(input_root, spec, organ, stage)
    if map_path.exists():
        mapping = pd.read_csv(map_path)
        if "domain_id" not in mapping.columns:
            raise KeyError(f"{map_path} is missing required column 'domain_id'.")
        mapping = mapping.copy()
        mapping["organ"] = organ
        mapping["stage"] = stage
        mapping["source_sample"] = sample
        mapping["source_domain_id"] = mapping["domain_id"].astype(str)
        if "domain_label" in mapping.columns:
            mapping["source_domain_label"] = mapping["domain_label"].astype(str)
        else:
            mapping["source_domain_label"] = mapping["source_domain_id"]
        mapping["domain_id"] = organ + "__" + mapping["source_domain_id"].astype(str)
    else:
        mapping = pd.DataFrame(
            columns=[
                "spot_id",
                "domain_id",
                "organ",
                "stage",
                "source_sample",
                "source_domain_id",
                "source_domain_label",
            ]
        )

    return part, mapping


def _concat_parts(parts: Sequence[ad.AnnData], output_stem: str, spec: CrossOrganLayerSpec, stage: str) -> ad.AnnData:
    combined = ad.concat(
        list(parts),
        axis=0,
        join="outer",
        merge="same",
        uns_merge="first",
        fill_value=0,
        index_unique=None,
    )
    if sp.issparse(combined.X):
        combined.X = combined.X.tocsr()
    combined.layers["count"] = combined.X.copy() if hasattr(combined.X, "copy") else combined.X
    combined.uns["cross_organ"] = True
    combined.uns["source_layer"] = spec.layer
    combined.uns["stage"] = stage
    combined.uns["sample_name"] = output_stem
    return combined


def build_cross_organ_layer_stage(
    input_root: Path,
    output_root: Path,
    spec: CrossOrganLayerSpec,
    stage: str,
    organs: Sequence[str] = ORGANS,
    overwrite: bool = False,
) -> dict:
    layer_out = output_root / spec.layer
    ensure_dir(layer_out)
    stem = _output_stem(spec, stage)
    out_h5ad = layer_out / f"{stem}.h5ad"
    out_map = layer_out / f"{stem}_spot_domain_map.csv"

    row: dict = {
        "layer": spec.layer,
        "stage": stage,
        "output_file": str(out_h5ad),
        "spot_domain_map": str(out_map),
        "status": "planned",
    }
    if out_h5ad.exists() and out_map.exists() and not overwrite:
        row["status"] = "exists_skipped"
        return row

    found: List[str] = []
    missing: List[str] = []
    parts: List[ad.AnnData] = []
    mappings: List[pd.DataFrame] = []
    source_files: List[str] = []

    for organ in organs:
        path = _input_h5ad(input_root, spec, organ, stage)
        if not path.exists():
            missing.append(organ)
            continue
        part, mapping = _prepare_part(input_root, spec, organ, stage)
        parts.append(part)
        mappings.append(mapping)
        found.append(organ)
        source_files.append(str(path))

    if not parts:
        raise FileNotFoundError(f"No input h5ad files found for layer={spec.layer}, stage={stage}, organs={list(organs)}.")

    combined = _concat_parts(parts, stem, spec, stage)
    if not combined.obs_names.is_unique:
        raise ValueError(f"Cross-organ obs_names are not unique for {spec.layer} {stage}.")

    combined.write_h5ad(out_h5ad)
    if mappings:
        pd.concat(mappings, ignore_index=True).to_csv(out_map, index=False)
    else:
        pd.DataFrame().to_csv(out_map, index=False)

    row.update(
        {
            "status": "written",
            "n_units": int(combined.n_obs),
            "n_genes": int(combined.n_vars),
            "organs_found": ";".join(found),
            "organs_missing": ";".join(missing),
            "source_files": ";".join(source_files),
        }
    )
    return row


def build_cross_organ_layers(
    input_root: Path = FACTORY_OUTPUT_ROOT,
    output_root: Path = FACTORY_OUTPUT_ROOT / "cross_organ",
    layers: Sequence[str] = ("seurat_k40", "louvain_k150"),
    organs: Sequence[str] = ORGANS,
    stages: Sequence[str] = STAGES,
    overwrite: bool = False,
    manifest_name: str = "domain_manifest_cross_organ.csv",
    manifest_root: Path | None = None,
) -> List[dict]:
    rows: List[dict] = []
    for layer in layers:
        if layer not in CROSS_ORGAN_LAYER_SPECS:
            raise ValueError(f"Unsupported cross-organ layer: {layer!r}. Expected one of {sorted(CROSS_ORGAN_LAYER_SPECS)}.")
        spec = CROSS_ORGAN_LAYER_SPECS[layer]
        for stage in stages:
            rows.append(
                build_cross_organ_layer_stage(
                    input_root=input_root,
                    output_root=output_root,
                    spec=spec,
                    stage=stage,
                    organs=organs,
                    overwrite=overwrite,
                )
            )

    manifest_dir = manifest_root if manifest_root is not None else input_root / "manifests"
    manifest = manifest_dir / manifest_name
    write_csv(manifest, rows)
    return rows
