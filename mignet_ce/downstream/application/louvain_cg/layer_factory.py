"""Discover and validate artifacts produced by existing Louvain/Data Factory jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from anndata import read_h5ad
import pandas as pd


@dataclass(frozen=True)
class LayerArtifact:
    sample_id: str
    h5ad: Path
    spot_domain_map: Path


def discover_layer(layer_root: Path, k: int, sample_ids: list[str]) -> dict[str, LayerArtifact]:
    manifest = layer_root / "manifests" / f"domain_manifest_louvain_k{k}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing Louvain manifest: {manifest}")
    rows = pd.read_csv(manifest)
    if not {"input_file", "output_file", "n_domains", "status"}.issubset(rows.columns):
        raise ValueError(f"Unexpected Louvain manifest schema: {manifest}")
    artifacts: dict[str, LayerArtifact] = {}
    for sample_id in sample_ids:
        selected = rows[rows["input_file"].astype(str).map(lambda value: Path(value).stem == sample_id)]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one Louvain manifest row for {sample_id}, found {len(selected)}")
        row = selected.iloc[0]
        if row["status"] not in {"written", "exists_skipped"}:
            raise ValueError(f"Louvain K{k} did not finish for {sample_id}: {row.to_dict()}")
        h5ad = Path(row["output_file"])
        map_path = h5ad.with_name(f"{h5ad.stem}_spot_domain_map.csv")
        if not h5ad.is_file() or not map_path.is_file():
            raise FileNotFoundError(f"Missing Louvain artifacts for {sample_id}: {h5ad}, {map_path}")
        domain_adata = read_h5ad(h5ad, backed="r")
        try:
            if domain_adata.n_obs != k:
                raise ValueError(f"Louvain K{k} H5AD has {domain_adata.n_obs} domains for {sample_id}.")
        finally:
            domain_adata.file.close()
        assignments = pd.read_csv(map_path)
        if not {"spot_id", "domain_id"}.issubset(assignments.columns):
            raise ValueError(f"Louvain spot-domain map has an invalid schema: {map_path}")
        if assignments["spot_id"].astype(str).duplicated().any() or assignments["domain_id"].astype(str).nunique() != k:
            raise ValueError(f"Louvain K{k} mapping is not an exact hard {k}-domain assignment: {map_path}")
        artifacts[sample_id] = LayerArtifact(sample_id, h5ad, map_path)
    return artifacts
