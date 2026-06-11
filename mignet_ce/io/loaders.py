from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import LAYER_SPECS, LayerSpec


@dataclass(frozen=True)
class LayerPaths:
    layer: str
    organ: str
    stage: str
    sample_stem: str
    h5ad: Path
    grn_edges: Path
    cci_manifest: Path
    cci_index: Path
    cci_lr_dir: Path
    spot_domain_map: Optional[Path]


@dataclass
class ExpressionData:
    units: List[str]
    genes: List[str]
    expr: pd.DataFrame
    coords: pd.DataFrame
    obs: pd.DataFrame


class LayerDataResolver:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def paths(self, layer: str, organ: str, stage: str) -> LayerPaths:
        spec = self.layer_spec(layer)
        sample = spec.sample_stem(organ, stage)
        spot_map = None
        if layer != "spot":
            spot_map = self.data_root / layer / organ / f"{sample}_spot_domain_map.csv"
        return LayerPaths(
            layer=layer,
            organ=organ,
            stage=stage,
            sample_stem=sample,
            h5ad=self.data_root / layer / organ / f"{sample}.h5ad",
            grn_edges=self.data_root / "grn" / layer / sample / "grn_edges.csv",
            cci_manifest=self.data_root / "cci" / layer / f"{sample}_COMMOT_lr_pairs.tsv",
            cci_index=self.data_root / "cci" / layer / f"{sample}_index.tsv",
            cci_lr_dir=self.data_root / "cci" / layer / f"{sample}_COMMOT_by_LR",
            spot_domain_map=spot_map,
        )

    @staticmethod
    def layer_spec(layer: str) -> LayerSpec:
        try:
            return LAYER_SPECS[layer]
        except KeyError as exc:
            raise ValueError(f"Unsupported layer {layer!r}. Expected one of {sorted(LAYER_SPECS)}.") from exc


def natural_sort(values: Sequence[str]) -> List[str]:
    def key(value: str):
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]

    return sorted(map(str, values), key=key)


def choose_count_matrix(adata: ad.AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def safe_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def read_expression_h5ad(path: Path) -> ExpressionData:
    adata = ad.read_h5ad(path)
    matrix = safe_dense(choose_count_matrix(adata)).astype(float, copy=False)
    units = adata.obs_names.astype(str).tolist()
    genes = adata.var_names.astype(str).tolist()
    expr = pd.DataFrame(matrix, index=units, columns=genes)
    obs = adata.obs.copy()
    obs.index = pd.Index(units, name=adata.obs.index.name)

    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=float)
        coords = pd.DataFrame(spatial[:, :2], index=units, columns=["x", "y"])
    elif {"x", "y"}.issubset(obs.columns):
        coords = obs.loc[:, ["x", "y"]].astype(float)
    else:
        coords = pd.DataFrame(np.zeros((len(units), 2), dtype=float), index=units, columns=["x", "y"])
    return ExpressionData(units=units, genes=genes, expr=expr, coords=coords, obs=obs)


def peek_h5ad_units(path: Path) -> List[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.obs_names.astype(str).tolist()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def peek_h5ad_genes(path: Path) -> List[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.var_names.astype(str).tolist()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"Cannot find any of columns {candidates} in {list(df.columns)}")


def standardize_grn_edges(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"regulator", "tf", "source", "src"}:
            rename[col] = "regulator"
        elif low in {"target", "target_gene", "gene", "dst"}:
            rename[col] = "target"
        elif low in {"weight", "importance", "score"}:
            rename[col] = "weight"
    work = df.rename(columns=rename)
    missing = {"regulator", "target", "weight"} - set(work.columns)
    if missing:
        raise ValueError(f"GRN edge file is missing columns {sorted(missing)}; got {list(df.columns)}")
    out = work.loc[:, ["regulator", "target", "weight"]].copy()
    out["regulator"] = out["regulator"].astype(str)
    out["target"] = out["target"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["weight"])
    return out


def read_grn_edges(path: Path, top_k_targets_per_regulator: Optional[int] = None) -> pd.DataFrame:
    grn = standardize_grn_edges(pd.read_csv(path))
    grn = grn.sort_values(["regulator", "weight"], ascending=[True, False]).reset_index(drop=True)
    if top_k_targets_per_regulator is not None:
        grn = grn.groupby("regulator", group_keys=False).head(top_k_targets_per_regulator).reset_index(drop=True)
    return grn


def read_commot_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t")
    required = {"filename", "ligand", "receptor"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"COMMOT manifest {path} is missing columns {sorted(missing)}.")
    if "lr_key" not in manifest.columns:
        manifest["lr_key"] = manifest["ligand"].astype(str) + "-" + manifest["receptor"].astype(str)
    return manifest


def read_commot_index(path: Path) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return []
    return df.iloc[:, 0].astype(str).tolist()
