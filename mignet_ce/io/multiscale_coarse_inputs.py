from __future__ import annotations

"""Resolve and validate multiscale maturity/CCI/GRN coarse-graining inputs."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.loaders import read_commot_index, standardize_grn_edges


DEFAULT_SCALES = ("spot", "seurat_k150", "seurat_k40")
DEFAULT_TIME_POINTS = ("11.5", "12.5", "13.5", "14.5")


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    sample_template: str
    expected_units: int | None = None
    requires_domain_map: bool = False

    def sample(self, organ: str, stage: str) -> str:
        return self.sample_template.format(organ=organ, stage=stage)


SCALE_SPECS: Mapping[str, ScaleSpec] = {
    "spot": ScaleSpec("spot", "spot_{organ}_{stage}"),
    "seurat_k150": ScaleSpec(
        "seurat_k150", "seurat150_{organ}_{stage}", expected_units=150, requires_domain_map=True
    ),
    "seurat_k40": ScaleSpec(
        "seurat_k40", "seurat_{organ}_{stage}", expected_units=40, requires_domain_map=True
    ),
}


@dataclass(frozen=True)
class InputRoots:
    spot_root: Path
    seurat_k150_root: Path
    seurat_k40_root: Path
    cci_root: Path
    grn_root: Path
    developmental_root: Path

    def h5ad_root(self, scale: str) -> Path:
        return Path(getattr(self, f"{scale}_root"))


@dataclass(frozen=True)
class ResolvedStageInputs:
    scale: str
    organ: str
    stage: str
    sample: str
    h5ad: Path
    cci_total: Path
    source_cci_index: Path
    grn_edges: Path
    maturity_features: Path
    source_spot_features: Path
    spot_domain_map: Path | None

    def manifest(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


@dataclass(frozen=True)
class PreparedStageInputs:
    resolved: ResolvedStageInputs
    cci_index: Path
    index_source: str
    n_units: int
    n_genes: int
    n_grn_overlap_genes: int

    def manifest(self) -> dict[str, object]:
        return {
            **self.resolved.manifest(),
            "cci_index": str(self.cci_index),
            "index_source": self.index_source,
            "n_units": self.n_units,
            "n_genes": self.n_genes,
            "n_grn_overlap_genes": self.n_grn_overlap_genes,
            "status": "ok",
        }


def resolve_stage_inputs(
    roots: InputRoots,
    *,
    scale: str,
    organ: str,
    stage: str,
) -> ResolvedStageInputs:
    try:
        spec = SCALE_SPECS[scale]
    except KeyError as exc:
        raise ValueError(f"Unsupported scale {scale!r}; expected one of {list(SCALE_SPECS)}.") from exc
    sample = spec.sample(str(organ), str(stage))
    h5ad_root = roots.h5ad_root(scale)
    h5ad = h5ad_root / str(organ) / f"{sample}.h5ad"
    domain_map = (
        h5ad_root / str(organ) / f"{sample}_spot_domain_map.csv"
        if spec.requires_domain_map
        else None
    )
    return ResolvedStageInputs(
        scale=scale,
        organ=str(organ),
        stage=str(stage),
        sample=sample,
        h5ad=h5ad,
        cci_total=Path(roots.cci_root) / scale / f"{sample}_CCI_total.npz",
        source_cci_index=Path(roots.cci_root) / scale / f"{sample}_index.tsv",
        grn_edges=Path(roots.grn_root) / scale / sample / "grn_edges.csv",
        maturity_features=Path(roots.developmental_root) / scale / f"{organ}_{stage}_features.csv",
        source_spot_features=Path(roots.developmental_root) / "spot" / f"{organ}_{stage}_features.csv",
        spot_domain_map=domain_map,
    )


def validate_stage_inputs(resolved: ResolvedStageInputs) -> dict[str, object]:
    spec = SCALE_SPECS[resolved.scale]
    for label, path in (
        ("H5AD", resolved.h5ad),
        ("CCI", resolved.cci_total),
        ("GRN", resolved.grn_edges),
        ("developmental features", resolved.maturity_features),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} input for {resolved.scale}/{resolved.sample}: {path}")

    obs_ids, gene_ids = _read_h5ad_axes(resolved.h5ad)
    _require_unique(obs_ids, f"H5AD observation IDs in {resolved.h5ad}")
    if spec.expected_units is not None and len(obs_ids) != spec.expected_units:
        raise ValueError(
            f"{resolved.scale}/{resolved.sample} expects {spec.expected_units} H5AD units, found {len(obs_ids)}."
        )

    cci = sp.load_npz(resolved.cci_total)
    if cci.ndim != 2 or cci.shape[0] != cci.shape[1]:
        raise ValueError(f"CCI matrix must be square: {resolved.cci_total} has shape {cci.shape}.")
    if cci.shape[0] != len(obs_ids):
        raise ValueError(
            f"CCI dimension {cci.shape[0]} does not match H5AD n_obs {len(obs_ids)} for {resolved.sample}."
        )

    if spec.requires_domain_map:
        _validate_domain_map(resolved, obs_ids)
    maturity = _validate_maturity(resolved.maturity_features, obs_ids)
    grn = standardize_grn_edges(pd.read_csv(resolved.grn_edges))
    grn_genes = set(grn["regulator"].astype(str)) | set(grn["target"].astype(str))
    overlap = set(gene_ids) & grn_genes
    if not overlap:
        raise ValueError(f"GRN {resolved.grn_edges} has no gene overlap with {resolved.h5ad}.")
    return {
        "n_units": len(obs_ids),
        "n_genes": len(gene_ids),
        "n_grn_overlap_genes": len(overlap),
        "maturity_min": float(maturity["pseudotime"].min()),
        "maturity_max": float(maturity["pseudotime"].max()),
    }


def prepare_cci_index(
    resolved: ResolvedStageInputs,
    *,
    out_root: Path,
    overwrite: bool = False,
) -> tuple[Path, str]:
    obs_ids, _ = _read_h5ad_axes(resolved.h5ad)
    _require_unique(obs_ids, f"H5AD observation IDs in {resolved.h5ad}")
    if resolved.source_cci_index.exists():
        index_ids = read_commot_index(resolved.source_cci_index)
        _require_unique(index_ids, f"CCI index IDs in {resolved.source_cci_index}")
        if set(index_ids) != set(obs_ids):
            raise ValueError(
                f"Source CCI index IDs do not exactly match H5AD observations for {resolved.sample}."
            )
        return resolved.source_cci_index, "source_index_tsv"

    matrix = sp.load_npz(resolved.cci_total)
    if matrix.shape != (len(obs_ids), len(obs_ids)):
        raise ValueError(
            f"Cannot reconstruct CCI index: matrix shape {matrix.shape} does not match H5AD n_obs={len(obs_ids)}."
        )
    expected_name = f"{resolved.sample}_CCI_total.npz"
    expected_h5ad = f"{resolved.sample}.h5ad"
    if resolved.cci_total.name != expected_name or resolved.h5ad.name != expected_h5ad:
        raise ValueError(
            f"Cannot reconstruct CCI index because sample filenames do not match: "
            f"CCI={resolved.cci_total.name!r}, H5AD={resolved.h5ad.name!r}, sample={resolved.sample!r}."
        )
    output = Path(out_root) / "_prepared_inputs" / "cci_index" / resolved.scale / f"{resolved.sample}_index.tsv"
    if output.exists() and not overwrite:
        existing = read_commot_index(output)
        if existing != obs_ids:
            raise FileExistsError(f"Prepared CCI index exists with different contents: {output}")
        return output, "reconstructed_from_h5ad_obs_order"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": obs_ids}).to_csv(output, sep="\t", index=False)
    return output, "reconstructed_from_h5ad_obs_order"


def prepare_multiscale_inputs(
    roots: InputRoots,
    *,
    out_root: Path,
    organ: str = "heart",
    time_points: Sequence[str] = DEFAULT_TIME_POINTS,
    scales: Sequence[str] = DEFAULT_SCALES,
    overwrite: bool = False,
) -> list[PreparedStageInputs]:
    prepared: list[PreparedStageInputs] = []
    for scale in map(str, scales):
        for stage in map(str, time_points):
            resolved = resolve_stage_inputs(roots, scale=scale, organ=organ, stage=stage)
            validation = validate_stage_inputs(resolved)
            index_path, index_source = prepare_cci_index(
                resolved, out_root=out_root, overwrite=overwrite
            )
            prepared.append(
                PreparedStageInputs(
                    resolved=resolved,
                    cci_index=index_path,
                    index_source=index_source,
                    n_units=int(validation["n_units"]),
                    n_genes=int(validation["n_genes"]),
                    n_grn_overlap_genes=int(validation["n_grn_overlap_genes"]),
                )
            )
    write_preparation_manifest(prepared, out_root=out_root)
    return prepared


def write_preparation_manifest(prepared: Sequence[PreparedStageInputs], *, out_root: Path) -> Path:
    path = Path(out_root) / "_prepared_inputs" / "preparation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "maturity_id_column": "unit_id",
        "maturity_column": "pseudotime",
        "stages": [item.manifest() for item in prepared],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _validate_domain_map(resolved: ResolvedStageInputs, obs_ids: Sequence[str]) -> None:
    assert resolved.spot_domain_map is not None
    if not resolved.spot_domain_map.exists():
        raise FileNotFoundError(f"Missing spot-domain map: {resolved.spot_domain_map}")
    mapping = pd.read_csv(resolved.spot_domain_map)
    if not {"spot_id", "domain_id"}.issubset(mapping.columns):
        raise ValueError(f"{resolved.spot_domain_map} must contain spot_id and domain_id.")
    spots = mapping["spot_id"].astype(str).tolist()
    domains = mapping["domain_id"].astype(str).tolist()
    _require_unique(spots, f"spot IDs in {resolved.spot_domain_map}")
    if set(domains) != set(obs_ids):
        raise ValueError(f"Domain-map domain_id set does not exactly match H5AD observations for {resolved.sample}.")
    if not resolved.source_spot_features.exists():
        raise FileNotFoundError(f"Missing source spot developmental features: {resolved.source_spot_features}")
    source = pd.read_csv(resolved.source_spot_features)
    if "unit_id" not in source.columns:
        raise ValueError(f"{resolved.source_spot_features} must contain unit_id.")
    feature_ids = source["unit_id"].astype(str).tolist()
    _require_unique(feature_ids, f"unit IDs in {resolved.source_spot_features}")
    feature_set = set(feature_ids)
    missing = [spot for spot in spots if spot not in feature_set]
    if missing and "organ" in mapping.columns:
        prefixed = (mapping["organ"].astype(str) + "__" + mapping["spot_id"].astype(str)).tolist()
        missing = [spot for spot, pref in zip(spots, prefixed) if spot not in feature_set and pref not in feature_set]
    if missing:
        raise ValueError(
            f"Domain map contains {len(missing)} spots absent from source developmental features, for example {missing[:5]}."
        )


def _validate_maturity(path: Path, obs_ids: Sequence[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"unit_id", "pseudotime"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")
    ids = table["unit_id"].astype(str).tolist()
    _require_unique(ids, f"unit IDs in {path}")
    if set(ids) != set(map(str, obs_ids)):
        raise ValueError(f"Developmental feature unit_id set does not exactly match H5AD observations: {path}")
    numeric_columns = [column for column in table.columns if column != "unit_id"]
    numeric = table.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Developmental features contain non-finite numeric values: {path}")
    return table


def _require_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(map(str, values))):
        raise ValueError(f"{label} are not unique.")


def _read_h5ad_axes(path: Path) -> tuple[list[str], list[str]]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.obs_names.astype(str).tolist(), adata.var_names.astype(str).tolist()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
