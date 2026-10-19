"""Canonical organ-aware input path resolution.

H5AD and domain maps live in <scale>/<organ>/; CCI and GRN share
<cci>/<scale>/ and <grn>/<scale>/ with the organ encoded in sample stems.
A flat H5AD fallback supports old exports, but ambiguous duplicate layouts fail.
"""
from __future__ import annotations
from pathlib import Path

STEMS = {"spot": "spot", "seurat_k150": "seurat150", "seurat_k40": "seurat", "seurat_k10": "seurat10"}


def sample_stem(scale: str, organ: str, stage: str) -> str:
    if scale not in STEMS:
        raise ValueError(f"Unknown input scale {scale!r}")
    return f"{STEMS[scale]}_{organ}_{stage}"


def organ_h5ad_dir(root: Path, organ: str, filename: str) -> Path:
    """Prefer nested organ directory; reject duplicates instead of guessing data versions."""
    root = Path(root)
    nested, flat = root / organ / filename, root / filename
    if nested.is_file() and flat.is_file():
        raise ValueError(f"Ambiguous organ H5AD/domain map (both locations exist): {nested} AND {flat}")
    return nested if nested.is_file() or not flat.is_file() else flat


def resolve_organ_paths(data_root: Path, scale: str, organ: str, stage: str,
                        *, cci_root: Path | None = None, grn_root: Path | None = None,
                        cci_index_root: Path | None = None) -> dict[str, Path | None]:
    base = Path(data_root)
    stem = sample_stem(scale, organ, stage)
    layer = base / scale
    cci = Path(cci_root) if cci_root is not None else base / "cci"
    grn = Path(grn_root) if grn_root is not None else base / "grn"
    indices = Path(cci_index_root) if cci_index_root is not None else cci
    return {
        "h5ad": organ_h5ad_dir(layer, organ, f"{stem}.h5ad"),
        "map": organ_h5ad_dir(layer, organ, f"{stem}_spot_domain_map.csv") if scale != "spot" else None,
        "cci": cci / scale / f"{stem}_CCI_total.npz",
        "cci_index": indices / scale / f"{stem}_index.tsv",
        "grn": grn / scale / stem / "grn_edges.csv",
    }
