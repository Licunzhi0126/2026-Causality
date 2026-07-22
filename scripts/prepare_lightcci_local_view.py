#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.io.loaders import LayerDataResolver, peek_h5ad_genes, peek_h5ad_units, read_commot_index


DEFAULT_LAYERS = ("spot", "seurat_k150", "seurat_k40")
DEFAULT_STAGES = ("11.5", "12.5", "13.5", "14.5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _materialize_file(source: Path, destination: Path) -> str:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"Destination exists and is not a file: {destination}")
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"Refusing to overwrite a different existing file: {destination}")
        return "reused_existing_identical"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _spot_expression_fallback(source_root: Path, sample_stem: str) -> Path:
    return source_root / "cci" / "spot" / f"{sample_stem}_COMMOT.h5ad"


def _iter_optional_domain_files(source_root: Path, layer: str, organ: str, sample_stem: str) -> Iterable[Path]:
    if layer == "spot":
        return ()
    layer_root = source_root / layer / organ
    return tuple(
        path
        for path in (
            layer_root / f"{sample_stem}_spot_domain_map.csv",
            layer_root / f"{sample_stem}_domain_organ_counts.csv",
        )
        if path.is_file()
    )


def build_data_view(
    *,
    source_root: Path,
    view_root: Path,
    organ: str,
    stages: Iterable[str],
    layers: Iterable[str] = DEFAULT_LAYERS,
) -> dict[str, object]:
    source_root = Path(source_root).resolve()
    view_root = Path(view_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source data root does not exist: {source_root}")
    if source_root == view_root or source_root in view_root.parents:
        raise ValueError("The data view must not be the source root or be created inside the source data root.")

    manifest_path = view_root / "input_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing completed data view: {manifest_path}")
    view_root.mkdir(parents=True, exist_ok=True)

    resolver = LayerDataResolver(source_root)
    file_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for layer in map(str, layers):
        for stage in map(str, stages):
            paths = resolver.paths(layer, organ, stage)
            source_h5ad = paths.h5ad
            expression_source = "canonical_h5ad"
            if not source_h5ad.is_file() and layer == "spot":
                source_h5ad = _spot_expression_fallback(source_root, paths.sample_stem)
                expression_source = "commot_h5ad_fallback"
            _require_file(source_h5ad, f"{layer} {stage} expression H5AD")
            _require_file(paths.cci_total, f"{layer} {stage} CCI total")
            _require_file(paths.cci_index, f"{layer} {stage} CCI index")
            _require_file(paths.grn_edges, f"{layer} {stage} GRN edges")

            h5ad_units = peek_h5ad_units(source_h5ad)
            index_units = read_commot_index(paths.cci_index)
            if h5ad_units != index_units:
                raise ValueError(
                    f"Unit order mismatch for {layer} {stage}: H5AD has {len(h5ad_units)} units and "
                    f"CCI index has {len(index_units)} units."
                )
            cci_shape = tuple(map(int, sp.load_npz(paths.cci_total).shape))
            expected_shape = (len(index_units), len(index_units))
            if cci_shape != expected_shape:
                raise ValueError(
                    f"CCI shape mismatch for {layer} {stage}: got {cci_shape}, expected {expected_shape}."
                )

            destinations = (
                (source_h5ad, view_root / layer / organ / f"{paths.sample_stem}.h5ad", "expression_h5ad"),
                (
                    paths.cci_total,
                    view_root / "cci" / layer / f"{paths.sample_stem}_CCI_total.npz",
                    "cci_total",
                ),
                (
                    paths.cci_index,
                    view_root / "cci" / layer / f"{paths.sample_stem}_index.tsv",
                    "cci_index",
                ),
                (
                    paths.grn_edges,
                    view_root / "grn" / layer / paths.sample_stem / "grn_edges.csv",
                    "grn_edges",
                ),
            )
            for source, destination, role in destinations:
                mode = _materialize_file(source, destination)
                file_rows.append(
                    {
                        "layer": layer,
                        "stage": stage,
                        "role": role,
                        "source": str(source.resolve()),
                        "destination": str(destination),
                        "materialization": mode,
                        "size_bytes": int(source.stat().st_size),
                        "sha256": sha256_file(source),
                    }
                )

            for source in _iter_optional_domain_files(source_root, layer, organ, paths.sample_stem):
                destination = view_root / layer / organ / source.name
                mode = _materialize_file(source, destination)
                file_rows.append(
                    {
                        "layer": layer,
                        "stage": stage,
                        "role": "domain_mapping_auxiliary",
                        "source": str(source.resolve()),
                        "destination": str(destination),
                        "materialization": mode,
                        "size_bytes": int(source.stat().st_size),
                        "sha256": sha256_file(source),
                    }
                )

            sample_rows.append(
                {
                    "organ": organ,
                    "layer": layer,
                    "stage": stage,
                    "sample_stem": paths.sample_stem,
                    "expression_source": expression_source,
                    "unit_count": len(index_units),
                    "gene_count": len(peek_h5ad_genes(source_h5ad)),
                    "h5ad_index_equals_cci_index": True,
                    "cci_shape": list(cci_shape),
                }
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "view_root": str(view_root),
        "source_root_modified": False,
        "organ": organ,
        "stages": list(map(str, stages)),
        "layers": list(map(str, layers)),
        "samples": sample_rows,
        "files": file_rows,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive LightCCI local data view with canonical spot H5AD paths."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = build_data_view(
        source_root=args.source_root,
        view_root=args.view_root,
        organ=str(args.organ),
        stages=list(map(str, args.stages)),
        layers=list(map(str, args.layers)),
    )
    print(
        json.dumps(
            {
                "status": "created",
                "view_root": payload["view_root"],
                "sample_count": len(payload["samples"]),
                "file_count": len(payload["files"]),
                "source_root_modified": payload["source_root_modified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
