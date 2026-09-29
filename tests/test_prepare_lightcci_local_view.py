from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from scripts.prepare_lightcci_local_view import build_data_view


def _write_sample(root: Path, *, layer: str, prefix: str, stage: str, commot_spot: bool = False) -> None:
    stem = f"{prefix}_heart_{stage}"
    units = ["u1", "u2"]
    genes = ["g1", "g2", "g3"]
    h5ad = (
        root / "cci" / "spot" / f"{stem}_COMMOT.h5ad"
        if commot_spot
        else root / layer / "heart" / f"{stem}.h5ad"
    )
    h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.arange(6, dtype=float).reshape(2, 3),
        obs=pd.DataFrame(index=pd.Index(units, name="unit_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obsm["spatial"] = np.array([[0.0, 1.0], [2.0, 3.0]])
    adata.write_h5ad(h5ad)

    cci_root = root / "cci" / layer
    cci_root.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cci_root / f"{stem}_CCI_total.npz", sp.csr_matrix([[0.0, 1.0], [2.0, 0.0]]))
    pd.DataFrame({"unit_id": units}).to_csv(cci_root / f"{stem}_index.tsv", sep="\t", index=False)

    grn_root = root / "grn" / layer / stem
    grn_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"regulator": ["g1", "g2"], "target": ["g2", "g3"], "weight": [1.0, 2.0]}
    ).to_csv(grn_root / "grn_edges.csv", index=False)


def test_build_data_view_uses_commot_spot_h5ad_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    view = tmp_path / "view"
    for stage in ("11.5", "12.5"):
        _write_sample(source, layer="spot", prefix="spot", stage=stage, commot_spot=True)
        _write_sample(source, layer="seurat_k150", prefix="seurat150", stage=stage)
        _write_sample(source, layer="seurat_k40", prefix="seurat", stage=stage)

    before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))
    payload = build_data_view(
        source_root=source,
        view_root=view,
        organ="heart",
        stages=["11.5", "12.5"],
    )
    after = sorted(str(path.relative_to(source)) for path in source.rglob("*"))

    assert before == after
    assert payload["source_root_modified"] is False
    assert len(payload["samples"]) == 6
    assert (view / "spot" / "heart" / "spot_heart_11.5.h5ad").is_file()
    assert (view / "input_manifest.json").is_file()
    spot = next(row for row in payload["samples"] if row["layer"] == "spot" and row["stage"] == "11.5")
    assert spot["expression_source"] == "commot_h5ad_fallback"
    assert spot["h5ad_index_equals_cci_index"] is True
    if os.name == "nt":
        assert any(row["materialization"] == "hardlink" for row in payload["files"])


def test_build_data_view_refuses_completed_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for layer, prefix, fallback in (
        ("spot", "spot", True),
        ("seurat_k150", "seurat150", False),
        ("seurat_k40", "seurat", False),
    ):
        _write_sample(source, layer=layer, prefix=prefix, stage="11.5", commot_spot=fallback)
    view = tmp_path / "view"
    build_data_view(source_root=source, view_root=view, organ="heart", stages=["11.5"])
    try:
        build_data_view(source_root=source, view_root=view, organ="heart", stages=["11.5"])
    except FileExistsError:
        pass
    else:
        raise AssertionError("A completed data view must not be overwritten.")
