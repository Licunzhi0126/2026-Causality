from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.io.multiscale_coarse_inputs import (
    InputRoots,
    prepare_cci_index,
    prepare_multiscale_inputs,
    resolve_stage_inputs,
    validate_stage_inputs,
)


def _roots(tmp_path: Path) -> InputRoots:
    return InputRoots(
        spot_root=tmp_path / "spot",
        seurat_k150_root=tmp_path / "seurat_k150",
        seurat_k40_root=tmp_path / "seurat_k40",
        cci_root=tmp_path / "cci_clean",
        grn_root=tmp_path / "grn",
        developmental_root=tmp_path / "developmental_features",
    )


def _write_stage(roots: InputRoots, scale: str, stage: str = "11.5", *, source_index: bool = False):
    resolved = resolve_stage_inputs(roots, scale=scale, organ="heart", stage=stage)
    count = {"spot": 6, "seurat_k40": 40, "seurat_k150": 150}[scale]
    units = [f"domain_{idx:03d}" for idx in range(1, count + 1)] if scale != "spot" else [f"s{idx}" for idx in range(count)]
    resolved.h5ad.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(
        X=sp.csr_matrix(np.ones((count, 3), dtype=float)),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    ).write_h5ad(resolved.h5ad)
    resolved.cci_total.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(resolved.cci_total, sp.eye(count, format="csr"))
    resolved.grn_edges.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"regulator": ["g1", "g2"], "target": ["g2", "g3"], "weight": [1.0, 0.5]}
    ).to_csv(resolved.grn_edges, index=False)
    resolved.maturity_features.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": units, "pseudotime": np.linspace(0.0, 1.0, count)}).to_csv(
        resolved.maturity_features, index=False
    )
    if scale != "spot":
        spot_ids = [f"s{idx}" for idx in range(count)]
        assert resolved.spot_domain_map is not None
        resolved.spot_domain_map.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"spot_id": spot_ids, "domain_id": units}).to_csv(resolved.spot_domain_map, index=False)
        resolved.source_spot_features.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit_id": spot_ids, "pseudotime": np.linspace(0.0, 1.0, count)}).to_csv(
            resolved.source_spot_features, index=False
        )
    if source_index:
        pd.DataFrame({"unit_id": units[::-1]}).to_csv(resolved.source_cci_index, sep="\t", index=False)
    return resolved, units


def test_resolves_documented_scale_filenames(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    k150 = resolve_stage_inputs(roots, scale="seurat_k150", organ="heart", stage="12.5")
    k40 = resolve_stage_inputs(roots, scale="seurat_k40", organ="heart", stage="12.5")
    assert k150.h5ad.name == "seurat150_heart_12.5.h5ad"
    assert k150.maturity_features == roots.developmental_root / "seurat_k150" / "heart_12.5_features.csv"
    assert k40.h5ad.name == "seurat_heart_12.5.h5ad"
    assert k40.cci_total.name == "seurat_heart_12.5_CCI_total.npz"


def test_validates_k40_and_reconstructs_missing_index_in_staging(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    resolved, units = _write_stage(roots, "seurat_k40")
    validation = validate_stage_inputs(resolved)
    assert validation["n_units"] == 40
    output, source = prepare_cci_index(resolved, out_root=tmp_path / "runs")
    assert source == "reconstructed_from_h5ad_obs_order"
    assert pd.read_csv(output, sep="\t")["unit_id"].tolist() == units
    assert output.parent == tmp_path / "runs" / "_prepared_inputs" / "cci_index" / "seurat_k40"


def test_uses_source_index_when_id_set_matches_h5ad(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    resolved, _ = _write_stage(roots, "spot", source_index=True)
    validate_stage_inputs(resolved)
    output, source = prepare_cci_index(resolved, out_root=tmp_path / "runs")
    assert output == resolved.source_cci_index
    assert source == "source_index_tsv"


def test_rejects_domain_maturity_id_mismatch(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    resolved, _ = _write_stage(roots, "seurat_k40")
    table = pd.read_csv(resolved.maturity_features)
    table.loc[0, "unit_id"] = "wrong_domain"
    table.to_csv(resolved.maturity_features, index=False)
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_stage_inputs(resolved)


def test_preparation_writes_manifest_without_copying_maturity(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    resolved, _ = _write_stage(roots, "spot")
    prepared = prepare_multiscale_inputs(
        roots,
        out_root=tmp_path / "runs",
        organ="heart",
        time_points=["11.5"],
        scales=["spot"],
    )
    assert len(prepared) == 1
    manifest_path = tmp_path / "runs" / "_prepared_inputs" / "preparation_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stages"][0]["maturity_features"] == str(resolved.maturity_features)
    assert not (tmp_path / "runs" / "_prepared_inputs" / "developmental_features").exists()
