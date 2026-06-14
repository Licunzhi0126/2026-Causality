from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import AnnData, read_h5ad

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))


@pytest.fixture()
def louvain_builder(monkeypatch):
    import importlib

    fake_scanpy = types.SimpleNamespace(settings=types.SimpleNamespace(verbosity=0))
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    sys.modules.pop("domain_builder_louvain", None)
    module = importlib.import_module("domain_builder_louvain")
    yield module
    sys.modules.pop("domain_builder_louvain", None)


def test_spatial_domain_export_uses_domain_output_contract(tmp_path, louvain_builder) -> None:
    counts = sp.csr_matrix(np.arange(120, dtype=np.float32).reshape(20, 6) + 1)
    obs = pd.DataFrame(
        {"annotation": ["Heart"] * 10 + ["Brain"] * 10},
        index=[f"spot_{idx:02d}" for idx in range(20)],
    )
    var = pd.DataFrame(index=[f"gene_{idx:02d}" for idx in range(6)])
    adata = AnnData(X=counts, obs=obs, var=var)
    adata.layers["count"] = counts.copy()
    adata.obsm["spatial"] = np.column_stack([np.arange(20), np.arange(20) % 5]).astype(np.float32)
    labels = np.repeat(np.arange(5, dtype=np.int32), 4)

    output_dir = tmp_path / "spatial_domain_k40" / "heart"
    file_stem = "spatialDomain40_heart_11.5"
    build_info = {"method": "spatial_domain_core", "mode": "exact_k", "n_domains": 5}

    louvain_builder.export_domain_result(
        spot_adata=adata,
        count_matrix=counts,
        labels=labels,
        output_dir=output_dir,
        file_stem=file_stem,
        build_info=build_info,
    )

    expected_files = [
        output_dir / f"{file_stem}.h5ad",
        output_dir / f"{file_stem}_spots_with_domain.h5ad",
        output_dir / f"{file_stem}_spot_domain_map.csv",
        output_dir / f"{file_stem}_domain_organ_counts.csv",
        output_dir / f"{file_stem}_cluster_sizes.csv",
        output_dir / f"{file_stem}_build_summary.json",
    ]
    for path in expected_files:
        assert path.exists()

    domain_adata = read_h5ad(output_dir / f"{file_stem}.h5ad")
    assert {"domain_id", "domain_label", "spot_count"}.issubset(domain_adata.obs.columns)
    assert "spatial" in domain_adata.obsm
    assert "count" in domain_adata.layers
    assert domain_adata.uns["X_name"] == "counts"
    assert domain_adata.n_obs == 5

    spot_domain_map = pd.read_csv(output_dir / f"{file_stem}_spot_domain_map.csv")
    assert {"spot_id", "domain_id", "domain_label", "annotation", "x", "y"}.issubset(spot_domain_map.columns)
    assert len(spot_domain_map) == 20

    summary = json.loads((output_dir / f"{file_stem}_build_summary.json").read_text(encoding="utf-8"))
    assert summary["n_domains"] == 5
    assert summary["build_info"]["method"] == "spatial_domain_core"
