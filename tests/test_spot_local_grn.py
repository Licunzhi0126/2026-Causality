from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad


LIB_DIR = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from spot_local_grn_runner import infer_spot_local_grn_tables, select_spatial_neighbors


def _fake_infer(adata, _grn):
    return (
        pd.DataFrame(
            {
                "regulator": ["A"],
                "target": ["B"],
                "weight": [2.0],
            }
        ),
        {
            "n_cells": int(adata.n_obs),
            "n_genes_used": int(adata.n_vars),
            "n_regulators_used": 1,
            "n_edges": 1,
            "expression_source": "X",
        },
    )


def test_spatial_neighbors_use_coordinates_only_and_exclude_center() -> None:
    ids = ["s0", "s1", "s2", "s3"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    neighbors = select_spatial_neighbors(ids, coords, ["s0"], k_neighbors=2)["s0"]

    assert neighbors["neighbor_unit_id"].tolist() == ["s1", "s2"]
    assert neighbors["neighbor_rank"].tolist() == [1, 2]
    assert "s0" not in set(neighbors["neighbor_unit_id"])


def test_spot_local_grn_outputs_center_edges_and_auditable_neighbors() -> None:
    adata = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame(
            {"domain_id": ["different", "labels", "are", "ignored"]},
            index=["s0", "s1", "s2", "s3"],
        ),
    )
    adata.var_names = ["A", "B"]
    adata.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [8.0, 0.0]]
    )

    edges, neighbors, summary = infer_spot_local_grn_tables(
        adata,
        ["s0"],
        k_neighbors=2,
        include_center=True,
        min_cells=3,
        infer_fn=_fake_infer,
    )

    assert edges.loc[0, "center_unit_id"] == "s0"
    assert edges.loc[0, "n_neighbors"] == 2
    assert edges.loc[0, "neighbor_mode"] == "spatial"
    assert neighbors["neighbor_unit_id"].tolist() == ["s1", "s2"]
    assert summary.loc[0, "n_cells"] == 3
