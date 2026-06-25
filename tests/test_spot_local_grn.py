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

from unit_grn_layer_runner import infer_spot_unit_grn_tables
from unit_observation_counter import build_spatial_neighbor_tables


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
    neighbors = build_spatial_neighbor_tables(ids, coords, k_neighbors=2)["s0"]

    assert neighbors["neighbor_unit_id"].tolist() == ["s1", "s2"]
    assert neighbors["neighbor_rank"].tolist() == [1, 2]
    assert "s0" not in set(neighbors["neighbor_unit_id"])


def test_spot_unit_grn_processes_all_spots_with_unified_outputs() -> None:
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

    edges, summary, neighbors = infer_spot_unit_grn_tables(
        adata,
        k_neighbors=2,
        include_center=True,
        min_cells_per_unit=3,
        infer_fn=_fake_infer,
    )

    assert set(edges["unit_id"]) == {"s0", "s1", "s2", "s3"}
    assert list(edges.columns) == [
        "unit_id",
        "regulator",
        "target",
        "weight",
        "weight_norm",
        "n_cells",
        "grn_status",
    ]
    assert set(neighbors["unit_id"]) == {"s0", "s1", "s2", "s3"}
    assert neighbors.loc[neighbors["unit_id"] == "s0", "neighbor_unit_id"].tolist() == ["s1", "s2"]
    assert set(summary["unit_id"]) == {"s0", "s1", "s2", "s3"}
    assert set(summary["n_cells"]) == {3}
