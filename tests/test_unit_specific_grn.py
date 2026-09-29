from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.graph.builder import build_layer_graph
from mignet_ce.io.loaders import ExpressionData, LayerPaths

LIB_DIR = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from unit_grn_layer_runner import infer_unit_grn_tables, normalize_edge_weights


def _fake_infer(adata, _grn):
    edges = pd.DataFrame(
        {
            "regulator": ["A", "A"],
            "target": ["B", "C"],
            "weight": [2.0, 1.0],
        }
    )
    return edges, {
        "n_cells": int(adata.n_obs),
        "n_genes_used": int(adata.n_vars),
        "n_regulators_used": 1,
        "n_edges": 2,
        "expression_source": "X",
    }


def test_normalize_edge_weights_is_unit_local() -> None:
    normalized = normalize_edge_weights(
        pd.DataFrame(
            {
                "regulator": ["A", "A"],
                "target": ["B", "C"],
                "weight": [1.0, 3.0],
            }
        )
    )
    assert normalized["weight_norm"].iloc[0] < normalized["weight_norm"].iloc[1]
    assert normalized["weight_norm"].iloc[1] == 1.0


def test_unit_grn_runner_groups_original_spots_by_domain() -> None:
    adata = ad.AnnData(
        X=np.ones((5, 3)),
        obs=pd.DataFrame(
            {"domain_id": ["d1", "d1", "d1", "d2", "d2"]},
            index=[f"s{i}" for i in range(5)],
        ),
    )
    adata.var_names = ["A", "B", "C"]

    edges, summary = infer_unit_grn_tables(
        adata,
        min_cells_per_unit=3,
        infer_fn=_fake_infer,
    )

    assert set(edges["unit_id"]) == {"d1"}
    assert list(edges.columns) == [
        "unit_id",
        "regulator",
        "target",
        "weight",
        "weight_norm",
        "n_cells",
        "grn_status",
    ]
    status = summary.set_index("unit_id")["status"].to_dict()
    assert status == {"d1": "written", "d2": "skipped"}


def test_layer_graph_reads_unit_specific_weights(tmp_path) -> None:
    grn_path = tmp_path / "grn_edges.csv"
    pd.DataFrame(
        {"regulator": ["A"], "target": ["B"], "weight": [1.0]}
    ).to_csv(grn_path, index=False)
    unit_grn_path = tmp_path / "unit_grn_edges.csv"
    pd.DataFrame(
        {
            "unit_id": ["u1", "u2"],
            "regulator": ["A", "A"],
            "target": ["B", "B"],
            "weight": [4.0, 2.0],
            "weight_norm": [0.9, 0.3],
        }
    ).to_csv(unit_grn_path, index=False)
    cci_dir = tmp_path / "cci"
    lr_dir = cci_dir / "lr"
    lr_dir.mkdir(parents=True)
    pd.DataFrame({"unit": ["u1", "u2"]}).to_csv(
        cci_dir / "index.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "filename": ["lr.npz"],
            "ligand": ["A"],
            "receptor": ["B"],
            "lr_key": ["A-B"],
        }
    ).to_csv(cci_dir / "manifest.tsv", sep="\t", index=False)
    sp.save_npz(lr_dir / "lr.npz", sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]]))
    paths = LayerPaths(
        layer="spot",
        organ="heart",
        stage="11.5",
        sample_stem="sample",
        candidate_sample_stems=["sample"],
        h5ad=tmp_path / "sample.h5ad",
        grn_edges=grn_path,
        cci_total=cci_dir / "total.npz",
        cci_manifest=cci_dir / "manifest.tsv",
        cci_index=cci_dir / "index.tsv",
        cci_lr_dir=lr_dir,
        spot_domain_map=None,
        unit_grn_edges=unit_grn_path,
    )
    expression = ExpressionData(
        units=["u1", "u2"],
        genes=["A", "B"],
        expr=pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["u1", "u2"], columns=["A", "B"]),
        coords=pd.DataFrame([[0.0, 0.0], [1.0, 0.0]], index=["u1", "u2"], columns=["x", "y"]),
        obs=pd.DataFrame(index=["u1", "u2"]),
    )

    graph = build_layer_graph(
        layer_name="spot",
        time_point="11.5",
        expression=expression,
        paths=paths,
        shared_genes=["A", "B"],
        inter_influence_mode="cci_only",
        inter_grn_pair_policy="zero_if_missing",
        cci_inter_use_expression_mask=False,
        grn_source="unit_specific",
    )

    weights = graph.intra_edges.set_index("src_unit")["influence_score"].to_dict()
    assert weights == {"u1": 0.9, "u2": 0.3}
    assert graph.metadata["unit_grn_file_found"] is True
    assert graph.metadata["unit_specific_fallback_units"] == []
