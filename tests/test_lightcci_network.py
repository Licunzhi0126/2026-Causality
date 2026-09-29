from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.compare._shared.features import adjacency_from_lightcci_graph


def _write_h5ad(path: Path, units: list[str], genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expr = np.arange(len(units) * len(genes), dtype=float).reshape(len(units), len(genes)) + 1.0
    obs = pd.DataFrame(index=pd.Index(units, name="unit_id"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=expr, obs=obs, var=var)
    adata.obsm["spatial"] = np.array([[float(idx), float(idx + 10)] for idx in range(len(units))])
    adata.write_h5ad(path)


def _write_cci(data_root: Path, layer: str, stage: str, units: list[str], matrix: np.ndarray) -> None:
    cci_dir = data_root / "cci" / layer
    cci_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{layer}_{'heart'}_{stage}"
    sp.save_npz(cci_dir / f"{stem}_CCI_total.npz", sp.csr_matrix(matrix))
    pd.DataFrame({"domain_id": units}).to_csv(cci_dir / f"{stem}_index.tsv", sep="\t", index=False)


def _write_spot_grn_for_gene_layer(data_root: Path, stage: str) -> None:
    stem = f"spot_heart_{stage}"
    grn_dir = data_root / "grn" / "spot" / stem
    grn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "regulator": ["g2", "g1", "g3"],
            "target": ["g1", "g3", "g2"],
            "weight": [-2.0, 1.0, 0.0],
        }
    ).to_csv(grn_dir / "grn_edges.csv", index=False)


def test_lightcci_builder_reads_gene_grn_and_spot_cci(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    stages = ["11.5", "12.5"]
    genes = ["g1", "g2", "g3"]
    for stage in stages:
        _write_h5ad(data_root / "spot" / "heart" / f"spot_heart_{stage}.h5ad", ["s2", "s1"], genes)
        _write_cci(data_root, "spot", stage, ["s1", "s2"], np.array([[0.0, 2.0], [3.0, 0.0]]))
        _write_spot_grn_for_gene_layer(data_root, stage)

    cfg = TemporalRunConfig(
        data_root=data_root,
        organs=["heart"],
        time_points=stages,
        level_pairs=[VerticalPairSpec("gene", "spot")],
        network_method="light_cci",
    )
    pair = VerticalPairSpec("gene", "spot")
    context = get_network_builder("light_cci").build_pair_context("heart", pair, cfg, LayerDataResolver(data_root))

    assert context.network_method == "light_cci"
    assert context.feature_alignment_space == "native_units"
    assert context.metadata["feature_source"] == "light_cci_graph_only"
    assert context.lower_graphs[0].metadata["edge_source"] == "grn"
    assert context.lower_graphs[0].metadata["grn_source_layer"] == "spot"
    assert context.lower_graphs[0].metadata["grn_source_sample_stem"] == "spot_heart_11.5"
    assert Path(str(context.lower_graphs[0].metadata["adjacency_path"])).parts[-4:-2] == ("grn", "spot")
    assert context.upper_graphs[0].metadata["edge_source"] == "cci"
    assert set(context.lower_units_by_time[0]) == {"g1", "g2", "g3"}
    assert context.upper_units_by_time[0] == ["s1", "s2"]
    assert context.lower_mats[0].shape == (3, 0)
    assert context.upper_mats[0].shape == (2, 0)

    lower_adj, lower_meta = adjacency_from_lightcci_graph(context.lower_graphs[0], context.lower_units_by_time[0])
    upper_adj, upper_meta = adjacency_from_lightcci_graph(context.upper_graphs[0], context.upper_units_by_time[0])

    assert lower_meta["edge_source"] == "grn"
    assert upper_meta["edge_source"] == "cci"
    assert lower_adj.nnz == 2
    assert sorted(lower_adj.data.tolist()) == [1.0, 2.0]
    assert upper_adj.toarray().tolist() == [[0.0, 2.0], [3.0, 0.0]]
