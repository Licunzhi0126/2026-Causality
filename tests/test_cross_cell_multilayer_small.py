from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.networks.cross_cell_multilayer import CrossCellMultilayerBuilder


ad = pytest.importorskip("anndata")


def _write_h5ad(path, units: list[str], genes: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.asarray(matrix, dtype=float),
        obs=pd.DataFrame(index=units),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = np.column_stack([np.arange(len(units), dtype=float), np.zeros(len(units), dtype=float)])
    adata.write_h5ad(path)


def _write_layer_inputs(root, layer: str, organ: str, stage: str, sample: str, units: list[str], genes: list[str]) -> None:
    _write_h5ad(
        root / layer / organ / f"{sample}.h5ad",
        units,
        genes,
        np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.5], [1.5, 1.0, 2.0]])[: len(units), :],
    )
    grn_dir = root / "grn" / layer / sample
    grn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "regulator": ["Lig", "Rec", "Lig"],
            "target": ["Rec", "Tar", "Tar"],
            "weight": [1.0, 0.8, 0.5],
        }
    ).to_csv(grn_dir / "grn_edges.csv", index=False)

    cci_dir = root / "cci" / layer
    lr_dir = cci_dir / f"{sample}_COMMOT_by_LR"
    lr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit": units}).to_csv(cci_dir / f"{sample}_index.tsv", sep="\t", index=False)
    mat = sp.csr_matrix(np.ones((len(units), len(units)), dtype=float) - np.eye(len(units), dtype=float))
    sp.save_npz(cci_dir / f"{sample}_CCI_total.npz", mat)
    sp.save_npz(lr_dir / "lr0.npz", mat)
    pd.DataFrame(
        {
            "filename": ["lr0.npz"],
            "ligand": ["Lig"],
            "receptor": ["Rec"],
            "lr_key": ["Lig-Rec"],
        }
    ).to_csv(cci_dir / f"{sample}_COMMOT_lr_pairs.tsv", sep="\t", index=False)


def _write_tiny_vertical_data(root) -> None:
    genes = ["Lig", "Rec", "Tar"]
    for stage in ("11.5", "12.5"):
        _write_layer_inputs(root, "spot", "heart", stage, f"spot_heart_{stage}", ["s1", "s2", "s3"], genes)
        _write_layer_inputs(
            root,
            "louvain_less_than5",
            "heart",
            stage,
            f"louvainLessThan5_heart_{stage}",
            ["d1", "d2"],
            genes,
        )
        map_dir = root / "louvain_less_than5" / "heart"
        map_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "spot_id": ["s1", "s2", "s3"],
                "domain_id": ["d1", "d1", "d2"],
            }
        ).to_csv(map_dir / f"louvainLessThan5_heart_{stage}_spot_domain_map.csv", index=False)


def test_cross_cell_multilayer_builds_target_aware_feature_blocks(tmp_path) -> None:
    _write_tiny_vertical_data(tmp_path)
    cfg = TemporalRunConfig(
        data_root=tmp_path,
        output_root=tmp_path / "out",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[VerticalPairSpec("spot", "louvain_less_than5")],
        network_method="cross_cell_multilayer",
        cross_cell_top_k_edges=20,
        export_features=False,
    )
    context = CrossCellMultilayerBuilder().build_pair_context(
        organ="heart",
        pair=VerticalPairSpec("spot", "louvain_less_than5"),
        cfg=cfg,
        resolver=LayerDataResolver(tmp_path),
    )

    assert context.network_method == "cross_cell_multilayer"
    assert set(context.feature_blocks) == {"grn_target", "cci_out_target", "cci_in_source", "lr_target"}
    assert all(len(block) == len(context.stable_upper_units) for block in context.feature_blocks.values())
    assert context.feature_names == [name for block in context.feature_blocks.values() for name in block]
    assert context.lower_mats[0].shape == (3, 4 * len(context.stable_upper_units))
    assert context.upper_mats[0].shape == (2, 4 * len(context.stable_upper_units))
    assert context.metadata["cross_cell_feature_mode"] == "target_aware_multichannel"
    assert context.metadata["ddi_handling"] == "disabled_in_target_aware_mode"
    assert "network_exports/11.5_ddi_edges.csv" not in context.exports
    assert "network_exports/11.5_lower_target_aware_lr_edges_topk.csv" in context.exports
