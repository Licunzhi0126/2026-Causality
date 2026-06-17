from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.io.loaders import LayerPaths
from mignet_ce.networks.cross_cell_multilayer import (
    _build_target_aware_feature_names,
    _identity_projection_weights,
    _project_cci_in_from_targets,
    _project_cci_out_to_targets,
    _project_grn_to_targets,
    _project_lr_to_targets,
)


def test_target_aware_feature_names_build_four_stable_upper_blocks() -> None:
    names, blocks = _build_target_aware_feature_names(["U1", "U2", "U3"])

    assert len(names) == 12
    assert blocks == {
        "grn_target": ["grntarget_to_U1", "grntarget_to_U2", "grntarget_to_U3"],
        "cci_out_target": ["cciout_to_U1", "cciout_to_U2", "cciout_to_U3"],
        "cci_in_source": ["cciin_from_U1", "cciin_from_U2", "cciin_from_U3"],
        "lr_target": ["lr_to_U1", "lr_to_U2", "lr_to_U3"],
    }
    assert names == [name for block in blocks.values() for name in block]


def test_projection_helpers_use_target_coordinate_space() -> None:
    cci = sp.csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    weights = np.eye(2, dtype=float)

    np.testing.assert_allclose(_project_grn_to_targets(np.array([5.0, 7.0]), weights), np.array([[5.0, 0.0], [0.0, 7.0]]))
    np.testing.assert_allclose(_project_cci_out_to_targets(cci, weights), cci.toarray())
    np.testing.assert_allclose(_project_cci_in_from_targets(cci, weights), cci.toarray().T)


def test_identity_projection_handles_missing_current_units() -> None:
    weights = _identity_projection_weights(["U2", "missing"], ["U1", "U2"])

    np.testing.assert_allclose(weights, np.array([[0.0, 1.0], [0.0, 0.0]]))


def test_lr_projection_defaults_to_no_grn_gate(tmp_path) -> None:
    cci_dir = tmp_path / "cci"
    lr_dir = cci_dir / "sample_COMMOT_by_LR"
    lr_dir.mkdir(parents=True)
    pd.DataFrame({"unit": ["u1", "u2"]}).to_csv(cci_dir / "sample_index.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "filename": ["lr0.npz"],
            "ligand": ["Lig"],
            "receptor": ["Rec"],
            "lr_key": ["Lig-Rec"],
        }
    ).to_csv(cci_dir / "sample_COMMOT_lr_pairs.tsv", sep="\t", index=False)
    sp.save_npz(lr_dir / "lr0.npz", sp.csr_matrix(np.array([[0.0, 2.0], [0.0, 0.0]])))
    paths = LayerPaths(
        layer="toy",
        organ="heart",
        stage="11.5",
        sample_stem="sample",
        candidate_sample_stems=["sample"],
        h5ad=tmp_path / "sample.h5ad",
        grn_edges=tmp_path / "grn_edges.csv",
        cci_total=cci_dir / "sample_CCI_total.npz",
        cci_manifest=cci_dir / "sample_COMMOT_lr_pairs.tsv",
        cci_index=cci_dir / "sample_index.tsv",
        cci_lr_dir=lr_dir,
        spot_domain_map=None,
    )
    expr = pd.DataFrame({"Lig": [3.0, 0.0], "Rec": [0.0, 5.0]}, index=["u1", "u2"])
    projection = np.eye(2, dtype=float)
    grn = pd.DataFrame(columns=["regulator", "target", "weight", "grn_weight_norm"])

    cfg = TemporalRunConfig(cross_cell_top_k_edges=10)
    features, edges, mode = _project_lr_to_targets("toy", "11.5", expr, paths, projection, grn, cfg)

    assert mode == "lr_no_grn_gate"
    np.testing.assert_allclose(features, np.array([[0.0, 30.0], [0.0, 0.0]]))
    assert len(edges) == 1

    gate_cfg = TemporalRunConfig(cross_cell_top_k_edges=10, cross_cell_lr_use_grn_gate=True)
    gated_features, gated_edges, gated_mode = _project_lr_to_targets("toy", "11.5", expr, paths, projection, grn, gate_cfg)

    assert gated_mode == "lr_grn_gate"
    np.testing.assert_allclose(gated_features, np.zeros((2, 2), dtype=float))
    assert gated_edges.empty
