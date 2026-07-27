from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from mignet_ce.io.regsim_h5ad import (
    build_regsim_feature_block,
    canonicalize_regulatory_column,
    common_regulatory_features,
)


def test_regulatory_column_canonicalization() -> None:
    assert canonicalize_regulatory_column("Module_3") == "Module_3"
    assert canonicalize_regulatory_column("Regulon - Sox2") == "Regulon::Sox2"
    assert canonicalize_regulatory_column("Regulon...Sox2") == "Regulon::Sox2"
    assert canonicalize_regulatory_column("cell_type") is None


def test_regsim_spot_alignment_uses_cci_order(tmp_path) -> None:
    path_t = tmp_path / "t.h5ad"
    path_tp = tmp_path / "tp.h5ad"
    obs_t = pd.DataFrame(
        {
            "Module_1": [1.0, 2.0, 3.0],
            "Regulon - Sox2": [4.0, 5.0, 6.0],
        },
        index=["s2", "s1", "s3"],
    )
    obs_tp = pd.DataFrame(
        {
            "Module_1": [7.0, 8.0],
            "Regulon...Sox2": [9.0, 10.0],
        },
        index=["q1", "q2"],
    )
    ad.AnnData(np.ones((3, 1)), obs=obs_t).write_h5ad(path_t)
    ad.AnnData(np.ones((2, 1)), obs=obs_tp).write_h5ad(path_tp)
    features = common_regulatory_features([path_t, path_tp])
    assert features == ["Module_1", "Regulon::Sox2"]
    block = build_regsim_feature_block(
        unit_h5ad_path=path_t,
        cci_unit_ids=["s1", "s2"],
        feature_names=features,
    )
    np.testing.assert_allclose(block.values, [[2.0, 5.0], [1.0, 4.0]])
    assert block.unit_ids == ["s1", "s2"]


def test_regsim_domain_aggregation(tmp_path) -> None:
    domain_path = tmp_path / "domains.h5ad"
    spots_path = tmp_path / "spots_with_domain.h5ad"
    ad.AnnData(np.ones((2, 1)), obs=pd.DataFrame(index=["d1", "d2"])).write_h5ad(
        domain_path
    )
    spot_obs = pd.DataFrame(
        {
            "domain_id": ["d1", "d1", "d2"],
            "Module_1": [1.0, 3.0, 8.0],
        },
        index=["s1", "s2", "s3"],
    )
    ad.AnnData(np.ones((3, 1)), obs=spot_obs).write_h5ad(spots_path)
    block = build_regsim_feature_block(
        unit_h5ad_path=domain_path,
        cci_unit_ids=["d2", "d1"],
        feature_names=["Module_1"],
        spots_with_domain_h5ad_path=spots_path,
    )
    np.testing.assert_allclose(block.values[:, 0], [8.0, 2.0])
    assert block.metadata["aggregation"] == "mean_spot_activity_by_domain_id"
