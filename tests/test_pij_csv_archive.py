from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.pij_exports import export_pij_csv_archive, pij_archive_directory
from mignet_ce.pij.base import TransitionKernels


def test_pij_csv_archive_uses_network_method_and_pij_method_directories(tmp_path) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs" / "experiment",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        network_method="legacy_mixed_grn_cci",
        pij_method="sr_ot",
        export_pij_topk=1,
    )
    pair = VerticalPairSpec("spot", "louvain_less_than5")
    matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    kernels = TransitionKernels(
        p_lower={(0, 1): matrix},
        p_upper={(0, 1): matrix},
        kernel_metadata={"pij_method": "sr_ot"},
    )
    kernels.kernel_diagnostics["lower"][(0, 1)] = {"sr_cost": np.zeros((2, 2))}
    kernels.kernel_diagnostics["upper"][(0, 1)] = {"sr_cost": np.zeros((2, 2))}

    archive_dir = export_pij_csv_archive(
        cfg=cfg,
        organ="heart",
        pair=pair,
        stable_upper_units=["u1", "u2"],
        kernels=kernels,
    )

    expected = (
        tmp_path
        / "dataset"
        / "pij"
        / "network=legacy_mixed_grn_cci"
        / "pij=sr_ot"
        / "organ=heart"
        / "pair=spot_to_louvain_less_than5"
    )
    assert archive_dir == expected
    assert pij_archive_directory(cfg, "heart", pair) == expected
    lower_path = expected / "11.5_to_12.5_lower_P_topk.csv"
    upper_path = expected / "11.5_to_12.5_upper_P_topk.csv"
    assert lower_path.exists()
    assert upper_path.exists()
    lower = pd.read_csv(lower_path)
    assert len(lower) == 2
    assert set(lower["pij_method"]) == {"sr_ot"}
    assert "sr_cost" in lower.columns
    with (expected / "kernel_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["network_method"] == "legacy_mixed_grn_cci"
    assert metadata["pij_method"] == "sr_ot"
    assert metadata["kernel_metadata"]["pij_method"] == "sr_ot"


def test_pair_artifacts_are_disabled_by_default() -> None:
    assert TemporalRunConfig().export_pair_artifacts is False
