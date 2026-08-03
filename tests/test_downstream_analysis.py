from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mignet_ce.visualization.downstream.analysis import _viterbi_path
from mignet_ce.visualization.downstream.config import DownstreamConfig
from mignet_ce.visualization.downstream.io import load_filtered_metrics


def test_viterbi_path_finds_global_maximum_instead_of_greedy_path() -> None:
    first = np.asarray([[0.6, 0.4], [0.5, 0.5]])
    second = np.asarray([[0.51, 0.49], [0.99, 0.01]])
    indices, probability = _viterbi_path([first, second], source_index=0)
    assert indices == [0, 1, 0]
    assert probability == pytest.approx(0.4 * 0.99)


def test_load_filtered_metrics_selects_exact_model_and_layer_pair(tmp_path: Path) -> None:
    rows = []
    times = ("11.5", "12.5", "13.5", "14.5")
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            pair = f"{times[i]}->{times[j]}"
            rows.append(
                {
                    "network_method": "light_cci_grn",
                    "pij_method": "NG_KLot",
                    "organ": "heart",
                    "lower_layer": "seurat_k150",
                    "upper_layer": "seurat_k40",
                    "time_pair": pair,
                    "EI_lower": 0.2,
                    "EI_upper": 0.5,
                    "EI_gain": 0.3,
                }
            )
            rows.append({**rows[-1], "lower_layer": "spot"})
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    cfg = DownstreamConfig(
        data_root=tmp_path,
        metrics_csv=metrics_path,
        pair_archive=tmp_path,
        output_dir=tmp_path / "out",
    )
    selected = load_filtered_metrics(cfg)
    assert len(selected) == 6
    assert set(selected["lower_layer"]) == {"seurat_k150"}
    assert selected["lag"].tolist() == [1, 2, 3, 1, 2, 1]


def test_config_rejects_non_four_time_point_figure_suite(tmp_path: Path) -> None:
    cfg = DownstreamConfig(
        data_root=tmp_path,
        metrics_csv=tmp_path,
        pair_archive=tmp_path,
        output_dir=tmp_path / "out",
        times=("11.5", "12.5", "13.5"),
    )
    with pytest.raises(ValueError, match="exactly four"):
        cfg.validate()
