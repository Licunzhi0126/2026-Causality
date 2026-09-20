from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mignet_ce.downstream.sensitive_analysis.analysis import (
    build_hierarchy_sensitivity,
    build_requested_wide_table,
)
from mignet_ce.downstream.sensitive_analysis.plots import LINE_ORDER, plot_alpha_sensitivity


def _layer_table() -> pd.DataFrame:
    rows = []
    for pair in ("11.5->12.5", "12.5->13.5", "11.5->13.5"):
        for alpha in (0.0, 0.1, 1.0):
            for layer, offset in (("spot", 0.0), ("seurat_k150", 0.2), ("seurat_k40", 0.5)):
                rows.append(
                    {
                        "time_pair": pair,
                        "alpha": alpha,
                        "layer": layer,
                        "EI_bits": 1.0 + alpha + offset,
                        "actual_grn_cost_share": 1.0 - alpha,
                        "tau": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_requested_table_and_hierarchy_identity() -> None:
    sensitivity = build_hierarchy_sensitivity(_layer_table())
    order = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
    wide = build_requested_wide_table(sensitivity, time_pair_order=order)
    assert list(wide.columns) == ["hierarchy_pair", "alpha", *order]
    for (_pair, _alpha), group in sensitivity.groupby(["time_pair", "alpha"]):
        values = group.set_index("hierarchy_pair")["delta_EI_bits"]
        assert np.isclose(
            values["spot:seuratK40"],
            values["spot:seuratK150"] + values["seuratK150:seuratK40"],
        )


def test_requested_figure_contains_three_data_curves(tmp_path) -> None:
    sensitivity = build_hierarchy_sensitivity(_layer_table())
    output = tmp_path / "sensitivity.png"
    plot_alpha_sensitivity(sensitivity, output)
    assert output.exists()
    assert len(LINE_ORDER) == 3
