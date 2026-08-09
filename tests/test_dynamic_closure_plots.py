from __future__ import annotations

import pandas as pd

from mignet_ce.visualization.downstream.dynamic_closure.ultradeep_plots import (
    plot_predictive_frontier,
)


def test_extended_plot_uses_shared_png_and_pdf_contract(tmp_path) -> None:
    rows = []
    pairs = ("11.5->12.5", "12.5->13.5", "13.5->14.5")
    for mapping, x in (
        ("Spot (uncompressed)", 0.0),
        ("Seurat K150", 0.01),
        ("Seurat K40", 0.02),
        ("Optimized coarse-graining", 0.03),
    ):
        for pair in pairs:
            rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "closure_floor_js": x,
                    "predictive_efficiency_bits_per_state_bit": 0.5,
                    "macro_sufficiency": 0.8,
                    "compression_fraction": 0.4,
                    "macro_entropy_bits": 3.0,
                    "predictive_information_bits": 1.5,
                    "effective_states": 8.0,
                    "independent_q_excess_js": 0.01,
                }
            )
    path = plot_predictive_frontier(pd.DataFrame(rows), tmp_path)
    assert path.name == "closure_predictive_compression_frontier.png"
    assert path.is_file()
    assert path.with_suffix(".pdf").is_file()
