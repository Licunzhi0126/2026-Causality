from __future__ import annotations

import pandas as pd

from mignet_ce.config import LAYER_SPECS, PAIR_PRESETS
from mignet_ce.mapping import (
    UnitAssignments,
    build_overlap_edge_table,
    build_overlap_mapping,
    build_spot_correspondence_table,
    summarize_overlap_quality,
)


def test_pair_presets_include_seurat_and_louvain_cross_level_pairs() -> None:
    assert "louvain_k40" in LAYER_SPECS
    assert "seurat_k150" in LAYER_SPECS
    assert "seurat_less_than5" in LAYER_SPECS
    assert "spatial_domain_less_than5" in LAYER_SPECS
    assert "spatial_domain_k150" in LAYER_SPECS
    assert "spatial_domain_k40" in LAYER_SPECS
    assert ("seurat", "seurat40") == LAYER_SPECS["seurat_k40"].sample_prefixes
    assert ("spatialDomainLessThan5",) == LAYER_SPECS["spatial_domain_less_than5"].sample_prefixes
    assert ("spatialDomain150",) == LAYER_SPECS["spatial_domain_k150"].sample_prefixes
    assert ("spatialDomain40",) == LAYER_SPECS["spatial_domain_k40"].sample_prefixes

    louvain_all = {(pair.lower_layer, pair.upper_layer) for pair in PAIR_PRESETS["louvain_all"]}
    seurat_all = {(pair.lower_layer, pair.upper_layer) for pair in PAIR_PRESETS["seurat_all"]}
    spatial_domain_all = {(pair.lower_layer, pair.upper_layer) for pair in PAIR_PRESETS["spatial_domain_all"]}
    assert ("spot", "louvain_k40") in louvain_all
    assert ("louvain_less_than5", "louvain_k40") in louvain_all
    assert ("spot", "seurat_k40") in seurat_all
    assert ("seurat_less_than5", "seurat_k40") in seurat_all
    assert ("spot", "spatial_domain_less_than5") in spatial_domain_all
    assert ("spatial_domain_less_than5", "spatial_domain_k150") in spatial_domain_all
    assert ("spatial_domain_k150", "spatial_domain_k40") in spatial_domain_all
    assert ("spot", "spatial_domain_k150") in spatial_domain_all
    assert ("spot", "spatial_domain_k40") in spatial_domain_all
    assert ("spatial_domain_less_than5", "spatial_domain_k40") in spatial_domain_all


def test_overlap_tables_report_spot_correspondence_and_quality() -> None:
    lower = UnitAssignments(
        layer="lower",
        rows=pd.DataFrame(
            {
                "spot_id": ["s1", "s2", "s3", "s4"],
                "unit_id": ["a", "a", "b", "b"],
                "organ": ["heart", "heart", "heart", "heart"],
            }
        ),
    )
    upper = UnitAssignments(
        layer="upper",
        rows=pd.DataFrame(
            {
                "spot_id": ["s1", "s2", "s3", "s4"],
                "unit_id": ["x", "y", "y", "z"],
            }
        ),
    )

    spot_table = build_spot_correspondence_table(lower, upper, "11.5", "lower", "upper")
    assert list(spot_table["lower_unit"]) == ["a", "a", "b", "b"]
    assert list(spot_table["upper_unit"]) == ["x", "y", "y", "z"]

    overlap = build_overlap_mapping(lower, upper, lower_units=["a", "b"], upper_units=["x", "y", "z"])
    edge_table = build_overlap_edge_table(overlap, "11.5", "lower", "upper")
    assert set(edge_table["lower_unit"]) == {"a", "b"}
    assert edge_table.loc[edge_table["lower_unit"] == "a", "overlap_spot_count"].sum() == 2.0

    quality = summarize_overlap_quality(edge_table)
    assert quality["n_lower_units"] == 2
    assert quality["n_upper_units"] == 3
    assert quality["n_overlap_edges"] == 4
    assert quality["matched_lower_units"] == 2
