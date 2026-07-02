from __future__ import annotations

import pandas as pd

from mignet_ce.visualization.common import apply_spatial_orientation
from mignet_ce.visualization.ei_existence import (
    LevelPair,
    build_time_pair_mean_table,
    compute_domain_parent_by_overlap,
    sample_membership_edges,
    time_pair_order,
)


def test_apply_spatial_orientation_reflects_coordinates() -> None:
    frame = pd.DataFrame({"x": [0.0, 10.0, 20.0], "y": [1.0, 5.0, 9.0]})

    inverted_y = apply_spatial_orientation(frame, invert_y=True)
    assert inverted_y["x"].tolist() == [0.0, 10.0, 20.0]
    assert inverted_y["y"].tolist() == [9.0, 5.0, 1.0]

    swapped_inverted_x = apply_spatial_orientation(frame, swap_xy=True, invert_x=True, invert_y=False)
    assert swapped_inverted_x["x"].tolist() == [9.0, 5.0, 1.0]
    assert swapped_inverted_x["y"].tolist() == [0.0, 10.0, 20.0]


def test_compute_domain_parent_by_overlap_uses_strongest_overlap() -> None:
    lower = pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3", "s4"],
            "domain_id": ["k150_a", "k150_a", "k150_b", "k150_b"],
        }
    )
    upper = pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3", "s4"],
            "domain_id": ["k40_a", "k40_b", "k40_b", "k40_b"],
        }
    )

    parents = compute_domain_parent_by_overlap(lower, upper)

    assert parents[["lower_domain_id", "upper_domain_id", "n_overlap"]].to_dict("records") == [
        {"lower_domain_id": "k150_a", "upper_domain_id": "k40_a", "n_overlap": 1},
        {"lower_domain_id": "k150_b", "upper_domain_id": "k40_b", "n_overlap": 2},
    ]


def test_build_time_pair_mean_table_uses_selected_level_pairs_and_duplicate_means() -> None:
    metrics = pd.DataFrame(
        [
            {"time_pair": "11.5->12.5", "lower_layer": "spot", "upper_layer": "seurat_k150", "EI_gain": 1.0},
            {"time_pair": "11.5->12.5", "lower_layer": "spot", "upper_layer": "seurat_k150", "EI_gain": 3.0},
            {"time_pair": "11.5->12.5", "lower_layer": "seurat_k150", "upper_layer": "seurat_k40", "EI_gain": 4.0},
            {"time_pair": "11.5->12.5", "lower_layer": "spot", "upper_layer": "seurat_k40", "EI_gain": 6.0},
            {"time_pair": "11.5->12.5", "lower_layer": "ignored", "upper_layer": "pair", "EI_gain": 100.0},
            {"time_pair": "11.5->13.5", "lower_layer": "spot", "upper_layer": "seurat_k150", "EI_gain": 2.0},
            {"time_pair": "11.5->13.5", "lower_layer": "seurat_k150", "upper_layer": "seurat_k40", "EI_gain": 5.0},
            {"time_pair": "11.5->13.5", "lower_layer": "spot", "upper_layer": "seurat_k40", "EI_gain": 8.0},
        ]
    )
    level_pairs = [
        LevelPair("spot", "seurat_k150"),
        LevelPair("seurat_k150", "seurat_k40"),
        LevelPair("spot", "seurat_k40"),
    ]

    table = build_time_pair_mean_table(metrics, time_pair_order(["11.5", "12.5", "13.5"]), level_pairs)

    first = table[table["time_pair"] == "11.5->12.5"].iloc[0]
    assert first["mean_EI_gain"] == 4.0
    assert first["n_level_pairs"] == 3
    second = table[table["time_pair"] == "11.5->13.5"].iloc[0]
    assert second["mean_EI_gain"] == 5.0
    missing = table[table["time_pair"] == "12.5->13.5"].iloc[0]
    assert pd.isna(missing["mean_EI_gain"])
    assert missing["n_level_pairs"] == 0


def test_sample_membership_edges_caps_each_group() -> None:
    edges = pd.DataFrame({"domain_id": ["a"] * 5 + ["b"] * 2, "value": range(7)})

    sampled = sample_membership_edges(edges, group_col="domain_id", max_per_group=3, random_state=1)

    assert sampled.groupby("domain_id").size().to_dict() == {"a": 3, "b": 2}

