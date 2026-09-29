from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad


LIB_DIR = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from unit_observation_counter import (
    count_domain_unit_observations,
    count_spot_unit_observations,
    discover_unit_grn_input_files,
    sample_name_from_unit_grn_input,
    summarize_unit_observation_counts,
)


def test_input_discovery_and_sample_names_are_shared_between_layers(tmp_path) -> None:
    spot_path = tmp_path / "spot" / "brain" / "spot_brain_11.5.h5ad"
    auxiliary_spot = tmp_path / "spot" / "brain" / "spot_brain_11.5_spots_with_domain.h5ad"
    domain_path = (
        tmp_path
        / "seurat_k150"
        / "brain"
        / "seurat150_brain_11.5_spots_with_domain.h5ad"
    )
    for path in (spot_path, auxiliary_spot, domain_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert discover_unit_grn_input_files(tmp_path, "spot") == [spot_path]
    assert discover_unit_grn_input_files(tmp_path, "seurat_k150") == [domain_path]
    assert sample_name_from_unit_grn_input(spot_path, "spot") == "spot_brain_11.5"
    assert (
        sample_name_from_unit_grn_input(domain_path, "seurat_k150")
        == "seurat150_brain_11.5"
    )


def test_domain_and_spot_counts_use_runner_semantics() -> None:
    domain = ad.AnnData(
        X=np.ones((5, 2)),
        obs=pd.DataFrame(
            {"domain_id": ["d1", "d1", "d1", "d2", "d2"]},
            index=[f"s{i}" for i in range(5)],
        ),
    )
    domain_counts = count_domain_unit_observations(
        domain,
        threshold=3,
    ).set_index("unit_id")
    assert domain_counts.loc["d1", "n_observations"] == 3
    assert bool(domain_counts.loc["d1", "below_threshold"]) is False
    assert bool(domain_counts.loc["d2", "below_threshold"]) is True

    spot = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame(index=["s0", "s1", "s2", "s3"]),
    )
    spot.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    spot_counts = count_spot_unit_observations(
        spot,
        spot_k_neighbors=2,
        include_center=True,
        threshold=3,
    )
    assert set(spot_counts["n_observations"]) == {3}
    assert not spot_counts["below_threshold"].any()


def test_sample_summary_reports_below_threshold_ratio() -> None:
    counts = pd.DataFrame(
        {
            "layer": ["seurat_k150", "seurat_k150"],
            "sample": ["sample", "sample"],
            "unit_id": ["d1", "d2"],
            "n_observations": [10, 40],
            "below_threshold": [True, False],
            "threshold": [30, 30],
            "unit_source": ["obs['domain_id']", "obs['domain_id']"],
            "input_file": ["input.h5ad", "input.h5ad"],
        }
    )

    summary = summarize_unit_observation_counts(counts)

    assert summary.loc[0, "n_units"] == 2
    assert summary.loc[0, "n_units_below_threshold"] == 1
    assert summary.loc[0, "below_threshold_ratio"] == 0.5
    assert summary.loc[0, "median_observations"] == 25.0


def test_observation_inspection_script_writes_required_reports(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    spot_path = data_root / "spot" / "brain" / "spot_brain_11.5.h5ad"
    spot_path.parent.mkdir(parents=True)
    spot = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame(index=["s0", "s1", "s2", "s3"]),
    )
    spot.obsm["spatial"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    spot.write_h5ad(spot_path)

    domain_path = (
        data_root
        / "seurat_k150"
        / "brain"
        / "seurat150_brain_11.5_spots_with_domain.h5ad"
    )
    domain_path.parent.mkdir(parents=True)
    domain = ad.AnnData(
        X=np.ones((5, 2)),
        obs=pd.DataFrame(
            {"domain_id": ["d1", "d1", "d1", "d2", "d2"]},
            index=[f"x{i}" for i in range(5)],
        ),
    )
    domain.write_h5ad(domain_path)

    output_root = tmp_path / "qc"
    script = (
        Path(__file__).resolve().parents[1]
        / "data_factory"
        / "scripts"
        / "inspect_unit_observation_counts.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--layers",
            "spot",
            "seurat_k150",
            "--output-root",
            str(output_root),
            "--min-cells-per-unit",
            "3",
            "--spot-k-neighbors",
            "2",
        ],
        check=True,
    )

    counts = pd.read_csv(output_root / "unit_observation_counts.csv")
    summary = pd.read_csv(output_root / "sample_unit_observation_summary.csv")
    below = pd.read_csv(output_root / "below_threshold_units.csv")
    assert set(counts["layer"]) == {"spot", "seurat_k150"}
    assert set(summary["sample"]) == {"spot_brain_11.5", "seurat150_brain_11.5"}
    assert below["unit_id"].tolist() == ["d2"]
