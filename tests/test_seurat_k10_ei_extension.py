from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))

from factory_common import parse_sample_stem  # noqa: E402
from layer_specs import get_domain_layer_spec  # noqa: E402
from mignet_ce.config import LAYER_SPECS, PAIR_PRESETS  # noqa: E402
from mignet_ce.downstream.analysis.visualization import ei_existence  # noqa: E402
from scripts.assemble_ei_existence_metrics import BASE_PAIRS, K10_PAIRS, assemble_metrics  # noqa: E402


def _metrics_rows(pairs: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    ei = {"spot": 1.0, "seurat_k150": 2.0, "seurat_k40": 3.0, "seurat_k10": 4.0}
    return pd.DataFrame(
        [
            {
                "network_method": "light_cci_grn",
                "pij_method": "NG_KLot",
                "organ": "heart",
                "lower_layer": lower,
                "upper_layer": upper,
                "time_pair": "11.5->12.5",
                "EI_lower": ei[lower],
                "EI_upper": ei[upper],
                "EI_gain": ei[upper] - ei[lower],
            }
            for lower, upper in pairs
        ]
    )


def test_k10_layer_is_registered_for_factory_and_vertical_ei() -> None:
    spec = get_domain_layer_spec("seurat_k10")
    assert (spec.family, spec.mode, spec.k, spec.sample_prefix) == ("seurat", "exact_k", 10, "seurat10")
    assert parse_sample_stem("seurat10_heart_11.5") == ("heart", "11.5")
    assert LAYER_SPECS["seurat_k10"].sample_stem("heart", "11.5") == "seurat10_heart_11.5"
    assert len(PAIR_PRESETS["seurat_k10_all"]) == 6


def test_assembly_requires_complete_consistent_grids(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    addition_path = tmp_path / "addition.csv"
    _metrics_rows(BASE_PAIRS).to_csv(base_path, index=False)
    addition = _metrics_rows(K10_PAIRS)
    addition.to_csv(addition_path, index=False)

    combined = assemble_metrics(base_path, addition_path, time_points=("11.5", "12.5"))
    assert len(combined) == 6
    assert set(combined["upper_layer"]) == {"seurat_k150", "seurat_k40", "seurat_k10"}

    addition.loc[addition["lower_layer"] == "seurat_k150", "EI_lower"] = 2.5
    addition.loc[addition["lower_layer"] == "seurat_k150", "EI_gain"] = 1.5
    addition.to_csv(addition_path, index=False)
    with pytest.raises(ValueError, match="changes across layer comparisons"):
        assemble_metrics(base_path, addition_path, time_points=("11.5", "12.5"))


def test_k10_figure_pipeline_uses_real_seurat_maps_and_adjacent_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "factory"
    result_root = tmp_path / "results"
    result_root.mkdir()
    metrics = pd.concat([_metrics_rows(BASE_PAIRS), _metrics_rows(K10_PAIRS)], ignore_index=True)
    metrics.to_csv(result_root / "metrics.csv", index=False)

    spots = [f"s{index}" for index in range(12)]
    for stage in ("11.5", "12.5"):
        for layer, prefix, divisor in (
            ("seurat_k150", "seurat150", 2),
            ("seurat_k40", "seurat", 4),
            ("seurat_k10", "seurat10", 6),
        ):
            directory = data_root / layer / "heart"
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "spot_id": spots,
                    "domain_id": [f"domain_{index // divisor:03d}" for index in range(12)],
                    "x": range(12),
                    "y": [index % 3 for index in range(12)],
                }
            ).to_csv(directory / f"{prefix}_heart_{stage}_spot_domain_map.csv", index=False)

    def fake_slice(_slice_root: Path, _stage: str, _organ: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spot_id": spots,
                "x": list(range(12)),
                "y": [index % 3 for index in range(12)],
                "annotation": ["Heart"] * 12,
                "is_target_organ": [True] * 12,
            }
        )

    monkeypatch.setattr(ei_existence, "load_full_slice", fake_slice)
    output_dir = tmp_path / "figures"
    paths = ei_existence.generate_ei_existence_figures(
        data_root=data_root,
        slice_root=tmp_path / "slices",
        result_root=result_root,
        output_dir=output_dir,
        organ="heart",
        network_method="light_cci_grn",
        pij_method="NG_KLot",
        time_points=("11.5", "12.5"),
        level_pairs=ei_existence.parse_level_pairs(ei_existence.K10_LEVEL_PAIRS),
        dpi=45,
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in (paths.fig1, paths.fig2, paths.fig3))
    heatmap = pd.read_csv(paths.fig2_table)
    timeline = pd.read_csv(paths.fig3_table)
    assert len(heatmap.columns) == 7
    assert timeline.loc[0, "n_level_pairs"] == 3
    assert timeline.loc[0, "mean_EI_gain"] == 1.0

    assert ei_existence.find_domain_map(data_root, "seurat_k150", "heart", "11.5") == (
        data_root / "seurat_k150" / "heart" / "seurat150_heart_11.5_spot_domain_map.csv"
    )
