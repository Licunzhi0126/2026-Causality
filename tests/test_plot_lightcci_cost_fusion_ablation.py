from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_lightcci_cost_fusion_ablation import (
    FEATURE_ORDER,
    METHOD_MAP,
    load_cost_fusion_metrics,
    plot_cost_fusion_ablation,
)


BASE_ROW = {
    "organ": "heart",
    "time_pair": "11.5->12.5",
    "lower_layer": "seurat_k40",
    "upper_layer": "seurat_k150",
    "EI_gain": 0.25,
}


def _write_metrics(root: Path, method: str, rows: list[dict[str, object]]) -> None:
    directory = root / "network=light_cci" / f"pij={method}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)


def test_fixed_twelve_methods_generate_one_six_by_two_png_and_ignore_concat(tmp_path: Path) -> None:
    for index, method in enumerate(METHOD_MAP):
        _write_metrics(tmp_path, method, [{**BASE_ROW, "EI_gain": index - 5.0}])
    _write_metrics(tmp_path, "compare_E_L_sot", [{**BASE_ROW, "EI_gain": 999.0}])

    metrics = load_cost_fusion_metrics(tmp_path)
    assert set(metrics["method"]) == set(METHOD_MAP)
    assert tuple(dict.fromkeys(group for group, _ in METHOD_MAP.values())) == FEATURE_ORDER
    assert 999.0 not in metrics["EI_gain"].tolist()
    paths = plot_cost_fusion_ablation(metrics, tmp_path / "figures")
    assert len(paths) == 1
    assert paths[0].is_file()
    assert paths[0].suffix == ".png"


def test_missing_euclidean_is_left_missing_not_filled_with_zero(tmp_path: Path) -> None:
    cosine_methods = [method for method, (_, distance) in METHOD_MAP.items() if distance == "Cosine"]
    for method in cosine_methods:
        _write_metrics(tmp_path, method, [BASE_ROW])
    metrics = load_cost_fusion_metrics(tmp_path)
    assert set(metrics["distance"]) == {"Cosine"}
    assert len(metrics) == len(cosine_methods)
    assert len(plot_cost_fusion_ablation(metrics, tmp_path / "figures")) == 1


def test_duplicate_rows_error_by_default_and_mean_only_when_requested(tmp_path: Path) -> None:
    method = next(iter(METHOD_MAP))
    _write_metrics(
        tmp_path,
        method,
        [{**BASE_ROW, "EI_gain": 1.0}, {**BASE_ROW, "EI_gain": 3.0}],
    )
    with pytest.raises(ValueError, match="Duplicate method/group rows"):
        load_cost_fusion_metrics(tmp_path)
    metrics = load_cost_fusion_metrics(tmp_path, duplicate_policy="mean")
    assert metrics.iloc[0]["EI_gain"] == pytest.approx(2.0)
