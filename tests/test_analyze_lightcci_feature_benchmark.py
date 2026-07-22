from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from scipy import sparse

from scripts.analyze_lightcci_feature_benchmark import (
    analyze_benchmark,
    sparse_entropy_decomposition,
)


METHODS = ("compare_N_kl", "compare_Ncomp_Gcos_v2")
TIMES = ("11.5->12.5", "12.5->13.5")
LEVELS = ("spot:seurat_k150", "seurat_k150:seurat_k40")


def _write_metrics(runs: Path, method: str, time_pair: str, values: list[float]) -> None:
    source, target = time_pair.split("->")
    root = runs / f"method={method}" / f"time={source}_to_{target}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for level_pair, gain in zip(LEVELS, values):
        lower, upper = level_pair.split(":")
        rows.append(
            {
                "network_method": "light_cci_grn",
                "pij_method": method,
                "organ": "heart",
                "lower_layer": lower,
                "upper_layer": upper,
                "time_pair": time_pair,
                "EI_lower": 1.0,
                "EI_upper": 1.0 + gain,
                "EI_gain": gain,
            }
        )
    pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)


def test_analyze_benchmark_validates_matrix_and_target_gate(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for method in METHODS:
        _write_metrics(runs, method, TIMES[0], [2.1, 2.2] if method.endswith("v2") else [0.1, -0.1])
        _write_metrics(runs, method, TIMES[1], [2.3, 2.4] if method.endswith("v2") else [0.2, 0.3])

    output = tmp_path / "comparison"
    gate = analyze_benchmark(
        runs_root=runs,
        output_dir=output,
        methods=METHODS,
        time_pairs=TIMES,
        level_pairs=LEVELS,
        target_positive_ratio=0.90,
        target_mean=2.0,
        split_name="test",
    )

    assert gate["any_method_passes"] is True
    summary = pd.read_csv(output / "method_summary.csv").set_index("pij_method")
    assert bool(summary.loc["compare_Ncomp_Gcos_v2", "passes_target"])
    assert not bool(summary.loc["compare_N_kl", "passes_target"])
    assert len(pd.read_csv(output / "all_metrics.csv")) == 8
    assert json.loads((output / "target_gate.json").read_text(encoding="utf-8"))["split"] == "test"
    assert "Method summary" in (output / "lightcci_feature_benchmark.md").read_text(encoding="utf-8")


def test_analyze_benchmark_refuses_missing_cell(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_metrics(runs, METHODS[0], TIMES[0], [0.1, 0.2])
    try:
        analyze_benchmark(
            runs_root=runs,
            output_dir=tmp_path / "comparison",
            methods=METHODS,
            time_pairs=TIMES,
            level_pairs=LEVELS,
            target_positive_ratio=0.90,
            target_mean=2.0,
            split_name="test",
        )
    except ValueError as exc:
        assert "Benchmark matrix mismatch" in str(exc)
    else:
        raise AssertionError("Missing benchmark cells must be rejected.")


def test_sparse_entropy_decomposition_matches_known_channels() -> None:
    deterministic = sparse.eye(8, format="csr")
    result = sparse_entropy_decomposition(deterministic)
    assert result["H_J"] == pytest.approx(3.0, abs=1e-9)
    assert result["H_J_given_I"] == pytest.approx(0.0, abs=1e-9)
    assert result["EI_recomputed"] == pytest.approx(3.0, abs=1e-9)
    assert result["mean_effective_row_support"] == pytest.approx(1.0, abs=1e-9)

    uniform = sparse.csr_matrix(np.full((4, 8), 1.0 / 8.0))
    result = sparse_entropy_decomposition(uniform)
    assert result["H_J"] == pytest.approx(3.0, abs=1e-9)
    assert result["H_J_given_I"] == pytest.approx(3.0, abs=1e-9)
    assert result["EI_recomputed"] == pytest.approx(0.0, abs=1e-9)
    assert result["mean_effective_row_support"] == pytest.approx(8.0, abs=1e-8)


def test_analyze_benchmark_can_require_entropy_artifacts(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for method in METHODS:
        for time_pair in TIMES:
            _write_metrics(runs, method, time_pair, [0.1, 0.2])
    with pytest.raises(ValueError, match="No pair-level"):
        analyze_benchmark(
            runs_root=runs,
            output_dir=tmp_path / "comparison",
            methods=METHODS,
            time_pairs=TIMES,
            level_pairs=LEVELS,
            target_positive_ratio=0.90,
            target_mean=2.0,
            split_name="test",
            require_entropy_artifacts=True,
        )
