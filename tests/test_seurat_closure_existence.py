from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
from mignet_ce.downstream.analysis.dynamic_closure.seurat_existence import (
    EXPECTED_K,
    SEURAT_LAYERS,
    SeuratClosureConfig,
    _bh_adjust,
    _hard_information_closure_budget,
    _matched_source_partition_null,
    _metric_row,
    _retained_from_hard_labels,
    run_seurat_closure_existence,
)


def _one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    result = np.zeros((len(labels), k), dtype=float)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def _write_index(path: Path, units: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": units}).to_csv(path, sep="\t", index=False)


def _write_units(path: Path, units: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": range(len(units)), "unit": units}).to_csv(path, index=False)


def _synthetic_archive(
    tmp_path: Path,
    times: tuple[str, ...] = ("11.5", "12.5"),
) -> tuple[Path, Path, np.ndarray]:
    data_root = tmp_path / "data"
    archive_root = tmp_path / "pij"
    n_spots = 150
    spots = [f"spot_{index:03d}" for index in range(n_spots)]
    p_micro = 0.7 * np.eye(n_spots) + 0.3 / n_spots
    for time in times:
        _write_index(data_root / "cci" / "spot" / f"spot_heart_{time}_index.tsv", spots)
    for layer in SEURAT_LAYERS:
        k = EXPECTED_K[layer]
        domains = [f"{layer}_{index:03d}" for index in range(k)]
        assignments = _one_hot(np.arange(n_spots) % k, k)
        for time in times:
            prefix = {"seurat_k150": "seurat150", "seurat_k40": "seurat", "seurat_k10": "seurat10"}[layer]
            _write_index(data_root / "cci" / layer / f"{prefix}_heart_{time}_index.tsv", domains)
            map_path = data_root / layer / "heart" / f"{prefix}_heart_{time}_spot_domain_map.csv"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "spot_id": spots,
                    "domain_id": [domains[index % k] for index in range(n_spots)],
                    "x": np.arange(n_spots),
                    "y": np.arange(n_spots) % 5,
                }
            ).to_csv(map_path, index=False)
        archive_dir = (
            archive_root / "network=light_cci_grn" / "pij=NG_KLot"
            / "organ=heart" / f"pair=spot_to_{layer}"
        )
        archive_dir.mkdir(parents=True)
        mappings = {}
        q_direct = (assignments.T @ p_micro @ assignments) / assignments.sum(axis=0)[:, None]
        for source_time, target_time in combinations(times, 2):
            for side, matrix, units in (("lower", p_micro, spots), ("upper", q_direct, domains)):
                label = f"{source_time}_to_{target_time}_{side}_P.npz"
                sp.save_npz(archive_dir / label, sp.csr_matrix(matrix))
                source_rel = f"units/{side}_{source_time}_units.csv"
                target_rel = f"units/{side}_{target_time}_units.csv"
                _write_units(archive_dir / source_rel, units)
                _write_units(archive_dir / target_rel, units)
                mappings[label] = {"source_units": source_rel, "target_units": target_rel}
        (archive_dir / "kernel_metadata.json").write_text(
            json.dumps(
                {
                    "network_method": "light_cci_grn",
                    "pij_method": "NG_KLot",
                    "organ": "heart",
                    "lower_layer": "spot",
                    "upper_layer": layer,
                    "feature_alignment_space": "native_units",
                    "full_matrix": True,
                    "time_points": list(times),
                    "unit_mapping_files": mappings,
                }
            ),
            encoding="utf-8",
        )
    return data_root, archive_root, p_micro


def test_fast_hard_partition_null_matches_shared_information_budget() -> None:
    p = np.asarray([[0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7], [0.0, 0.0, 0.2, 0.8]])
    source = _one_hot(np.asarray([0, 0, 1, 1]), 2)
    target = _one_hot(np.asarray([0, 0, 1, 1]), 2)
    budget = information_closure_budget(p, source, target)
    hard_budget = _hard_information_closure_budget(p, source, target)
    for key in ("I_available_bits", "I_macro_retained_bits", "closure_leakage_bits", "closure_quality"):
        assert hard_budget[key] == pytest.approx(budget[key])
    assert np.allclose(hard_budget["q_induced"], budget["q_induced"])
    retained = _retained_from_hard_labels(
        budget["observed"], np.asarray([0, 0, 1, 1]), np.asarray([2, 2]), budget["observed"].mean(axis=0)
    )
    assert retained == pytest.approx(budget["I_macro_retained_bits"])
    rows, summary = _matched_source_partition_null(
        budget["observed"], source, budget["I_available_bits"], budget["closure_quality"],
        budget["I_macro_retained_bits"], repeats=19, seed=7, layer="seurat_k10", pair="11.5->12.5",
    )
    assert len(rows) == 20
    assert summary["p_empirical"] >= 1 / 20
    assert _bh_adjust(np.asarray([0.001, 0.02, np.nan, 0.5])).tolist()[:2] == pytest.approx([0.003, 0.03])


def test_seurat_closure_pipeline_writes_cg_and_closure_tables(tmp_path: Path) -> None:
    times = ("11.5", "12.5", "13.5", "14.5")
    data_root, archive_root, _ = _synthetic_archive(tmp_path, times)
    cfg = SeuratClosureConfig(
        data_root=data_root,
        pij_archive_root=archive_root,
        output_root=tmp_path / "results",
        times=times,
        null_repeats=3,
        seed=10,
    )
    outputs = run_seurat_closure_existence(cfg)
    metrics = pd.read_csv(outputs["closure_metrics"])
    null = pd.read_csv(outputs["null_distribution"])
    audit = pd.read_csv(outputs["input_audit"])
    assert len(metrics) == 18
    assert len(null) == 18 * 4
    assert metrics["time_pair"].nunique() == 6
    assert set(metrics["macro_layer"]) == set(SEURAT_LAYERS)
    assert set(metrics["cg_pair"]) == {f"spot:{layer}" for layer in SEURAT_LAYERS}
    assert (metrics["source_states"] == metrics["macro_layer"].map(EXPECTED_K)).all()
    assert np.allclose(
        metrics["delta_EI_vertical_pair_bits"],
        metrics["EI_macro_direct_bits"] - metrics["EI_micro_pair_spot_bits"],
    )
    assert np.allclose(metrics["I_available_bits"], metrics["I_retained_bits"] + metrics["closure_leakage_bits"])
    assert (metrics["closure_quality"] > 0).all()
    assert metrics["p_fdr_bh"].between(0, 1).all()
    assert (audit["spot_pij_max_abs_diff_vs_reference"] == 0).all()
    assert json.loads(outputs["manifest"].read_text(encoding="utf-8"))["row_count"] == 18
    with pytest.raises(RuntimeError, match="not empty"):
        run_seurat_closure_existence(cfg)


def test_low_signal_is_not_claimed() -> None:
    cfg = SeuratClosureConfig(Path("."), Path("."), Path("unused"), times=("11.5", "12.5"))
    p = np.full((4, 4), 0.25)
    assignment = _one_hot(np.asarray([0, 0, 1, 1]), 2)
    row, budget = _metric_row(cfg, "11.5->12.5", "seurat_k10", p, assignment, assignment, np.full((2, 2), 0.5))
    assert row["signal_status"] == "low-signal"
    assert budget["I_available_bits"] == pytest.approx(0)


def test_common_spot_reference_is_used_when_pair_spot_pij_differs(tmp_path: Path) -> None:
    data_root, archive_root, p_reference = _synthetic_archive(tmp_path)
    k40_spot_path = (
        archive_root / "network=light_cci_grn" / "pij=NG_KLot" / "organ=heart"
        / "pair=spot_to_seurat_k40" / "11.5_to_12.5_lower_P.npz"
    )
    sp.save_npz(k40_spot_path, sp.csr_matrix(np.full_like(p_reference, 1 / len(p_reference))))
    cfg = SeuratClosureConfig(data_root, archive_root, tmp_path / "results", times=("11.5", "12.5"), null_repeats=2)
    outputs = run_seurat_closure_existence(cfg)
    metrics = pd.read_csv(outputs["closure_metrics"]).set_index("macro_layer")
    k40 = metrics.loc["seurat_k40"]
    assert k40["spot_pij_max_abs_diff_vs_reference"] > 0
    assert k40["EI_micro_pair_spot_bits"] == pytest.approx(0, abs=1e-9)
    assert k40["EI_micro_spot_bits"] > 0
    assert k40["closure_quality"] > 0


def test_uniform_spot_dynamics_is_marked_low_signal(tmp_path: Path) -> None:
    data_root, archive_root, p_reference = _synthetic_archive(tmp_path)
    for layer in SEURAT_LAYERS:
        path = (
            archive_root / "network=light_cci_grn" / "pij=NG_KLot" / "organ=heart"
            / f"pair=spot_to_{layer}" / "11.5_to_12.5_lower_P.npz"
        )
        sp.save_npz(path, sp.csr_matrix(np.full_like(p_reference, 1 / len(p_reference))))
    cfg = SeuratClosureConfig(data_root, archive_root, tmp_path / "results", times=("11.5", "12.5"), null_repeats=2)
    metrics = pd.read_csv(run_seurat_closure_existence(cfg)["closure_metrics"])
    assert (metrics["signal_status"] == "low-signal").all()
    assert (metrics["claim_status"] == "low_signal").all()
    assert metrics["closure_quality_for_claim"].isna().all()
    assert not metrics["evidence_vs_size_matched_null"].any()


def test_archive_unit_mismatch_fails_before_writing(tmp_path: Path) -> None:
    data_root, archive_root, _ = _synthetic_archive(tmp_path)
    units_path = (
        archive_root / "network=light_cci_grn" / "pij=NG_KLot" / "organ=heart"
        / "pair=spot_to_seurat_k150" / "units" / "lower_11.5_units.csv"
    )
    units = pd.read_csv(units_path)
    units.loc[0, "unit"] = "wrong_spot"
    units.to_csv(units_path, index=False)
    cfg = SeuratClosureConfig(data_root, archive_root, tmp_path / "results", times=("11.5", "12.5"), null_repeats=2)
    with pytest.raises(ValueError, match="Spot PIJ units"):
        run_seurat_closure_existence(cfg)
    assert not cfg.output_root.exists()
