from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mignet_ce.visualization.downstream.config import FullDeltaEIBenchmarkProfile
from mignet_ce.visualization.downstream.determinism_degeneracy.analysis import build_unified_ei_tables
from mignet_ce.visualization.downstream.dynamic_closure.analysis import (
    build_cross_representation_consistency,
    build_unified_closure_table,
    effective_information,
)
from mignet_ce.visualization.downstream.fate_path.analysis import build_unified_fate_paths
from mignet_ce.visualization.downstream.mappings import MAPPINGS, MappingRecord, is_optimized
from mignet_ce.visualization.downstream.null_model.analysis import build_unified_matched_null
from mignet_ce.visualization.downstream.perturbation.analysis import build_unified_perturbation_curves
from mignet_ce.visualization.downstream.spatial.analysis import (
    build_unified_effective_states,
    build_unified_spatial_metrics,
    build_unified_spatial_spots,
)
from mignet_ce.visualization.downstream.workflow import (
    UNIFIED_FIGURE_FILES,
    UNIFIED_TABLE_FILES,
    _render_unified,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_unified_downstream_analysis.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("run_unified_downstream_analysis_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_cli_exposes_no_reduced_deltaei_controls() -> None:
    parser = _load_cli_module().build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--optimized-epochs" not in option_strings
    assert "--nmf-max-iter" not in option_strings
    assert "--skip-prepare" not in option_strings
    assert "--cache-root" in option_strings
    assert "--large-target-nmf-max-iter" not in option_strings


def test_unified_output_contract_is_eleven_tables_and_nine_figures() -> None:
    assert len(UNIFIED_TABLE_FILES) == 11
    assert len(UNIFIED_FIGURE_FILES) == 9
    assert set(UNIFIED_FIGURE_FILES) == {
        "ei",
        "spatial_ei",
        "null",
        "consistency",
        "effective",
        "mechanism",
        "fate",
        "perturbation",
        "closure",
    }


def _synthetic_records():
    pairs = ("11.5->12.5", "12.5->13.5", "13.5->14.5")
    spots = tuple(f"s{index}" for index in range(6))
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    assignment = np.zeros((6, 3), dtype=float)
    assignment[np.arange(6), labels] = 1.0
    transition = np.asarray(
        [
            [0.55, 0.20, 0.10, 0.05, 0.05, 0.05],
            [0.50, 0.25, 0.10, 0.05, 0.05, 0.05],
            [0.45, 0.25, 0.15, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.15, 0.25, 0.45],
            [0.05, 0.05, 0.05, 0.10, 0.25, 0.50],
            [0.05, 0.05, 0.05, 0.10, 0.20, 0.55],
        ],
        dtype=float,
    )
    q_matrix = np.asarray(
        [[0.72, 0.18, 0.10], [0.17, 0.73, 0.10], [0.20, 0.20, 0.60]],
        dtype=float,
    )
    coords = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    records = {}
    for mapping in MAPPINGS:
        method = mapping.replace("Optimized ", "")
        for pair in pairs:
            micro_ei = effective_information(transition)
            macro_ei = effective_information(q_matrix)
            summary = (
                {
                    "K": 3,
                    "EI_micro_fixed": micro_ei,
                    "EI_macro_best_checkpoint": macro_ei,
                    "delta_EI_best_checkpoint": macro_ei - micro_ei,
                }
                if is_optimized(mapping)
                else {}
            )
            records[(mapping, pair)] = MappingRecord(
                mapping=mapping,
                pair=pair,
                p=transition,
                hs=assignment,
                ht=assignment,
                q_direct=q_matrix,
                spots_s=spots,
                spots_t=spots,
                soft_s=assignment,
                soft_t=assignment,
                coords_s=coords,
                coords_t=coords,
                summary=summary,
                method=method,
            )
    return pairs, records


def test_topic_split_pipeline_renders_nine_png_and_pdf_figures(tmp_path) -> None:
    pairs, records = _synthetic_records()
    cfg = SimpleNamespace(
        times=("11.5", "12.5", "13.5", "14.5"),
        adjacent_pairs=pairs,
        mapping_names=MAPPINGS,
        spatial_knn=2,
        profile=FullDeltaEIBenchmarkProfile(),
    )
    metrics, states = build_unified_ei_tables(cfg, records)
    spatial_spots = build_unified_spatial_spots(cfg, records)
    closure = build_unified_closure_table(cfg, records)
    spatial = build_unified_spatial_metrics(cfg, records)
    effective = build_unified_effective_states(cfg, records)
    matched_null = build_unified_matched_null(cfg, records)
    consistency = build_cross_representation_consistency(cfg, records)
    fate = build_unified_fate_paths(cfg, records)
    mechanism_rows = []
    for pair in pairs:
        for mapping in MAPPINGS:
            for state in range(3):
                mechanism_rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "state_index": state,
                        "state_ei": 0.2 + 0.1 * state,
                        "grn_concentration": 0.35 + 0.1 * state,
                        "cci_out_log": 0.6 + 0.1 * state,
                        "cci_in_log": 0.5 + 0.1 * state,
                    }
                )
    mechanism = pd.DataFrame(mechanism_rows)
    perturbation = build_unified_perturbation_curves(cfg, records, mechanism)
    assert set(metrics["model_source_states"]) == {3}
    assert set(metrics["hard_active_source_states"]) == {2}
    assert set(closure["model_k_source"]) == {3}
    assert set(closure["active_k_source"]) == {2}
    assert set(states["state_index"]) == {0, 1, 2}
    tables = {
        "metrics": metrics,
        "states": states,
        "spatial_spots": spatial_spots,
        "closure": closure,
        "spatial": spatial,
        "effective": effective,
        "mechanism": mechanism,
        "null": matched_null,
        "consistency": consistency,
        "fate": fate,
        "perturbation": perturbation,
    }
    figures = _render_unified(tables, tmp_path / "figures")
    assert len(figures) == 9
    assert all(path.exists() for path in figures)
    assert all(path.with_suffix(".pdf").exists() for path in figures)
