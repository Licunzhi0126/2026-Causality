from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mignet_ce.visualization.downstream.config import MAPPING_COMPLETE
from mignet_ce.visualization.downstream.deltaei_contract import deltaei_contract_row
from mignet_ce.visualization.downstream.determinism_degeneracy.analysis import build_unified_ei_tables
from mignet_ce.visualization.downstream.dynamic_closure.analysis import (
    build_unified_closure_table,
    effective_information,
    to_hard_assignment,
)
from mignet_ce.visualization.downstream.mappings import MappingRecord


def _record() -> MappingRecord:
    p_micro = np.asarray(
        [
            [0.65, 0.20, 0.10, 0.05],
            [0.55, 0.25, 0.10, 0.10],
            [0.10, 0.10, 0.25, 0.55],
            [0.05, 0.10, 0.20, 0.65],
        ]
    )
    soft = np.asarray(
        [
            [0.90, 0.09, 0.01],
            [0.80, 0.18, 0.02],
            [0.10, 0.88, 0.02],
            [0.08, 0.90, 0.02],
        ]
    )
    q_full = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.65, 0.15],
            [0.15, 0.15, 0.70],
        ]
    )
    micro_ei = effective_information(p_micro)
    macro_ei = effective_information(q_full)
    return MappingRecord(
        mapping=MAPPING_COMPLETE,
        pair="1->2",
        p=p_micro,
        hs=to_hard_assignment(soft),
        ht=to_hard_assignment(soft),
        q_direct=q_full,
        spots_s=("s0", "s1", "s2", "s3"),
        spots_t=("s0", "s1", "s2", "s3"),
        soft_s=soft,
        soft_t=soft,
        coords_s=np.zeros((4, 2)),
        coords_t=np.zeros((4, 2)),
        summary={
            "K": 3,
            "EI_micro_fixed": micro_ei,
            "EI_macro_best_checkpoint": macro_ei,
            "delta_EI_best_checkpoint": macro_ei - micro_ei,
        },
        method="complete_combined_coarse",
    )


def test_full_model_q_survives_when_hard_projection_has_inactive_prototype() -> None:
    record = _record()
    assert record.q_model_full.shape == (3, 3)
    assert record.soft_s_full.shape == (4, 3)
    assert np.count_nonzero(record.hard_s_full.sum(axis=0)) == 2
    assert deltaei_contract_row(record)["passed"]


def test_metrics_use_full_q_but_closure_compacts_locally() -> None:
    record = _record()
    cfg = SimpleNamespace(
        adjacent_pairs=("1->2",),
        mapping_names=(MAPPING_COMPLETE,),
    )
    records = {(MAPPING_COMPLETE, "1->2"): record}
    metrics, states = build_unified_ei_tables(cfg, records)
    closure = build_unified_closure_table(cfg, records)
    assert metrics.loc[0, "model_source_states"] == 3
    assert metrics.loc[0, "hard_active_source_states"] == 2
    assert len(states) == 3
    assert closure.loc[0, "model_k_source"] == 3
    assert closure.loc[0, "active_k_source"] == 2
    assert closure.loc[0, "analysis_space"] == "closure_hard_active"
