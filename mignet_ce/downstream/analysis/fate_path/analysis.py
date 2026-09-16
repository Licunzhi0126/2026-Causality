from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import entropy, row_normalize
from ..dynamic_closure.analysis import entropy_rows, state_level_ei


def _soft_alignment(
    left_spots,
    left_assignment: np.ndarray,
    right_spots,
    right_assignment: np.ndarray,
) -> np.ndarray:
    left_lookup = {unit: index for index, unit in enumerate(map(str, left_spots))}
    right_lookup = {unit: index for index, unit in enumerate(map(str, right_spots))}
    common = sorted(set(left_lookup) & set(right_lookup))
    if not common:
        raise ValueError("Consecutive optimized mappings have no common intermediate spots")
    left = left_assignment[[left_lookup[unit] for unit in common]]
    right = right_assignment[[right_lookup[unit] for unit in common]]
    return row_normalize(left.T @ right)


def build_unified_fate_paths(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mapping in cfg.mapping_names:
        records = [records_by_pair[(mapping, pair)] for pair in cfg.adjacent_pairs]
        transitions = [row_normalize(record.q_direct) for record in records]
        alignments: list[np.ndarray] = []
        for index in range(len(records) - 1):
            alignment = _soft_alignment(
                records[index].spots_t,
                records[index].target_assignment,
                records[index + 1].spots_s,
                records[index + 1].source_assignment,
            )
            alignments.append(alignment)
        q01, q12, q23 = transitions
        align1, align2 = alignments
        composed = row_normalize(q01 @ align1 @ q12 @ align2 @ q23)
        source_ei = state_level_ei(q01)
        first_entropy = entropy_rows(q01)
        for source_state in range(q01.shape[0]):
            state1 = int(np.argmax(q01[source_state]))
            state1_aligned = int(np.argmax(align1[state1]))
            state2 = int(np.argmax(q12[state1_aligned]))
            state2_aligned = int(np.argmax(align2[state2]))
            state3 = int(np.argmax(q23[state2_aligned]))
            probability = (
                q01[source_state, state1]
                * align1[state1, state1_aligned]
                * q12[state1_aligned, state2]
                * align2[state2, state2_aligned]
                * q23[state2_aligned, state3]
            )
            rows.append(
                {
                    "mapping": mapping,
                    "state_t0": source_state,
                    "state_t1": state1,
                    "state_t2": state2,
                    "state_t3": state3,
                    "path_probability": float(probability),
                    "endpoint_entropy": float(entropy_rows(composed[source_state : source_state + 1])[0]),
                    "source_ei": float(source_ei[source_state]),
                    "first_branch_entropy": float(first_entropy[source_state]),
                    "path_method": "greedy_aligned_macro_chain",
                }
            )
    return pd.DataFrame(rows)
