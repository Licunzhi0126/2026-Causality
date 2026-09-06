from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import compose_transitions, entropy, row_normalize
from ..dynamic_closure.analysis import entropy_rows, state_level_ei
from ..mappings import is_optimized


def _viterbi_path(transitions: list[np.ndarray], source_index: int) -> tuple[list[int], float]:
    first_size = transitions[0].shape[0]
    scores = np.full(first_size, -np.inf, dtype=float)
    scores[source_index] = 0.0
    backpointers: list[np.ndarray] = []
    for transition in transitions:
        probabilities = row_normalize(transition)
        candidate = scores[:, None] + np.log(np.maximum(probabilities, 1e-300))
        backpointer = np.argmax(candidate, axis=0)
        scores = candidate[backpointer, np.arange(candidate.shape[1])]
        backpointers.append(backpointer)
    endpoint = int(np.argmax(scores))
    indices = [endpoint]
    current = endpoint
    for backpointer in reversed(backpointers):
        current = int(backpointer[current])
        indices.append(current)
    indices.reverse()
    return indices, float(np.exp(scores[endpoint]))


def build_fate_paths(cfg: DownstreamConfig, state_table: pd.DataFrame) -> pd.DataFrame:
    transitions = [row_normalize(load_pij(cfg.pair_archive, pair, "upper")) for pair in cfg.adjacent_pairs]
    composed = compose_transitions(transitions)
    units = {time: load_units(cfg.pair_archive, "upper", time) for time in cfg.times}
    first_pair = cfg.adjacent_pairs[0]
    source_ei = state_table[
        (state_table["time_pair"] == first_pair) & (state_table["layer"] == cfg.upper_layer)
    ].set_index("state")["state_ei"]
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(units[cfg.times[0]]):
        indices, probability = _viterbi_path(transitions, source_index)
        row: dict[str, object] = {
            f"state_{time}": units[time][index]
            for time, index in zip(cfg.times, indices)
        }
        row.update(
            {
                "path_probability": probability,
                "endpoint_entropy": float(entropy(composed[source_index])),
                "source_ei": float(source_ei.get(source, np.nan)),
                "first_branch_entropy": float(entropy(transitions[0][source_index])),
                "path_method": "global_viterbi_max_product",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _hard_alignment(
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
    counts = np.zeros((left_assignment.shape[1], right_assignment.shape[1]), dtype=float)
    for unit in common:
        left_state = int(np.argmax(left_assignment[left_lookup[unit]]))
        right_state = int(np.argmax(right_assignment[right_lookup[unit]]))
        counts[left_state, right_state] += 1.0
    return row_normalize(counts)


def _soft_alignment(
    left_spots,
    left_assignment: np.ndarray,
    right_spots,
    right_assignment: np.ndarray,
) -> np.ndarray:
    """Align complete model prototypes through common intermediate spots."""

    left_lookup = {unit: index for index, unit in enumerate(map(str, left_spots))}
    right_lookup = {unit: index for index, unit in enumerate(map(str, right_spots))}
    common = sorted(set(left_lookup) & set(right_lookup))
    if not common:
        raise ValueError("Consecutive optimized mappings have no common intermediate spots")
    left = np.asarray(
        [left_assignment[left_lookup[unit]] for unit in common], dtype=float
    )
    right = np.asarray(
        [right_assignment[right_lookup[unit]] for unit in common], dtype=float
    )
    return row_normalize(left.T @ right)


def build_unified_fate_paths(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mapping in cfg.mapping_names:
        records = [records_by_pair[(mapping, pair)] for pair in cfg.adjacent_pairs]
        transitions = [row_normalize(record.q_model_full) for record in records]
        alignments: list[np.ndarray] = []
        for index in range(len(records) - 1):
            if is_optimized(mapping):
                alignment = _soft_alignment(
                    records[index].spots_t,
                    records[index].soft_t_full,
                    records[index + 1].spots_s,
                    records[index + 1].soft_s_full,
                )
            else:
                if transitions[index].shape[1] != transitions[index + 1].shape[0]:
                    raise ValueError(
                        f"Natural state dimensions do not chain for {mapping}: "
                        f"{transitions[index].shape} -> {transitions[index + 1].shape}"
                    )
                alignment = np.eye(transitions[index].shape[1])
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
                    "path_method": (
                        "greedy_full_soft_aligned_macro_chain"
                        if is_optimized(mapping)
                        else "greedy_full_macro_chain"
                    ),
                    "analysis_space": "full_model",
                }
            )
    return pd.DataFrame(rows)
