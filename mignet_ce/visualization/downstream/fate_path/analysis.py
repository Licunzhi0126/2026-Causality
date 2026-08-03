from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import compose_transitions, entropy, row_normalize


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
