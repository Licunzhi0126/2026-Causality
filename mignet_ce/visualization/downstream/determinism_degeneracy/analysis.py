from __future__ import annotations

import pandas as pd

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import ei_decomposition, entropy, row_normalize, state_ei


def build_ei_tables(
    cfg: DownstreamConfig,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decompositions: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    for time_pair in metrics["time_pair"].astype(str):
        source_time, target_time = time_pair.split("->")
        for space, layer in (("lower", cfg.lower_layer), ("upper", cfg.upper_layer)):
            transition = load_pij(cfg.pair_archive, time_pair, space)
            values: dict[str, object] = ei_decomposition(transition)
            values.update(
                {
                    "time_pair": time_pair,
                    "source_time": source_time,
                    "target_time": target_time,
                    "space": space,
                    "layer": layer,
                }
            )
            decompositions.append(values)
            units = load_units(cfg.pair_archive, space, source_time)
            normalized = row_normalize(transition)
            row_entropy = entropy(normalized, axis=1)
            contribution = state_ei(normalized)
            states.extend(
                {
                    "time_pair": time_pair,
                    "source_time": source_time,
                    "target_time": target_time,
                    "space": space,
                    "layer": layer,
                    "state": unit,
                    "state_ei": float(ei_value),
                    "transition_entropy": float(entropy_value),
                }
                for unit, ei_value, entropy_value in zip(units, contribution, row_entropy)
            )
    return pd.DataFrame(decompositions), pd.DataFrame(states)
