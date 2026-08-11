from __future__ import annotations

import pandas as pd
import numpy as np

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import ei_decomposition, entropy, row_normalize, state_ei
from ..dynamic_closure.analysis import effective_information, entropy_rows, state_level_ei


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


def build_unified_ei_tables(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build EI/decomposition tables for all four formal macro representations."""

    metric_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        spot_p = records_by_pair[(cfg.mapping_names[0], pair)].p
        spot_ei = effective_information(spot_p)
        for mapping in cfg.mapping_names:
            q_matrix = row_normalize(records_by_pair[(mapping, pair)].q_direct)
            effect = q_matrix.mean(axis=0)
            positive = effect[effect > 1e-12]
            h_effect = float(-np.sum(positive * np.log2(positive)))
            h_noise = float(np.mean(entropy_rows(q_matrix)))
            determinism = float(np.log2(max(q_matrix.shape[1], 1)) - h_noise)
            degeneracy = float(np.log2(max(q_matrix.shape[1], 1)) - h_effect)
            macro_ei = effective_information(q_matrix)
            metric_rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "source_time": source,
                    "target_time": target,
                    "EI": macro_ei,
                    "spot_EI": spot_ei,
                    "delta_EI_vs_spot": macro_ei - spot_ei,
                    "H_effect": h_effect,
                    "H_noise": h_noise,
                    "determinism": determinism,
                    "degeneracy": degeneracy,
                    "source_states": q_matrix.shape[0],
                    "target_states": q_matrix.shape[1],
                }
            )
            contributions = state_level_ei(q_matrix)
            transition_entropy = entropy_rows(q_matrix)
            for state_index, (ei_value, entropy_value) in enumerate(
                zip(contributions, transition_entropy)
            ):
                state_rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "source_time": source,
                        "target_time": target,
                        "state_index": state_index,
                        "state": f"M{state_index + 1:03d}",
                        "state_ei": float(ei_value),
                        "transition_entropy": float(entropy_value),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(state_rows)
