from __future__ import annotations

import pandas as pd
import numpy as np

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import ei_decomposition, entropy, row_normalize, state_ei
from ..dynamic_closure.analysis import effective_information, entropy_rows, state_level_ei
from ..deltaei_contract import deltaei_contract_row
from ..mappings import is_optimized


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
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            q_matrix = row_normalize(record.q_model_full)
            micro_ei = effective_information(record.p_model_micro)
            effect = q_matrix.mean(axis=0)
            positive = effect[effect > 1e-12]
            h_effect = float(-np.sum(positive * np.log2(positive)))
            h_noise = float(np.mean(entropy_rows(q_matrix)))
            determinism = float(np.log2(max(q_matrix.shape[1], 1)) - h_noise)
            degeneracy = float(np.log2(max(q_matrix.shape[1], 1)) - h_effect)
            macro_ei = effective_information(q_matrix)
            delta_ei = macro_ei - micro_ei
            active_source = int(np.count_nonzero(np.asarray(record.hard_s_full).sum(axis=0)))
            active_target = int(np.count_nonzero(np.asarray(record.hard_t_full).sum(axis=0)))
            if is_optimized(mapping):
                contract = deltaei_contract_row(record)
                if not contract["passed"]:
                    raise RuntimeError(f"Full-model DeltaEI contract failed: {contract}")
                summary_delta = float(contract["summary_delta_EI"])
                consistency_error = float(contract["delta_EI_abs_error"])
            else:
                summary_delta = np.nan
                consistency_error = np.nan
            metric_rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "source_time": source,
                    "target_time": target,
                    "EI_macro_model_full": macro_ei,
                    "EI_micro_matched": micro_ei,
                    "delta_EI_matched_spot": delta_ei,
                    "delta_EI_summary": summary_delta,
                    "delta_EI_consistency_error": consistency_error,
                    "model_source_states": q_matrix.shape[0],
                    "model_target_states": q_matrix.shape[1],
                    "hard_active_source_states": active_source,
                    "hard_active_target_states": active_target,
                    # Compatibility aliases for historical table readers.
                    "EI": macro_ei,
                    "spot_EI": micro_ei,
                    "delta_EI_vs_spot": delta_ei,
                    "H_effect": h_effect,
                    "H_noise": h_noise,
                    "determinism": determinism,
                    "degeneracy": degeneracy,
                    "source_states": q_matrix.shape[0],
                    "target_states": q_matrix.shape[1],
                    "analysis_space": "full_model",
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
                        "analysis_space": "full_model",
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(state_rows)
