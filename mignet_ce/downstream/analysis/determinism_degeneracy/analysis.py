from __future__ import annotations

import numpy as np
import pandas as pd

from ..dynamic_closure.analysis import effective_information, entropy_rows, state_level_ei
from ..deltaei_contract import deltaei_contract_row
from ..mappings import is_optimized
from ..metrics import row_normalize


def build_unified_ei_tables(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build EI/decomposition tables for all configured macro representations."""

    metric_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            q_matrix = row_normalize(record.q_direct)
            effect = q_matrix.mean(axis=0)
            positive = effect[effect > 1e-12]
            h_effect = float(-np.sum(positive * np.log2(positive)))
            h_noise = float(np.mean(entropy_rows(q_matrix)))
            determinism = float(np.log2(max(q_matrix.shape[1], 1)) - h_noise)
            degeneracy = float(np.log2(max(q_matrix.shape[1], 1)) - h_effect)
            macro_ei = effective_information(q_matrix)
            micro_ei = effective_information(record.p)
            delta_ei = macro_ei - micro_ei
            source_usage = record.source_assignment.mean(axis=0)
            target_usage = record.target_assignment.mean(axis=0)
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
                    "EI": macro_ei,
                    "micro_EI": micro_ei,
                    "delta_EI": delta_ei,
                    "EI_macro_model_full": macro_ei,
                    "EI_micro_matched": micro_ei,
                    "delta_EI_matched_spot": delta_ei,
                    "delta_EI_summary": summary_delta,
                    "delta_EI_consistency_error": consistency_error,
                    "H_effect": h_effect,
                    "H_noise": h_noise,
                    "determinism": determinism,
                    "degeneracy": degeneracy,
                    "source_states": q_matrix.shape[0],
                    "target_states": q_matrix.shape[1],
                    "model_source_states": q_matrix.shape[0],
                    "model_target_states": q_matrix.shape[1],
                    "soft_active_source_states": int(np.count_nonzero(source_usage > 1e-8)),
                    "soft_active_target_states": int(np.count_nonzero(target_usage > 1e-8)),
                    "analysis_space": "full_soft_model",
                    "Keff_source": float(np.exp(-np.sum(record.source_assignment.mean(axis=0) * np.log(np.maximum(record.source_assignment.mean(axis=0), 1e-12))))),
                    "Keff_target": float(np.exp(-np.sum(record.target_assignment.mean(axis=0) * np.log(np.maximum(record.target_assignment.mean(axis=0), 1e-12))))),
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
