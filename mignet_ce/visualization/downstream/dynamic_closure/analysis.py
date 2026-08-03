from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_domain_map, load_pij, load_units
from ..metrics import (
    aggregate_transition_by_overlap,
    compose_transitions,
    effective_state_number,
    ei_decomposition,
    entropy,
    hierarchy_counts,
    mean_row_js,
    purity_entropy_from_counts,
    relative_frobenius,
    row_normalize,
)


def build_multistep_closure(cfg: DownstreamConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_index in range(len(cfg.times)):
        for target_index in range(source_index + 2, len(cfg.times)):
            source = cfg.times[source_index]
            target = cfg.times[target_index]
            steps = [
                f"{cfg.times[index]}->{cfg.times[index + 1]}"
                for index in range(source_index, target_index)
            ]
            direct_pair = f"{source}->{target}"
            for space, layer in (("lower", "K150"), ("upper", "K40")):
                composed = compose_transitions([load_pij(cfg.pair_archive, pair, space) for pair in steps])
                direct = row_normalize(load_pij(cfg.pair_archive, direct_pair, space))
                composed_ei = ei_decomposition(composed)["EI"]
                direct_ei = ei_decomposition(direct)["EI"]
                rows.append(
                    {
                        "comparison": f"{source}->{target}",
                        "direct_pair": direct_pair,
                        "space": space,
                        "layer": layer,
                        "relative_frobenius": relative_frobenius(composed, direct),
                        "mean_row_js": mean_row_js(composed, direct),
                        "EI_composed": composed_ei,
                        "EI_direct": direct_ei,
                        "EI_difference": composed_ei - direct_ei,
                    }
                )
    return pd.DataFrame(rows)


def build_single_step_closure(
    cfg: DownstreamConfig,
    counts_by_time: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        p_lower = row_normalize(load_pij(cfg.pair_archive, pair, "lower"))
        q_direct = row_normalize(load_pij(cfg.pair_archive, pair, "upper"))
        counts_source = counts_by_time[source]
        counts_target = counts_by_time[target]
        source_membership = counts_source / np.maximum(counts_source.sum(axis=1, keepdims=True), 1e-12)
        target_membership = counts_target / np.maximum(counts_target.sum(axis=1, keepdims=True), 1e-12)
        observed = p_lower @ target_membership
        q_aggregated = aggregate_transition_by_overlap(p_lower, counts_source, counts_target)
        q_best = np.linalg.lstsq(source_membership, observed, rcond=None)[0]
        q_best = row_normalize(np.maximum(q_best, 0.0))
        for method, q_matrix in (
            ("direct_K40_Pij", q_direct),
            ("overlap_aggregated", q_aggregated),
            ("clipped_least_squares", q_best),
        ):
            predicted = source_membership @ q_matrix
            rows.append(
                {
                    "time_pair": pair,
                    "macro_Q": method,
                    "relative_frobenius": relative_frobenius(predicted, observed),
                    "mean_row_js": mean_row_js(predicted, observed),
                    "EI_Q": ei_decomposition(q_matrix)["EI"],
                }
            )
    return pd.DataFrame(rows)


def build_hierarchy_tables(
    cfg: DownstreamConfig,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    counts_by_time: dict[str, np.ndarray] = {}
    purity_rows: list[pd.DataFrame] = []
    effective_rows: list[dict[str, object]] = []
    for time in cfg.times:
        lower_units = load_units(cfg.pair_archive, "lower", time)
        upper_units = load_units(cfg.pair_archive, "upper", time)
        lower_map = load_domain_map(cfg.data_root, cfg.lower_layer, time, cfg.organ)
        upper_map = load_domain_map(cfg.data_root, cfg.upper_layer, time, cfg.organ)
        counts, _ = hierarchy_counts(lower_map, upper_map, lower_units, upper_units)
        counts_by_time[time] = counts
        purity = purity_entropy_from_counts(counts)
        purity.insert(0, "k150_state", lower_units)
        purity.insert(0, "time", time)
        purity_rows.append(purity)
        for layer, mapping, units in (
            (cfg.lower_layer, lower_map, lower_units),
            (cfg.upper_layer, upper_map, upper_units),
        ):
            usage = mapping["domain_id"].value_counts().reindex(units, fill_value=0).to_numpy(dtype=float)
            total = max(float(usage.sum()), 1.0)
            effective_rows.append(
                {
                    "time": time,
                    "layer": layer,
                    "hardK": int(np.count_nonzero(usage)),
                    "Keff": effective_state_number(usage),
                    "max_usage": float(usage.max() / total),
                    "min_nonzero_usage": float(usage[usage > 0].min() / total) if np.any(usage > 0) else 0.0,
                    "usage_entropy_bits": float(entropy(usage / total)),
                    "spot_count": int(usage.sum()),
                }
            )
    return counts_by_time, pd.concat(purity_rows, ignore_index=True), pd.DataFrame(effective_rows)


def build_multiscale_consistency(
    cfg: DownstreamConfig,
    counts_by_time: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in cfg.all_pairs:
        source, target = pair.split("->")
        lower = load_pij(cfg.pair_archive, pair, "lower")
        via_lower = aggregate_transition_by_overlap(lower, counts_by_time[source], counts_by_time[target])
        direct = row_normalize(load_pij(cfg.pair_archive, pair, "upper"))
        via_ei = ei_decomposition(via_lower)["EI"]
        direct_ei = ei_decomposition(direct)["EI"]
        rows.append(
            {
                "time_pair": pair,
                "relative_frobenius": relative_frobenius(via_lower, direct),
                "mean_row_js": mean_row_js(via_lower, direct),
                "EI_via_K150": via_ei,
                "EI_direct_K40": direct_ei,
                "EI_path_difference": via_ei - direct_ei,
            }
        )
    return pd.DataFrame(rows)
