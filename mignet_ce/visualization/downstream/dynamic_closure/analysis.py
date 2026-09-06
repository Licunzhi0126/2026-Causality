from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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


# ---------------------------------------------------------------------------
# Unified four-representation closure analysis.
# The legacy K150/K40 functions above remain unchanged for compatibility.
# ---------------------------------------------------------------------------

EPS = 1e-12


def normalize_assignment(matrix: np.ndarray) -> np.ndarray:
    values = np.maximum(np.nan_to_num(np.asarray(matrix, dtype=float)), 0.0)
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional assignment, got {values.shape}")
    return row_normalize(values)


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    values = row_normalize(probabilities)
    return -np.sum(np.where(values > EPS, values * np.log2(np.maximum(values, EPS)), 0.0), axis=1)


def kl_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = row_normalize(left)
    q = row_normalize(right)
    if p.shape != q.shape:
        raise ValueError(f"KL inputs must have the same shape, got {p.shape} and {q.shape}")
    return np.sum(np.where(p > EPS, p * np.log2(np.maximum(p, EPS) / np.maximum(q, EPS)), 0.0), axis=1)


def js_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = row_normalize(left)
    q = row_normalize(right)
    if p.shape != q.shape:
        raise ValueError(f"JS inputs must have the same shape, got {p.shape} and {q.shape}")
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_rows(p, midpoint) + 0.5 * kl_rows(q, midpoint)


def state_level_ei(probabilities: np.ndarray) -> np.ndarray:
    values = row_normalize(probabilities)
    effect = values.mean(axis=0, keepdims=True)
    return kl_rows(values, np.repeat(effect, values.shape[0], axis=0)) / values.shape[0]


def effective_information(probabilities: np.ndarray) -> float:
    return float(np.sum(state_level_ei(probabilities)))


def to_hard_assignment(matrix: np.ndarray) -> np.ndarray:
    values = normalize_assignment(matrix)
    labels = np.argmax(values, axis=1)
    hard = np.zeros_like(values)
    hard[np.arange(len(labels)), labels] = 1.0
    return hard


def compact_hard_assignment(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hard = to_hard_assignment(matrix)
    active = np.flatnonzero(hard.sum(axis=0) > 0)
    if len(active) == 0:
        raise ValueError("Assignment contains no active macrostate")
    return hard[:, active], active


def state_balanced_micro_weights(hard_assignment: np.ndarray) -> np.ndarray:
    hard, _ = compact_hard_assignment(hard_assignment)
    labels = np.argmax(hard, axis=1)
    counts = np.bincount(labels, minlength=hard.shape[1]).astype(float)
    weights = 1.0 / (hard.shape[1] * counts[labels])
    return weights / weights.sum()


def induced_macro_q(
    observed_micro_to_macro: np.ndarray,
    source_assignment: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = row_normalize(observed_micro_to_macro)
    source = normalize_assignment(source_assignment)
    if observed.shape[0] != source.shape[0]:
        raise ValueError("Observed rows do not match source assignment rows")
    if weights is None:
        weights = np.full(source.shape[0], 1.0 / source.shape[0])
    values = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if len(values) != source.shape[0]:
        raise ValueError("weights length does not match source assignment rows")
    values /= max(float(values.sum()), EPS)
    weighted_source = source * values[:, None]
    macro_mass = weighted_source.sum(axis=0)
    q_matrix = weighted_source.T @ observed
    q_matrix = np.divide(
        q_matrix,
        macro_mass[:, None],
        out=np.zeros_like(q_matrix),
        where=macro_mass[:, None] > EPS,
    )
    return row_normalize(q_matrix), macro_mass


def weighted_information(probabilities: np.ndarray, weights: np.ndarray) -> float:
    values = row_normalize(probabilities)
    mass = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if len(mass) != values.shape[0]:
        raise ValueError("weights length does not match probability rows")
    mass /= max(float(mass.sum()), EPS)
    effect = mass @ values
    repeated = np.repeat(effect[None, :], values.shape[0], axis=0)
    return float(np.sum(mass * kl_rows(values, repeated)))


def information_closure_budget(
    p_micro: np.ndarray,
    source_assignment: np.ndarray,
    target_assignment: np.ndarray,
    *,
    weighting: str = "state_balanced",
    low_signal_threshold_bits: float | None = None,
) -> dict[str, object]:
    p_matrix = row_normalize(p_micro)
    source_hard, active_source = compact_hard_assignment(source_assignment)
    target_hard, active_target = compact_hard_assignment(target_assignment)
    if p_matrix.shape != (source_hard.shape[0], target_hard.shape[0]):
        raise ValueError(
            f"P {p_matrix.shape} is incompatible with assignments "
            f"{source_hard.shape} and {target_hard.shape}"
        )
    observed = row_normalize(p_matrix @ target_hard)
    if weighting == "state_balanced":
        weights = state_balanced_micro_weights(source_hard)
    elif weighting == "uniform_spot":
        weights = np.full(source_hard.shape[0], 1.0 / source_hard.shape[0])
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    q_induced, macro_mass = induced_macro_q(observed, source_hard, weights)
    predicted = row_normalize(source_hard @ q_induced)
    available = weighted_information(observed, weights)
    retained = weighted_information(q_induced, macro_mass)
    leakage = float(np.sum(weights * kl_rows(observed, predicted)))
    identity_error = float(abs(available - retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(
            "Information closure identity failed: "
            f"available={available}, retained={retained}, leakage={leakage}, error={identity_error}"
        )
    if low_signal_threshold_bits is None:
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(target_hard.shape[1], 2)))
    status = "informative" if available >= low_signal_threshold_bits else "low-signal"
    quality = retained / available if available > EPS else np.nan
    return {
        "observed": observed,
        "source_hard": source_hard,
        "target_hard": target_hard,
        "source_active_columns": active_source,
        "target_active_columns": active_target,
        "weights": weights,
        "q_induced": q_induced,
        "macro_mass": macro_mass,
        "predicted_induced": predicted,
        "I_available_bits": float(available),
        "I_macro_retained_bits": float(retained),
        "closure_leakage_bits": leakage,
        "closure_quality": float(quality) if np.isfinite(quality) else np.nan,
        "information_identity_error": identity_error,
        "signal_status": status,
        "signal_threshold_bits": float(low_signal_threshold_bits),
        "weighting": weighting,
    }


def information_closure_budget_from_observed(
    observed_micro_to_macro: np.ndarray,
    source_assignment: np.ndarray,
    *,
    weighting: str = "state_balanced",
    low_signal_threshold_bits: float | None = None,
) -> dict[str, object]:
    observed = row_normalize(observed_micro_to_macro)
    source_hard, active_source = compact_hard_assignment(source_assignment)
    if observed.shape[0] != source_hard.shape[0]:
        raise ValueError("Observed rows do not match source assignment rows")
    if weighting == "state_balanced":
        weights = state_balanced_micro_weights(source_hard)
    elif weighting == "uniform_spot":
        weights = np.full(source_hard.shape[0], 1.0 / source_hard.shape[0])
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    q_induced, macro_mass = induced_macro_q(observed, source_hard, weights)
    predicted = row_normalize(source_hard @ q_induced)
    available = weighted_information(observed, weights)
    retained = weighted_information(q_induced, macro_mass)
    leakage = float(np.sum(weights * kl_rows(observed, predicted)))
    identity_error = float(abs(available - retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(f"Information closure identity error {identity_error}")
    if low_signal_threshold_bits is None:
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(observed.shape[1], 2)))
    status = "informative" if available >= low_signal_threshold_bits else "low-signal"
    quality = retained / available if available > EPS else np.nan
    return {
        "observed": observed,
        "source_hard": source_hard,
        "source_active_columns": active_source,
        "weights": weights,
        "q_induced": q_induced,
        "macro_mass": macro_mass,
        "predicted_induced": predicted,
        "I_available_bits": float(available),
        "I_macro_retained_bits": float(retained),
        "closure_leakage_bits": leakage,
        "closure_quality": float(quality) if np.isfinite(quality) else np.nan,
        "information_identity_error": identity_error,
        "signal_status": status,
        "signal_threshold_bits": float(low_signal_threshold_bits),
        "weighting": weighting,
    }


def crossfit_macro_q_error(
    observed: np.ndarray,
    source_hard: np.ndarray,
    *,
    folds: int = 5,
    weights: np.ndarray | None = None,
    seed: int = 20260809,
) -> dict[str, float]:
    obs = row_normalize(observed)
    hard, _ = compact_hard_assignment(source_hard)
    labels = np.argmax(hard, axis=1)
    if weights is None:
        weights = state_balanced_micro_weights(hard)
    mass = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    mass /= max(float(mass.sum()), EPS)
    rng = np.random.default_rng(seed)
    predicted = np.zeros_like(obs)
    estimable = np.zeros(len(labels), dtype=bool)
    singleton = np.zeros(len(labels), dtype=bool)
    for state in np.unique(labels):
        indices = np.flatnonzero(labels == state)
        if len(indices) == 1:
            singleton[indices] = True
            continue
        estimable[indices] = True
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        fold_count = min(max(2, int(folds)), len(indices))
        fold_ids = np.arange(len(shuffled)) % fold_count
        for fold in range(fold_count):
            test = shuffled[fold_ids == fold]
            train = shuffled[fold_ids != fold]
            train_mass = mass[train]
            train_mass /= max(float(train_mass.sum()), EPS)
            predicted[test] = train_mass @ obs[train]
    coverage = float(np.sum(mass[estimable]))
    singleton_fraction = float(np.sum(mass[singleton]))
    if coverage <= EPS:
        return {
            "crossfit_kl_bits": np.nan,
            "crossfit_js": np.nan,
            "crossfit_coverage": 0.0,
            "singleton_weight_fraction": singleton_fraction,
        }
    normalized_mass = mass[estimable] / coverage
    return {
        "crossfit_kl_bits": float(np.sum(normalized_mass * kl_rows(obs[estimable], predicted[estimable]))),
        "crossfit_js": float(np.sum(normalized_mass * js_rows(obs[estimable], predicted[estimable]))),
        "crossfit_coverage": coverage,
        "singleton_weight_fraction": singleton_fraction,
    }


def build_unified_closure_table(cfg, records_by_pair: dict[tuple[str, str], object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            budget = information_closure_budget(
                record.p_model_micro,
                record.hard_s_full,
                record.hard_t_full,
            )
            direct_full = row_normalize(record.q_model_full)
            active_source = np.asarray(budget["source_active_columns"], dtype=int)
            active_target = np.asarray(budget["target_active_columns"], dtype=int)
            if (
                len(active_source) == 0
                or len(active_target) == 0
                or active_source.max() >= direct_full.shape[0]
                or active_target.max() >= direct_full.shape[1]
            ):
                raise ValueError(
                    f"Full direct Q {direct_full.shape} cannot cover closure-active states "
                    f"for {mapping} {pair}"
                )
            direct_compact = row_normalize(
                direct_full[np.ix_(active_source, active_target)]
            )
            if direct_compact.shape != budget["q_induced"].shape:
                raise ValueError(
                    f"Closure-compact direct Q {direct_compact.shape} does not match induced Q "
                    f"{budget['q_induced'].shape} for {mapping} {pair}"
                )
            rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "source_time": source,
                    "target_time": target,
                    "I_available_bits": budget["I_available_bits"],
                    "I_retained_bits": budget["I_macro_retained_bits"],
                    "closure_leakage_bits": budget["closure_leakage_bits"],
                    "closure_quality": budget["closure_quality"],
                    "signal_status": budget["signal_status"],
                    "information_identity_error": budget["information_identity_error"],
                    "direct_induced_q_js": float(
                        np.mean(js_rows(direct_compact, budget["q_induced"]))
                    ),
                    "active_k_source": budget["source_hard"].shape[1],
                    "active_k_target": budget["target_hard"].shape[1],
                    "model_k_source": direct_full.shape[0],
                    "model_k_target": direct_full.shape[1],
                    "analysis_space": "closure_hard_active",
                }
            )
    return pd.DataFrame(rows)


def build_cross_representation_consistency(cfg, records_by_pair: dict[tuple[str, str], object]) -> pd.DataFrame:
    def labels_at_time(mapping: str, time: str) -> dict[str, int]:
        index = cfg.times.index(time)
        if index < len(cfg.times) - 1:
            record = records_by_pair[(mapping, cfg.adjacent_pairs[index])]
            assignment, spots = record.hard_s_full, record.spots_s
        else:
            record = records_by_pair[(mapping, cfg.adjacent_pairs[-1])]
            assignment, spots = record.hard_t_full, record.spots_t
        labels = np.argmax(assignment, axis=1)
        return dict(zip(map(str, spots), map(int, labels)))

    rows: list[dict[str, object]] = []
    for time in cfg.times:
        label_maps = {mapping: labels_at_time(mapping, time) for mapping in cfg.mapping_names}
        for index, left in enumerate(cfg.mapping_names):
            for right in cfg.mapping_names[index + 1 :]:
                common = sorted(set(label_maps[left]) & set(label_maps[right]))
                if not common:
                    raise ValueError(f"No common spots for {left} and {right} at {time}")
                left_values = [label_maps[left][spot] for spot in common]
                right_values = [label_maps[right][spot] for spot in common]
                rows.append(
                    {
                        "time": time,
                        "mapping_a": left,
                        "mapping_b": right,
                        "common_spots": len(common),
                        "NMI": normalized_mutual_info_score(left_values, right_values),
                        "ARI": adjusted_rand_score(left_values, right_values),
                        "analysis_space": "hard_partition",
                    }
                )
    return pd.DataFrame(rows)
