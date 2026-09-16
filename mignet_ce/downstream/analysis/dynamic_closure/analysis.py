from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import entropy, row_normalize

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
    weighting: str = "uniform_spot",
    low_signal_threshold_bits: float | None = None,
) -> dict[str, object]:
    p_matrix = row_normalize(p_micro)
    source = normalize_assignment(source_assignment)
    target = normalize_assignment(target_assignment)
    if p_matrix.shape != (source.shape[0], target.shape[0]):
        raise ValueError(
            f"P {p_matrix.shape} is incompatible with assignments "
            f"{source.shape} and {target.shape}"
        )
    observed = row_normalize(p_matrix @ target)
    if weighting != "uniform_spot":
        raise ValueError(f"Unknown weighting {weighting!r}")
    weights = np.full(source.shape[0], 1.0 / source.shape[0])
    q_induced, macro_mass = induced_macro_q(observed, source, weights)
    predicted = row_normalize(source @ q_induced)
    available = weighted_information(observed, weights)
    retained = weighted_information(q_induced, macro_mass)
    # The membership-weighted conditional KL is the exact soft-assignment leakage.
    pair_kl = np.sum(
        np.where(
            observed[:, None, :] > EPS,
            observed[:, None, :] * np.log2(np.maximum(observed[:, None, :], EPS) / np.maximum(q_induced[None, :, :], EPS)),
            0.0,
        ),
        axis=2,
    )
    leakage = float(np.sum(weights[:, None] * source * pair_kl))
    identity_error = float(abs(available - retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(
            "Information closure identity failed: "
            f"available={available}, retained={retained}, leakage={leakage}, error={identity_error}"
        )
    if low_signal_threshold_bits is None:
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(target.shape[1], 2)))
    status = "informative" if available >= low_signal_threshold_bits else "low-signal"
    quality = retained / available if available > EPS else np.nan
    return {
        "observed": observed,
        "source_assignment": source,
        "target_assignment": target,
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
    weighting: str = "uniform_spot",
    low_signal_threshold_bits: float | None = None,
) -> dict[str, object]:
    observed = row_normalize(observed_micro_to_macro)
    source = normalize_assignment(source_assignment)
    if observed.shape[0] != source.shape[0]:
        raise ValueError("Observed rows do not match source assignment rows")
    if weighting != "uniform_spot":
        raise ValueError(f"Unknown weighting {weighting!r}")
    weights = np.full(source.shape[0], 1.0 / source.shape[0])
    q_induced, macro_mass = induced_macro_q(observed, source, weights)
    predicted = row_normalize(source @ q_induced)
    available = weighted_information(observed, weights)
    retained = weighted_information(q_induced, macro_mass)
    pair_kl = np.sum(
        np.where(
            observed[:, None, :] > EPS,
            observed[:, None, :] * np.log2(np.maximum(observed[:, None, :], EPS) / np.maximum(q_induced[None, :, :], EPS)),
            0.0,
        ),
        axis=2,
    )
    leakage = float(np.sum(weights[:, None] * source * pair_kl))
    identity_error = float(abs(available - retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(f"Information closure identity error {identity_error}")
    if low_signal_threshold_bits is None:
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(observed.shape[1], 2)))
    status = "informative" if available >= low_signal_threshold_bits else "low-signal"
    quality = retained / available if available > EPS else np.nan
    return {
        "observed": observed,
        "source_assignment": source,
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
    source_assignment: np.ndarray,
    *,
    folds: int = 5,
    weights: np.ndarray | None = None,
    seed: int = 20260809,
) -> dict[str, float]:
    obs = row_normalize(observed)
    source = normalize_assignment(source_assignment)
    if weights is None:
        weights = np.full(source.shape[0], 1.0 / source.shape[0])
    mass = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    mass /= max(float(mass.sum()), EPS)
    rng = np.random.default_rng(seed)
    predicted = np.zeros_like(obs)
    shuffled = rng.permutation(obs.shape[0])
    fold_ids = np.arange(obs.shape[0]) % min(max(2, int(folds)), obs.shape[0])
    fold_ids = fold_ids[np.argsort(shuffled)]
    estimable = np.zeros(obs.shape[0], dtype=bool)
    for fold in np.unique(fold_ids):
        test = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        q_train, _ = induced_macro_q(obs[train], source[train], mass[train])
        predicted[test] = row_normalize(source[test] @ q_train)
        estimable[test] = True
    coverage = float(np.sum(mass[estimable]))
    if coverage <= EPS:
        return {
            "crossfit_kl_bits": np.nan,
            "crossfit_js": np.nan,
            "crossfit_coverage": 0.0,
        }
    normalized_mass = mass[estimable] / coverage
    return {
        "crossfit_kl_bits": float(np.sum(normalized_mass * kl_rows(obs[estimable], predicted[estimable]))),
        "crossfit_js": float(np.sum(normalized_mass * js_rows(obs[estimable], predicted[estimable]))),
        "crossfit_coverage": coverage,
    }


def build_unified_closure_table(cfg, records_by_pair: dict[tuple[str, str], object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            budget = information_closure_budget(record.p, record.source_assignment, record.target_assignment)
            direct = row_normalize(record.q_direct)
            if direct.shape != budget["q_induced"].shape:
                raise ValueError(
                    f"Direct Q {direct.shape} does not match induced Q "
                    f"{budget['q_induced'].shape} for {mapping} {pair}"
                )
            crossfit = crossfit_macro_q_error(
                budget["observed"],
                budget["source_assignment"],
                folds=cfg.profile.crossfit_folds,
                weights=budget["weights"],
                seed=cfg.profile.random_seed,
            )
            predicted_direct = row_normalize(budget["source_assignment"] @ direct)
            direct_js = float(np.sum(budget["weights"] * js_rows(budget["observed"], predicted_direct)))
            induced_js = float(
                np.sum(budget["weights"] * js_rows(budget["observed"], budget["predicted_induced"]))
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
                    "direct_induced_q_js": float(np.mean(js_rows(direct, budget["q_induced"]))),
                    "induced_mean_js": induced_js,
                    "closure_js": direct_js,
                    "within_dynamics_js": induced_js,
                    "q_operational_gap_js": direct_js - induced_js,
                    "source_states": budget["source_assignment"].shape[1],
                    "target_states": budget["target_assignment"].shape[1],
                    "Keff_source": float(np.exp(-np.sum((budget["source_assignment"].mean(axis=0)) * np.log(np.maximum(budget["source_assignment"].mean(axis=0), EPS))))),
                    "Keff_target": float(np.exp(-np.sum((budget["target_assignment"].mean(axis=0)) * np.log(np.maximum(budget["target_assignment"].mean(axis=0), EPS))))),
                    **crossfit,
                }
            )
    return pd.DataFrame(rows)


def build_cross_representation_consistency(cfg, records_by_pair: dict[tuple[str, str], object]) -> pd.DataFrame:
    def assignment_at_time(mapping: str, time: str) -> tuple[dict[str, int], np.ndarray]:
        index = cfg.times.index(time)
        if index < len(cfg.times) - 1:
            record = records_by_pair[(mapping, cfg.adjacent_pairs[index])]
            assignment, spots = record.source_assignment, record.spots_s
        else:
            record = records_by_pair[(mapping, cfg.adjacent_pairs[-1])]
            assignment, spots = record.target_assignment, record.spots_t
        return {spot: index for index, spot in enumerate(map(str, spots))}, normalize_assignment(assignment)

    rows: list[dict[str, object]] = []
    for time in cfg.times:
        assignments = {mapping: assignment_at_time(mapping, time) for mapping in cfg.mapping_names}
        for index, left in enumerate(cfg.mapping_names):
            for right in cfg.mapping_names[index + 1 :]:
                left_lookup, left_matrix = assignments[left]
                right_lookup, right_matrix = assignments[right]
                common = sorted(set(left_lookup) & set(right_lookup))
                if not common:
                    raise ValueError(f"No common spots for {left} and {right} at {time}")
                left_values = left_matrix[[left_lookup[spot] for spot in common]]
                right_values = right_matrix[[right_lookup[spot] for spot in common]]
                joint = (left_values.T @ right_values) / len(common)
                px, py = joint.sum(axis=1), joint.sum(axis=0)
                valid = joint > EPS
                mutual_information = float(np.sum(joint[valid] * np.log2(joint[valid] / (px[:, None] * py[None, :])[valid])))
                h_left, h_right = float(entropy(px)), float(entropy(py))
                rows.append(
                    {
                        "time": time,
                        "mapping_a": left,
                        "mapping_b": right,
                        "common_spots": len(common),
                        "soft_NMI": mutual_information / np.sqrt(h_left * h_right) if h_left > EPS and h_right > EPS else np.nan,
                    }
                )
    return pd.DataFrame(rows)
