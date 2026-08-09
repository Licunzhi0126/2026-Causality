from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

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
EPS = 1e-12


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.maximum(
        np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got {values.shape}.")
    sums = values.sum(axis=1, keepdims=True)
    zero = sums[:, 0] <= EPS
    if np.any(zero):
        values[zero] = 1.0 / max(values.shape[1], 1)
        sums = values.sum(axis=1, keepdims=True)
    return values / np.maximum(sums, EPS)


def normalize_assignment(matrix: np.ndarray) -> np.ndarray:
    return row_normalize(matrix)


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    p = row_normalize(probabilities)
    return -np.sum(p * np.log2(np.maximum(p, EPS)), axis=1)


def kl_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = row_normalize(left)
    q = row_normalize(right)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.repeat(q, p.shape[0], axis=0)
    if p.shape != q.shape:
        raise ValueError(f"KL row shapes differ: {p.shape} vs {q.shape}.")
    return np.sum(p * (np.log2(np.maximum(p, EPS)) - np.log2(np.maximum(q, EPS))), axis=1)


def js_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = row_normalize(left)
    q = row_normalize(right)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.repeat(q, p.shape[0], axis=0)
    if p.shape != q.shape:
        raise ValueError(f"JS row shapes differ: {p.shape} vs {q.shape}.")
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_rows(p, midpoint) + 0.5 * kl_rows(q, midpoint)


def state_level_ei(probabilities: np.ndarray) -> np.ndarray:
    p = row_normalize(probabilities)
    reference = p.mean(axis=0, keepdims=True)
    return kl_rows(p, reference)


def effective_information(probabilities: np.ndarray) -> float:
    return float(np.mean(state_level_ei(probabilities)))


def weighted_information(probabilities: np.ndarray, weights: np.ndarray) -> float:
    p = row_normalize(probabilities)
    w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if p.shape[0] != len(w):
        raise ValueError(f"Weights length {len(w)} != row count {p.shape[0]}.")
    w = w / max(float(w.sum()), EPS)
    reference = (w @ p).reshape(1, -1)
    return float(np.sum(w * kl_rows(p, np.repeat(reference, len(p), axis=0))))


def relative_frobenius(predicted: np.ndarray, observed: np.ndarray) -> float:
    denominator = float(np.linalg.norm(observed))
    return float(np.linalg.norm(np.asarray(predicted) - np.asarray(observed)) / max(denominator, EPS))


def effective_state_number(weights: np.ndarray) -> float:
    values = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    values = values / max(float(values.sum()), EPS)
    positive = values[values > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def read_h5ad_units_coords(path: Path) -> tuple[list[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        key = obs.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        raw = np.asarray(obs[str(key)])
        units = [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in raw]
        if "obsm" in handle and "spatial" in handle["obsm"]:
            node = handle["obsm/spatial"]
            if isinstance(node, h5py.Dataset):
                coords = np.asarray(node, dtype=float)[:, :2]
            else:
                if "data" in node:
                    coords = np.asarray(node["data"], dtype=float)[:, :2]
                elif {"x", "y"}.issubset(node.keys()):
                    coords = np.column_stack([np.asarray(node["x"], dtype=float), np.asarray(node["y"], dtype=float)])
                else:
                    columns = node.attrs.get("column-order", [])
                    columns = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in columns]
                    if len(columns) >= 2:
                        coords = np.column_stack([np.asarray(node[columns[0]], dtype=float), np.asarray(node[columns[1]], dtype=float)])
                    else:
                        coords = np.zeros((len(units), 2), dtype=float)
        elif {"x", "y"}.issubset(obs.keys()):
            coords = np.column_stack([np.asarray(obs["x"], dtype=float), np.asarray(obs["y"], dtype=float)])
        else:
            coords = np.zeros((len(units), 2), dtype=float)
    return units, coords


def load_domain_assignment(
    map_path: Path,
    spot_units: Sequence[str],
    state_order: Sequence[str],
) -> np.ndarray:
    frame = pd.read_csv(map_path)
    id_column = next((name for name in ("spot_id", "cell_name", "obs_id") if name in frame.columns), None)
    state_column = next((name for name in ("domain_id", "domain_label", "seurat_clusters") if name in frame.columns), None)
    if id_column is None or state_column is None:
        raise ValueError(f"Domain map {map_path} needs spot and domain columns.")
    frame[id_column] = frame[id_column].astype(str)
    frame[state_column] = frame[state_column].astype(str)
    lookup = frame.drop_duplicates(id_column).set_index(id_column)[state_column]
    missing = [unit for unit in map(str, spot_units) if unit not in lookup.index]
    if missing:
        raise ValueError(f"Domain map {map_path} misses {len(missing)} spots; examples={missing[:5]}.")
    states = list(map(str, state_order))
    state_lookup = {state: index for index, state in enumerate(states)}
    labels = lookup.loc[list(map(str, spot_units))].astype(str).to_numpy()
    unknown = sorted(set(labels) - set(states))
    if unknown:
        raise ValueError(f"Domain map contains states absent from H5AD order: {unknown[:10]}.")
    assignment = np.zeros((len(labels), len(states)), dtype=float)
    assignment[np.arange(len(labels)), [state_lookup[label] for label in labels]] = 1.0
    return assignment


def overlap_assignment(lower_spot_assignment: np.ndarray, upper_spot_assignment: np.ndarray) -> np.ndarray:
    lower = normalize_assignment(lower_spot_assignment)
    upper = normalize_assignment(upper_spot_assignment)
    mass = lower.sum(axis=0)
    matrix = lower.T @ upper
    matrix = np.divide(matrix, mass[:, None], out=np.zeros_like(matrix), where=mass[:, None] > EPS)
    return row_normalize(matrix)


def to_hard_assignment(matrix: np.ndarray) -> np.ndarray:
    """Convert an assignment matrix to a deterministic one-hot macrostate map.

    Dynamic-closure primary analyses use this representation so Seurat and
    optimized coarse-grainings are evaluated with the same information object:
    a discrete macrostate label, not a micro-specific soft coordinate.
    """
    values = normalize_assignment(matrix)
    hard = np.zeros_like(values, dtype=float)
    hard[np.arange(values.shape[0]), np.argmax(values, axis=1)] = 1.0
    return hard


def compact_hard_assignment(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harden and drop empty nominal macrostates, returning active column ids."""
    hard = to_hard_assignment(matrix)
    active = np.flatnonzero(hard.sum(axis=0) > EPS)
    if len(active) == 0:
        raise ValueError("Hard assignment has no active macrostates.")
    return hard[:, active], active


def state_balanced_micro_weights(hard_assignment: np.ndarray) -> np.ndarray:
    """Give every active macrostate equal intervention mass and every member equal mass within state."""
    hard = to_hard_assignment(hard_assignment)
    mass = hard.sum(axis=0)
    active = mass > EPS
    k = int(active.sum())
    if k == 0:
        raise ValueError("No active macrostates for state-balanced weighting.")
    inv = np.zeros_like(mass, dtype=float)
    inv[active] = 1.0 / (k * mass[active])
    weights = hard @ inv
    weights /= max(float(weights.sum()), EPS)
    return weights


def induced_macro_q(
    observed_micro_to_macro: np.ndarray,
    source_assignment: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Micro-induced/oracle macro kernel, i.e. the KL centroid within each macrostate.

    This is an *existence/reference* kernel extracted from micro dynamics. It is
    not evidence by itself that an independently estimated macro predictor can
    reproduce the dynamics.
    """
    observed = row_normalize(observed_micro_to_macro)
    source = normalize_assignment(source_assignment)
    if weights is None:
        weights = np.full(source.shape[0], 1.0 / source.shape[0], dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if len(w) != source.shape[0]:
        raise ValueError("weights length does not match source assignment rows")
    w /= max(float(w.sum()), EPS)
    weighted_source = source * w[:, None]
    mass = weighted_source.sum(axis=0)
    q = weighted_source.T @ observed
    q = np.divide(q, mass[:, None], out=np.zeros_like(q), where=mass[:, None] > EPS)
    return row_normalize(q), mass


def best_macro_q(observed_micro_to_macro: np.ndarray, source_assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated compatibility alias for :func:`induced_macro_q`."""
    warnings.warn(
        "best_macro_q() is deprecated; use induced_macro_q(). The kernel is a micro-induced/oracle reference, not a generic JS-optimal or independently runnable macro model.",
        DeprecationWarning,
        stacklevel=2,
    )
    return induced_macro_q(observed_micro_to_macro, source_assignment)


def information_closure_budget(
    p_micro: np.ndarray,
    source_assignment: np.ndarray,
    target_assignment: np.ndarray,
    *,
    weighting: str = "state_balanced",
    low_signal_threshold_bits: float | None = None,
) -> dict[str, object]:
    """Compute a hard-macro information budget in bits.

    For deterministic M_t=f(X_t):
        I(X_t;M_{t+1}) = I(M_t;M_{t+1}) + I(X_t;M_{t+1}|M_t).
    The last term is the primary closure leakage: lower is better.
    """
    p = row_normalize(p_micro)
    h_t, active_t = compact_hard_assignment(source_assignment)
    h_tp, active_tp = compact_hard_assignment(target_assignment)
    if p.shape != (h_t.shape[0], h_tp.shape[0]):
        raise ValueError(f"P {p.shape} incompatible with hard assignments {h_t.shape}, {h_tp.shape}")
    observed = row_normalize(p @ h_tp)
    if weighting == "state_balanced":
        weights = state_balanced_micro_weights(h_t)
    elif weighting == "uniform_spot":
        weights = np.full(h_t.shape[0], 1.0 / h_t.shape[0], dtype=float)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    q_induced, macro_mass = induced_macro_q(observed, h_t, weights)
    predicted = row_normalize(h_t @ q_induced)
    i_available = weighted_information(observed, weights)
    i_macro = weighted_information(q_induced, macro_mass)
    leakage_direct = float(np.sum(weights * kl_rows(observed, predicted)))
    identity_error = float(abs(i_available - i_macro - leakage_direct))
    if identity_error > 1e-8:
        raise AssertionError(
            f"Information closure identity failed: available={i_available}, macro={i_macro}, leakage={leakage_direct}, error={identity_error}"
        )
    if low_signal_threshold_bits is None:
        # A transparent practical threshold, not a significance test. With model
        # transition probabilities there is no sampling null unless biological
        # replicates/count uncertainty are supplied.
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(h_tp.shape[1], 2)))
    signal_status = "informative" if i_available >= low_signal_threshold_bits else "low-signal"
    sufficiency = (i_macro / i_available) if signal_status == "informative" and i_available > EPS else np.nan
    return {
        "observed": observed,
        "source_hard": h_t,
        "target_hard": h_tp,
        "source_active_columns": active_t,
        "target_active_columns": active_tp,
        "weights": weights,
        "q_induced": q_induced,
        "macro_mass": macro_mass,
        "predicted_induced": predicted,
        "I_available_bits": float(i_available),
        "I_macro_retained_bits": float(i_macro),
        "closure_leakage_bits": float(leakage_direct),
        "macro_sufficiency": float(sufficiency) if np.isfinite(sufficiency) else np.nan,
        "information_identity_error": identity_error,
        "signal_status": signal_status,
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
    """Evaluate the canonical closure budget for precomputed future-macro profiles."""
    observed = row_normalize(observed_micro_to_macro)
    source_hard, active_source = compact_hard_assignment(source_assignment)
    if observed.shape[0] != source_hard.shape[0]:
        raise ValueError(
            f"Observed row count {observed.shape[0]} != source assignment rows {source_hard.shape[0]}."
        )
    if weighting == "state_balanced":
        weights = state_balanced_micro_weights(source_hard)
    elif weighting == "uniform_spot":
        weights = np.full(source_hard.shape[0], 1.0 / source_hard.shape[0], dtype=float)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    q_induced, macro_mass = induced_macro_q(observed, source_hard, weights)
    predicted = row_normalize(source_hard @ q_induced)
    i_available = weighted_information(observed, weights)
    i_retained = weighted_information(q_induced, macro_mass)
    leakage = float(np.sum(weights * kl_rows(observed, predicted)))
    identity_error = float(abs(i_available - i_retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(
            f"Information closure identity failed: available={i_available}, retained={i_retained}, leakage={leakage}, error={identity_error}"
        )
    if low_signal_threshold_bits is None:
        low_signal_threshold_bits = max(0.01, 0.01 * np.log2(max(observed.shape[1], 2)))
    signal_status = "informative" if i_available >= low_signal_threshold_bits else "low-signal"
    sufficiency = i_retained / i_available if signal_status == "informative" and i_available > EPS else np.nan
    return {
        "observed": observed,
        "source_hard": source_hard,
        "source_active_columns": active_source,
        "weights": weights,
        "q_induced": q_induced,
        "macro_mass": macro_mass,
        "predicted_induced": predicted,
        "I_available_bits": float(i_available),
        "I_macro_retained_bits": float(i_retained),
        "closure_leakage_bits": leakage,
        "macro_sufficiency": float(sufficiency) if np.isfinite(sufficiency) else np.nan,
        "information_identity_error": identity_error,
        "signal_status": signal_status,
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
    """Cross-fit the induced macro Q across microstates within each hard macrostate.

    This separates in-sample partition existence from held-out predictive
    generalization. Folds are stratified by macrostate; tiny states fall back to
    leave-one-out where possible. Singleton states are excluded from the error
    instead of receiving an artificially perfect self-prediction.
    """
    obs = row_normalize(observed)
    hard = to_hard_assignment(source_hard)
    labels = np.argmax(hard, axis=1)
    n = len(labels)
    if weights is None:
        weights = state_balanced_micro_weights(hard)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0); w /= max(float(w.sum()), EPS)
    rng = np.random.default_rng(seed)
    pred = np.zeros_like(obs)
    estimable = np.zeros(n, dtype=bool)
    singleton = np.zeros(n, dtype=bool)
    for state in np.unique(labels):
        idx = np.flatnonzero(labels == state)
        if len(idx) == 1:
            singleton[idx] = True
            continue
        estimable[idx] = True
        shuffled = idx.copy(); rng.shuffle(shuffled)
        k = min(max(2, int(folds)), len(idx))
        fold_ids = np.arange(len(shuffled)) % k
        for fold in range(k):
            test = shuffled[fold_ids == fold]
            train = shuffled[fold_ids != fold]
            tw = w[train]
            tw = tw / max(float(tw.sum()), EPS)
            q = (tw @ obs[train]).reshape(1, -1)
            pred[test] = q
    coverage = float(np.sum(w[estimable]))
    singleton_fraction = float(np.sum(w[singleton]))
    if coverage <= EPS:
        return {
            "crossfit_kl_bits": np.nan,
            "crossfit_js": np.nan,
            "crossfit_coverage": 0.0,
            "singleton_weight_fraction": singleton_fraction,
        }
    normalized_weights = w[estimable] / coverage
    return {
        "crossfit_kl_bits": float(np.sum(normalized_weights * kl_rows(obs[estimable], pred[estimable]))),
        "crossfit_js": float(np.sum(normalized_weights * js_rows(obs[estimable], pred[estimable]))),
        "crossfit_coverage": coverage,
        "singleton_weight_fraction": singleton_fraction,
    }


@dataclass
class ClosureResult:
    summary: dict[str, float | str]
    state_table: pd.DataFrame
    source_residuals: pd.DataFrame
    observed: np.ndarray
    q_induced: np.ndarray
    predicted_induced: np.ndarray
    q_direct: np.ndarray | None
    predicted_direct: np.ndarray | None
    source_active_columns: np.ndarray
    target_active_columns: np.ndarray

    @property
    def q_best(self) -> np.ndarray:  # compatibility
        return self.q_induced

    @property
    def predicted_best(self) -> np.ndarray:  # compatibility
        return self.predicted_induced


def closure_analysis(
    p_micro: np.ndarray,
    source_assignment: np.ndarray,
    target_assignment: np.ndarray,
    *,
    label: str,
    time_pair: str,
    q_direct: np.ndarray | None = None,
    source_state_names: Sequence[str] | None = None,
    weighting: str = "state_balanced",
    crossfit_folds: int = 5,
    low_signal_threshold_bits: float | None = None,
) -> ClosureResult:
    # Primary macro object is deterministic/hard for every method, including
    # optimized coarse-graining. This removes the unfair micro-specific soft-S_i
    # predictor that previously made optimized JS closure artificially easier.
    source_soft = normalize_assignment(source_assignment)
    original_confidence = np.max(source_soft, axis=1)
    budget = information_closure_budget(
        p_micro, source_assignment, target_assignment, weighting=weighting,
        low_signal_threshold_bits=low_signal_threshold_bits,
    )
    p = row_normalize(p_micro)
    s_t = budget["source_hard"]
    s_tp = budget["target_hard"]
    observed = budget["observed"]
    q_induced = budget["q_induced"]
    predicted_induced = budget["predicted_induced"]
    weights = budget["weights"]
    macro_mass = budget["macro_mass"]
    active_src = budget["source_active_columns"]
    active_tgt = budget["target_active_columns"]
    residual_js = js_rows(observed, predicted_induced)
    residual_kl = kl_rows(observed, predicted_induced)
    crossfit = crossfit_macro_q_error(observed, s_t, folds=crossfit_folds, weights=weights)

    i_micro_future_macro = float(budget["I_available_bits"])
    i_macro_future_macro = float(budget["I_macro_retained_bits"])
    leakage = float(budget["closure_leakage_bits"])

    # Downward reach is retained only as a descriptive sensitivity metric. Use the
    # same hard source macro and state-balanced intervention semantics.
    q_to_future_micro, source_weights = induced_macro_q(p, s_t, weights)
    i_macro_future_micro = weighted_information(q_to_future_micro, source_weights)
    i_micro_future_micro = weighted_information(p, weights)

    direct = None
    predicted_direct = None
    direct_js = np.nan
    direct_frob = np.nan
    direct_ei = np.nan
    direct_excess_js = np.nan
    direct_excess_frob = np.nan
    direct_q_gap_js = np.nan
    if q_direct is not None:
        raw_direct = row_normalize(q_direct)
        # Compare only active hard macrostates; empty optimized prototypes must not
        # count as states or provide hidden capacity.
        if raw_direct.shape[0] < int(np.max(active_src)) + 1 or raw_direct.shape[1] < int(np.max(active_tgt)) + 1:
            raise ValueError(f"Direct Q shape {raw_direct.shape} is incompatible with active hard columns for {label}.")
        direct = row_normalize(raw_direct[np.ix_(active_src, active_tgt)])
        if direct.shape != q_induced.shape:
            raise ValueError(f"Direct Q compact shape {direct.shape} != induced Q shape {q_induced.shape} for {label}.")
        predicted_direct = row_normalize(s_t @ direct)
        direct_js = float(np.sum(weights * js_rows(observed, predicted_direct)))
        direct_frob = relative_frobenius(predicted_direct * weights[:, None] ** 0.5, observed * weights[:, None] ** 0.5)
        direct_ei = weighted_information(direct, macro_mass)
        induced_js = float(np.sum(weights * residual_js))
        direct_excess_js = direct_js - induced_js
        induced_frob = relative_frobenius(predicted_induced * weights[:, None] ** 0.5, observed * weights[:, None] ** 0.5)
        direct_excess_frob = direct_frob - induced_frob
        direct_q_gap_js = float(np.sum(macro_mass * js_rows(q_induced, direct)))

    weighted_mean_js = float(np.sum(weights * residual_js))
    weighted_mean_kl = float(np.sum(weights * residual_kl))
    weighted_frob = relative_frobenius(predicted_induced * weights[:, None] ** 0.5, observed * weights[:, None] ** 0.5)
    summary: dict[str, float | str] = {
        "mapping": label,
        "time_pair": time_pair,
        "primary_macro_representation": "hard",
        "primary_weighting": weighting,
        "source_nodes": float(p.shape[0]),
        "target_nodes": float(p.shape[1]),
        "source_macro_states": float(s_t.shape[1]),
        "target_macro_states": float(s_tp.shape[1]),
        "EI_micro_dynamics": i_micro_future_micro,
        "I_micro_to_future_macro": i_micro_future_macro,
        "I_macro_to_future_macro": i_macro_future_macro,
        "micro_residual_information": leakage,
        "closure_leakage_bits": leakage,
        "macro_sufficiency": budget["macro_sufficiency"],
        "residual_fraction": (leakage / i_micro_future_macro) if i_micro_future_macro > EPS else np.nan,
        "signal_status": budget["signal_status"],
        "signal_threshold_bits": budget["signal_threshold_bits"],
        "information_identity_error": budget["information_identity_error"],
        "I_macro_to_future_micro": i_macro_future_micro,
        "downward_reach_ratio": i_macro_future_micro / max(i_micro_future_micro, EPS),
        # Keep legacy columns but explicitly define them as induced/oracle diagnostics.
        "best_closure_mean_js": weighted_mean_js,
        "best_closure_median_js": float(np.median(residual_js)),
        "best_closure_p95_js": float(np.quantile(residual_js, 0.95)),
        "best_closure_max_js": float(np.max(residual_js)),
        "best_closure_mean_kl": weighted_mean_kl,
        "best_closure_relative_frobenius": weighted_frob,
        "induced_closure_mean_js": weighted_mean_js,
        "induced_closure_mean_kl_bits": weighted_mean_kl,
        "induced_closure_relative_frobenius": weighted_frob,
        "crossfit_closure_kl_bits": crossfit["crossfit_kl_bits"],
        "crossfit_closure_js": crossfit["crossfit_js"],
        "crossfit_closure_coverage": crossfit["crossfit_coverage"],
        "crossfit_singleton_weight_fraction": crossfit["singleton_weight_fraction"],
        "EI_induced_Q_uniform": effective_information(q_induced),
        "EI_induced_Q_weighted": i_macro_future_macro,
        "Keff_source_macro": effective_state_number(macro_mass),
        "direct_closure_mean_js": direct_js,
        "direct_closure_relative_frobenius": direct_frob,
        "EI_direct_Q": direct_ei,
        "direct_excess_js_above_best": direct_excess_js,
        "direct_excess_frobenius_above_best": direct_excess_frob,
        "independent_q_excess_js": direct_excess_js,
        "independent_q_gap_js": direct_q_gap_js,
    }

    if source_state_names is None:
        names = [f"S{int(col)}" for col in active_src]
    else:
        all_names = list(map(str, source_state_names))
        if len(all_names) <= int(np.max(active_src)):
            raise ValueError("source_state_names does not cover active source assignment columns.")
        names = [all_names[int(col)] for col in active_src]
    state_ei = state_level_ei(q_induced)
    state_rows: list[dict[str, float | str]] = []
    for state_index, name in enumerate(names):
        weights = s_t[:, state_index]
        total = float(weights.sum())
        if total <= EPS:
            continue
        state_rows.append(
            {
                "mapping": label,
                "time_pair": time_pair,
                "state": name,
                "state_index": state_index,
                "mass": total,
                "mean_js": float(np.sum(weights * residual_js) / total),
                "mean_kl": float(np.sum(weights * residual_kl) / total),
                "state_ei": float(state_ei[state_index]),
                "mean_assignment": float(np.mean(weights)),
            }
        )

    source_table = pd.DataFrame(
        {
            "mapping": label,
            "time_pair": time_pair,
            "source_index": np.arange(len(residual_js)),
            "closure_js": residual_js,
            "closure_kl": residual_kl,
            "source_ei_to_future_macro": state_level_ei(observed),
            "source_ei_micro": state_level_ei(p),
            "assignment_confidence": original_confidence,
            "hard_state": np.argmax(s_t, axis=1),
        }
    )
    return ClosureResult(
        summary=summary,
        state_table=pd.DataFrame(state_rows),
        source_residuals=source_table,
        observed=observed,
        q_induced=q_induced,
        predicted_induced=predicted_induced,
        q_direct=direct,
        predicted_direct=predicted_direct,
        source_active_columns=np.asarray(active_src, dtype=int),
        target_active_columns=np.asarray(active_tgt, dtype=int),
    )


def compose_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("At least one matrix is required for composition.")
    result = row_normalize(matrices[0])
    for matrix in matrices[1:]:
        result = row_normalize(result @ row_normalize(matrix))
    return result


def multistep_closure(
    p_chain: Sequence[np.ndarray],
    assignments: Sequence[np.ndarray],
    q_chain: Sequence[np.ndarray],
    *,
    mapping: str,
    interval: str,
) -> dict[str, float | str]:
    """Hard-macro multi-step closure/semigroup diagnostic.

    The induced endpoint kernel is an oracle reference; composed independent Q
    tests operational semigroup consistency.
    """
    if len(assignments) != len(p_chain) + 1 or len(q_chain) != len(p_chain):
        raise ValueError("Multistep chain lengths are inconsistent.")
    hard_assignments=[]; active_cols=[]
    for a in assignments:
        h, active=compact_hard_assignment(a); hard_assignments.append(h); active_cols.append(active)
    p_composed = compose_matrices(p_chain)
    observed = row_normalize(p_composed @ hard_assignments[-1])
    weights = state_balanced_micro_weights(hard_assignments[0])
    # Compact each independent macro Q to active state coordinates at the
    # corresponding source/target time before composition.
    q_compact=[]
    for step,q in enumerate(q_chain):
        qq=row_normalize(q)
        q_compact.append(row_normalize(qq[np.ix_(active_cols[step],active_cols[step+1])]))
    q_composed = compose_matrices(q_compact)
    predicted = row_normalize(hard_assignments[0] @ q_composed)
    q_induced, macro_mass = induced_macro_q(observed, hard_assignments[0], weights)
    predicted_induced = row_normalize(hard_assignments[0] @ q_induced)
    induced_js=float(np.sum(weights*js_rows(observed,predicted_induced)))
    composed_js=float(np.sum(weights*js_rows(observed,predicted)))
    return {
        "mapping": mapping,
        "interval": interval,
        "steps": float(len(p_chain)),
        "composed_mean_js": composed_js,
        "composed_relative_frobenius": relative_frobenius(predicted * weights[:,None]**0.5, observed * weights[:,None]**0.5),
        "best_possible_mean_js": induced_js,
        "best_possible_relative_frobenius": relative_frobenius(predicted_induced * weights[:,None]**0.5, observed * weights[:,None]**0.5),
        "induced_endpoint_mean_js": induced_js,
        "semigroup_excess_js": composed_js-induced_js,
        "semigroup_excess_frobenius": relative_frobenius(predicted * weights[:,None]**0.5, observed * weights[:,None]**0.5)-relative_frobenius(predicted_induced * weights[:,None]**0.5, observed * weights[:,None]**0.5),
        "EI_composed_Q": weighted_information(q_composed, macro_mass),
        "EI_best_Q": weighted_information(q_induced, macro_mass),
        "I_endpoint_available_bits": weighted_information(observed,weights),
        "I_endpoint_macro_retained_bits": weighted_information(q_induced,macro_mass),
        "endpoint_closure_leakage_bits": float(np.sum(weights*kl_rows(observed,predicted_induced))),
    }


def align_assignments(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    left = normalize_assignment(reference)
    right = normalize_assignment(candidate)
    if left.shape != right.shape:
        raise ValueError(f"Assignment shapes differ at shared time: {left.shape} vs {right.shape}.")
    overlap = left.T @ right
    rows, cols = linear_sum_assignment(-overlap)
    permutation = np.empty(right.shape[1], dtype=int)
    permutation[rows] = cols
    aligned = right[:, permutation]
    hard_left = np.argmax(left, axis=1)
    hard_right = np.argmax(aligned, axis=1)
    normalized_overlap = overlap[rows, cols].sum() / max(float(len(left)), EPS)
    return aligned, permutation, {
        "soft_overlap": float(normalized_overlap),
        "hard_ari": float(adjusted_rand_score(hard_left, hard_right)),
        "mean_reference_confidence": float(np.max(left, axis=1).mean()),
        "mean_candidate_confidence": float(np.max(right, axis=1).mean()),
    }


def boundary_fraction(coords: np.ndarray, hard_labels: np.ndarray, k: int = 6) -> np.ndarray:
    values = np.asarray(coords, dtype=float)
    labels = np.asarray(hard_labels)
    effective = min(max(int(k) + 1, 2), len(values))
    indices = NearestNeighbors(n_neighbors=effective).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    return np.mean(labels[indices] != labels[:, None], axis=1)


def connected_components(coords: np.ndarray, hard_labels: np.ndarray, k: int = 6) -> dict[int, int]:
    values = np.asarray(coords, dtype=float)
    labels = np.asarray(hard_labels, dtype=int)
    effective = min(max(int(k) + 1, 2), len(values))
    indices = NearestNeighbors(n_neighbors=effective).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    result: dict[int, int] = {}
    for state in np.unique(labels):
        members = np.flatnonzero(labels == state)
        member_set = set(members.tolist())
        seen: set[int] = set()
        count = 0
        for start in members:
            if int(start) in seen:
                continue
            count += 1
            stack = [int(start)]
            seen.add(int(start))
            while stack:
                node = stack.pop()
                for neighbor in indices[node]:
                    neighbor = int(neighbor)
                    if neighbor in member_set and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        result[int(state)] = count
    return result


def correlation_summary(frame: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    clean = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3 or clean[x].nunique() < 2 or clean[y].nunique() < 2:
        return {"n": float(len(clean)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = pearsonr(clean[x], clean[y])
    sr, spv = spearmanr(clean[x], clean[y])
    return {"n": float(len(clean)), "pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(spv)}


def matched_partition_null(
    observed_micro_to_macro: np.ndarray,
    source_assignment: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Matched hard-partition null under state-balanced macro interventions."""
    observed = row_normalize(observed_micro_to_macro)
    source, _ = compact_hard_assignment(source_assignment)
    hard = np.argmax(source, axis=1)
    states = source.shape[1]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    threshold=max(0.01,0.01*np.log2(max(observed.shape[1],2)))
    for repeat in range(int(repeats)):
        permuted_hard = rng.permutation(hard)
        permuted = np.zeros((len(hard), states), dtype=float)
        permuted[np.arange(len(hard)), permuted_hard] = 1.0
        weights=state_balanced_micro_weights(permuted)
        q, mass = induced_macro_q(observed, permuted, weights)
        predicted = row_normalize(permuted @ q)
        i_xb = weighted_information(observed,weights)
        i_ab = weighted_information(q,mass)
        leak=float(np.sum(weights*kl_rows(observed,predicted)))
        rows.append({
            "repeat": float(repeat),
            "macro_sufficiency": (i_ab/i_xb) if i_xb>=threshold else np.nan,
            "residual_fraction": (leak/i_xb) if i_xb>=threshold else np.nan,
            "closure_leakage_bits": leak,
            "macro_predictive_bits": i_ab,
            "available_information_bits": i_xb,
            "signal_status": "informative" if i_xb>=threshold else "low-signal",
            "mean_js": float(np.sum(weights*js_rows(observed,predicted))),
            "relative_frobenius": relative_frobenius(predicted*weights[:,None]**0.5, observed*weights[:,None]**0.5),
        })
    return pd.DataFrame(rows)
