from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from mignet_ce.pij.base import PairFeatures
from mignet_ce.transition.cosine import build_cosine_transition_kernel


def _as_clean_nonnegative_array(matrix: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    return arr


def _stabilize_nonnegative_factor(factor: np.ndarray, eps: float) -> np.ndarray:
    factor = np.nan_to_num(factor, nan=eps, posinf=eps, neginf=eps)
    return np.maximum(factor, eps)


def pairwise_joint_nmf(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_components: int = 5,
    max_iter: int = 300,
    seed: int = 42,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorize two matrices with a pair-specific shared H.

    X_source ~= W_source H and X_target ~= W_target H.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    eps = max(float(eps), 1e-12)
    source = _as_clean_nonnegative_array(X_source, name="X_source")
    target = _as_clean_nonnegative_array(X_target, name="X_target")
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "pairwise_joint_nmf requires source and target matrices with identical column counts; "
            f"got {source.shape[1]} and {target.shape[1]}."
        )

    rng = np.random.default_rng(seed)
    w_source = rng.random((source.shape[0], n_components), dtype=float) + eps
    w_target = rng.random((target.shape[0], n_components), dtype=float) + eps
    h_matrix = rng.random((n_components, source.shape[1]), dtype=float) + eps
    for _ in range(max_iter):
        w_source *= (source @ h_matrix.T) / (w_source @ h_matrix @ h_matrix.T + eps)
        w_target *= (target @ h_matrix.T) / (w_target @ h_matrix @ h_matrix.T + eps)
        w_source = _stabilize_nonnegative_factor(w_source, eps)
        w_target = _stabilize_nonnegative_factor(w_target, eps)

        numerator = w_source.T @ source + w_target.T @ target
        denominator = (w_source.T @ w_source + w_target.T @ w_target) @ h_matrix + eps
        h_matrix *= numerator / denominator
        h_matrix = _stabilize_nonnegative_factor(h_matrix, eps)
    return w_source, w_target, h_matrix


def pairwise_shared_core_directed_nmf(
    A_source: np.ndarray,
    A_target: np.ndarray,
    n_components: int = 5,
    max_iter: int = 300,
    seed: int = 42,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Factorize two directed square adjacency matrices with a shared core B.

    A_source ~= U_source B V_source.T and A_target ~= U_target B V_target.T.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    eps = max(float(eps), 1e-12)
    source = _as_clean_nonnegative_array(A_source, name="A_source")
    target = _as_clean_nonnegative_array(A_target, name="A_target")
    if source.shape[0] != source.shape[1]:
        raise ValueError(f"A_source must be square, got shape {source.shape}.")
    if target.shape[0] != target.shape[1]:
        raise ValueError(f"A_target must be square, got shape {target.shape}.")

    rng = np.random.default_rng(seed)
    u_source = rng.random((source.shape[0], n_components), dtype=float) + eps
    v_source = rng.random((source.shape[0], n_components), dtype=float) + eps
    u_target = rng.random((target.shape[0], n_components), dtype=float) + eps
    v_target = rng.random((target.shape[0], n_components), dtype=float) + eps
    core = rng.random((n_components, n_components), dtype=float) + eps

    for _ in range(max_iter):
        source_vtv = v_source.T @ v_source
        source_utu = u_source.T @ u_source
        target_vtv = v_target.T @ v_target
        target_utu = u_target.T @ u_target

        u_source *= (source @ v_source @ core.T) / (u_source @ core @ source_vtv @ core.T + eps)
        u_target *= (target @ v_target @ core.T) / (u_target @ core @ target_vtv @ core.T + eps)
        u_source = _stabilize_nonnegative_factor(u_source, eps)
        u_target = _stabilize_nonnegative_factor(u_target, eps)

        source_utu = u_source.T @ u_source
        target_utu = u_target.T @ u_target
        v_source *= (source.T @ u_source @ core) / (v_source @ core.T @ source_utu @ core + eps)
        v_target *= (target.T @ u_target @ core) / (v_target @ core.T @ target_utu @ core + eps)
        v_source = _stabilize_nonnegative_factor(v_source, eps)
        v_target = _stabilize_nonnegative_factor(v_target, eps)

        source_vtv = v_source.T @ v_source
        target_vtv = v_target.T @ v_target
        numerator = u_source.T @ source @ v_source + u_target.T @ target @ v_target
        denominator = source_utu @ core @ source_vtv + target_utu @ core @ target_vtv + eps
        core *= numerator / denominator
        core = _stabilize_nonnegative_factor(core, eps)

    return u_source, v_source, u_target, v_target, core


def effective_information(P, eps: float = 1e-12) -> float:
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return 0.0
    row_sums = P.sum(axis=1)
    if np.any(row_sums == 0):
        P[row_sums == 0] = 1.0 / P.shape[1]
    P = P / P.sum(axis=1, keepdims=True)
    Pj = P.mean(axis=0)
    h_j = -np.sum(Pj * np.log2(Pj + eps))
    h_j_given_i = -np.mean(np.sum(P * np.log2(P + eps), axis=1))
    return float(h_j - h_j_given_i)


class TemporalMetricsEngine:
    @staticmethod
    def temporal_joint_nmf(
        X_list: Sequence[np.ndarray],
        n_components: int = 5,
        max_iter: int = 300,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        if not X_list:
            raise ValueError("X_list is empty.")
        cols = X_list[0].shape[1]
        if any(X.shape[1] != cols for X in X_list):
            raise ValueError("All graph matrices must have the same number of columns.")

        rng = np.random.default_rng(seed)
        W_list = [rng.random((X.shape[0], n_components)) for X in X_list]
        H = rng.random((n_components, cols))
        eps = 1e-10
        for _ in range(max_iter):
            for i, X in enumerate(X_list):
                W_list[i] *= (X @ H.T) / (W_list[i] @ H @ H.T + eps)
            num = sum(W_list[i].T @ X_list[i] for i in range(len(X_list)))
            den = sum(W_list[i].T @ W_list[i] for i in range(len(X_list))) @ H + eps
            H *= num / den
        return W_list, H

    @staticmethod
    def kraskov_entropy(X: np.ndarray, k: int = 3) -> float:
        X = np.asarray(X, dtype=float)
        n_samples, d = X.shape
        if n_samples <= k or d == 0:
            return 0.0
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        rk = distances[:, k] + 1e-10
        return float(digamma(n_samples) - digamma(k) + d * np.mean(np.log(2 * rk)))

    @classmethod
    def kraskov_conditional_entropy(cls, Y: np.ndarray, X: np.ndarray, k: int = 3) -> float:
        XY = np.hstack([X, Y])
        return float(cls.kraskov_entropy(XY, k=k) - cls.kraskov_entropy(X, k=k))

    @staticmethod
    def global_scale_features(lower_feat_raw: Sequence[np.ndarray], upper_feat_raw: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lower_scaler = MinMaxScaler()
        upper_scaler = MinMaxScaler()
        lower_scaler.fit(np.vstack(lower_feat_raw))
        upper_scaler.fit(np.vstack(upper_feat_raw))
        return [lower_scaler.transform(x) for x in lower_feat_raw], [upper_scaler.transform(x) for x in upper_feat_raw]

    @staticmethod
    def build_time_pairs_all(time_points: Sequence[str]) -> List[Tuple[int, int]]:
        return list(combinations(range(len(time_points)), 2))

    @staticmethod
    def build_transition_kernel(source_embedding: np.ndarray, target_embedding: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return build_cosine_transition_kernel(source_embedding, target_embedding, temperature=temperature)

    def calculate_metrics_for_pairs(
        self,
        lower_feat: Sequence[np.ndarray],
        upper_feat: Sequence[np.ndarray],
        time_points: Sequence[str],
        pairs: Sequence[Tuple[int, int]],
        organ: str,
        lower_layer: str,
        upper_layer: str,
        pij_method: str = "joint_nmf",
        pij_temperature: float = 1.0,
        kraskov_k: int = 3,
        precomputed_p_lower: Dict[Tuple[int, int], np.ndarray] | None = None,
        precomputed_p_upper: Dict[Tuple[int, int], np.ndarray] | None = None,
        pairwise_lower_features: PairFeatures | None = None,
        pairwise_upper_features: PairFeatures | None = None,
        feature_alignment_space: str = "stable_upper_units",
    ) -> pd.DataFrame:
        rows = []
        for t0, t1 in pairs:
            if pairwise_lower_features is not None and (t0, t1) in pairwise_lower_features:
                y_t, y_t1 = pairwise_lower_features[(t0, t1)]
            else:
                y_t = lower_feat[t0]
                y_t1 = lower_feat[t1]
            if pairwise_upper_features is not None and (t0, t1) in pairwise_upper_features:
                x_t, x_t1 = pairwise_upper_features[(t0, t1)]
            else:
                x_t = upper_feat[t0]
                x_t1 = upper_feat[t1]

            conditional_metrics_compatible = (
                y_t.shape[0] == y_t1.shape[0] == x_t.shape[0]
            )
            if conditional_metrics_compatible:
                h_base = self.kraskov_conditional_entropy(y_t1, y_t, k=kraskov_k)
                h_full = self.kraskov_conditional_entropy(y_t1, np.hstack([y_t, x_t]), k=kraskov_k)
                h_macro = self.kraskov_conditional_entropy(y_t1, x_t, k=kraskov_k)
                te_raw = h_base - h_full
                di_raw = h_macro - h_full
                te = max(0.0, te_raw)
                di = max(0.0, di_raw)
            else:
                h_base = np.nan
                h_full = np.nan
                h_macro = np.nan
                te_raw = np.nan
                di_raw = np.nan
                te = np.nan
                di = np.nan
            p_lower = (
                precomputed_p_lower[(t0, t1)]
                if precomputed_p_lower is not None and (t0, t1) in precomputed_p_lower
                else self.build_transition_kernel(y_t, y_t1, temperature=pij_temperature)
            )
            p_upper = (
                precomputed_p_upper[(t0, t1)]
                if precomputed_p_upper is not None and (t0, t1) in precomputed_p_upper
                else self.build_transition_kernel(x_t, x_t1, temperature=pij_temperature)
            )
            ei_lower = effective_information(p_lower)
            ei_upper = effective_information(p_upper)
            if feature_alignment_space == "native_units":
                metric_alignment = (
                    "native_units_full"
                    if conditional_metrics_compatible
                    else "native_units_ei_only"
                )
            else:
                metric_alignment = "stable_upper_units"
            rows.append(
                {
                    "pij_method": pij_method,
                    "organ": organ,
                    "lower_layer": lower_layer,
                    "upper_layer": upper_layer,
                    "time_pair": f"{time_points[t0]}->{time_points[t1]}",
                    "lag": t1 - t0,
                    "H_base": h_base,
                    "H_full": h_full,
                    "H_macro": h_macro,
                    "EI_lower": ei_lower,
                    "EI_upper": ei_upper,
                    "EI_gain": ei_upper - ei_lower,
                    "metric_alignment": metric_alignment,
                    "TE_raw": te_raw,
                    "TE": te,
                    "DI_raw": di_raw,
                    "DI": di,
                }
            )
        return pd.DataFrame(rows)
