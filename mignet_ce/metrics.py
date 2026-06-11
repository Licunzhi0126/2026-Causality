from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


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
    def build_transition_kernel(source_embedding: np.ndarray, target_embedding: np.ndarray) -> np.ndarray:
        sim_matrix = cosine_similarity(source_embedding, target_embedding)
        P = np.exp(sim_matrix)
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def calculate_metrics_for_pairs(
        self,
        lower_feat: Sequence[np.ndarray],
        upper_feat: Sequence[np.ndarray],
        time_points: Sequence[str],
        pairs: Sequence[Tuple[int, int]],
        organ: str,
        lower_layer: str,
        upper_layer: str,
        kraskov_k: int = 3,
    ) -> pd.DataFrame:
        rows = []
        for t0, t1 in pairs:
            y_t = lower_feat[t0]
            y_t1 = lower_feat[t1]
            x_t = upper_feat[t0]
            x_t1 = upper_feat[t1]

            h_base = self.kraskov_conditional_entropy(y_t1, y_t, k=kraskov_k)
            h_full = self.kraskov_conditional_entropy(y_t1, np.hstack([y_t, x_t]), k=kraskov_k)
            h_macro = self.kraskov_conditional_entropy(y_t1, x_t, k=kraskov_k)
            p_lower = self.build_transition_kernel(y_t, y_t1)
            p_upper = self.build_transition_kernel(x_t, x_t1)
            te_raw = h_base - h_full
            di_raw = h_macro - h_full
            rows.append(
                {
                    "organ": organ,
                    "lower_layer": lower_layer,
                    "upper_layer": upper_layer,
                    "time_pair": f"{time_points[t0]}->{time_points[t1]}",
                    "lag": t1 - t0,
                    "H_base": h_base,
                    "H_full": h_full,
                    "H_macro": h_macro,
                    "EI_lower": effective_information(p_lower),
                    "EI_upper": effective_information(p_upper),
                    "TE_raw": te_raw,
                    "TE": max(0.0, te_raw),
                    "DI_raw": di_raw,
                    "DI": max(0.0, di_raw),
                }
            )
        return pd.DataFrame(rows)
