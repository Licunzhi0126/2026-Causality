from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def global_scale_feature_lists(
    lower_features: Sequence[np.ndarray],
    upper_features: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    lower_scaler = MinMaxScaler()
    upper_scaler = MinMaxScaler()
    lower_scaler.fit(np.vstack(lower_features))
    upper_scaler.fit(np.vstack(upper_features))
    return [lower_scaler.transform(x) for x in lower_features], [upper_scaler.transform(x) for x in upper_features]


def reduce_feature_lists(
    lower_features: Sequence[np.ndarray],
    upper_features: Sequence[np.ndarray],
    n_components: int | None,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], dict[str, object]]:
    if n_components is None or n_components <= 0:
        return list(lower_features), list(upper_features), {"reduced": False}
    all_features = np.vstack([*lower_features, *upper_features])
    max_components = min(all_features.shape[0], all_features.shape[1])
    if max_components <= 1:
        return list(lower_features), list(upper_features), {"reduced": False, "reason": "not_enough_rank"}
    actual = min(int(n_components), max_components)
    pca = PCA(n_components=actual, random_state=seed)
    pca.fit(all_features)
    return (
        [pca.transform(x) for x in lower_features],
        [pca.transform(x) for x in upper_features],
        {
            "reduced": True,
            "requested_components": int(n_components),
            "actual_components": int(actual),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    )
