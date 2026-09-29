from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mignet_ce.utils.matrix import row_softmax


def build_cosine_transition_kernel(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    source = np.asarray(source_embedding, dtype=float)
    target = np.asarray(target_embedding, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {source.shape} and {target.shape}.")
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros((source.shape[0], target.shape[0]), dtype=float)
    sim_matrix = cosine_similarity(source, target)
    return row_softmax(sim_matrix, temperature=temperature)
