from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from ..config import DownstreamConfig
from ..io import load_domain_map


def build_spatial_state_maps(cfg: DownstreamConfig, state_table: pd.DataFrame) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for time_pair in cfg.adjacent_pairs:
        source, target = time_pair.split("->")
        for layer in (cfg.lower_layer, cfg.upper_layer):
            domain_map = load_domain_map(cfg.data_root, layer, source, cfg.organ)
            scores = state_table[
                (state_table["time_pair"] == time_pair) & (state_table["layer"] == layer)
            ][["state", "state_ei"]]
            frame = domain_map.merge(scores, left_on="domain_id", right_on="state", how="left")
            frame["state_ei"] = frame["state_ei"].fillna(0.0)
            frame["layer"] = layer
            frame["source_time"] = source
            frame["target_time"] = target
            frames.append(frame)
    return frames


def _spatial_graph(frame: pd.DataFrame, k: int) -> sp.csr_matrix:
    coordinates = frame[["x", "y"]].to_numpy(dtype=float)
    count = len(frame)
    if count < 2:
        return sp.csr_matrix((count, count), dtype=float)
    k_used = min(max(2, k + 1), count)
    indices = NearestNeighbors(n_neighbors=k_used).fit(coordinates).kneighbors(coordinates)[1]
    rows = np.repeat(np.arange(count), k_used - 1)
    columns = indices[:, 1:].reshape(-1)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(count, count)).tocsr()
    graph = ((graph + graph.T) > 0).astype(float).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def build_spatial_metrics(cfg: DownstreamConfig, state_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        time, _ = pair.split("->")
        for layer in (cfg.lower_layer, cfg.upper_layer):
            frame = load_domain_map(cfg.data_root, layer, time, cfg.organ).reset_index(drop=True)
            graph = _spatial_graph(frame, cfg.spatial_knn)
            degree_sum = float(graph.sum())
            coordinates = frame[["x", "y"]].to_numpy(dtype=float)
            global_span = max(float(np.ptp(coordinates[:, 0])), float(np.ptp(coordinates[:, 1])), 1.0)
            scores = state_table[
                (state_table["time_pair"] == pair) & (state_table["layer"] == layer)
            ].set_index("state")
            labels = frame["domain_id"].astype(str).to_numpy()
            for state in dict.fromkeys(labels.tolist()):
                mask = labels == state
                indices = np.flatnonzero(mask)
                subgraph = graph[indices][:, indices]
                components = int(connected_components(subgraph, directed=False, return_labels=False))
                cross = float(graph[indices][:, np.flatnonzero(~mask)].sum())
                total = float(graph[indices].sum())
                center = coordinates[indices].mean(axis=0)
                radius = float(np.linalg.norm(coordinates[indices] - center, axis=1).mean() / global_span)
                indicator = mask.astype(float)
                centered = indicator - indicator.mean()
                denominator = float(np.sum(centered**2))
                moran = float(
                    len(frame)
                    / max(degree_sum, 1.0)
                    * (centered @ (graph @ centered))
                    / max(denominator, 1e-12)
                )
                score = scores.loc[state] if state in scores.index else None
                rows.append(
                    {
                        "time": time,
                        "time_pair": pair,
                        "layer": layer,
                        "state": state,
                        "spot_count": int(mask.sum()),
                        "connected_components": components,
                        "fragmentation": components / max(int(mask.sum()), 1),
                        "boundary_ratio": cross / max(total, 1.0),
                        "radius_norm": radius,
                        "moran_i": moran,
                        "state_ei": float(score["state_ei"]) if score is not None else np.nan,
                        "transition_entropy": float(score["transition_entropy"]) if score is not None else np.nan,
                    }
                )
    return pd.DataFrame(rows)
