from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from ..config import DownstreamConfig
from ..io import load_domain_map
from ..dynamic_closure.analysis import entropy_rows, state_level_ei
from ..mappings import is_optimized


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


def build_unified_spatial_spots(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    """Expand spot and macro state-EI contributions onto the shared spot substrate."""

    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        base = records_by_pair[(cfg.mapping_names[0], pair)]
        spot_ei = state_level_ei(base.p)
        for index, (spot, coords, value) in enumerate(zip(base.spots_s, base.coords_s, spot_ei)):
            rows.append(
                {
                    "mapping": "Spot",
                    "time_pair": pair,
                    "source_time": source,
                    "target_time": target,
                    "spot_id": spot,
                    "state_index": index,
                    "state_ei": float(value),
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                }
            )
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            labels = np.argmax(record.hs, axis=1)
            macro_ei = state_level_ei(record.q_direct)
            for spot, label, coords in zip(record.spots_s, labels, record.coords_s):
                rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "source_time": source,
                        "target_time": target,
                        "spot_id": spot,
                        "state_index": int(label),
                        "state_ei": float(macro_ei[label]),
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                    }
                )
    return pd.DataFrame(rows)


def build_unified_spatial_metrics(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mapping, pair), record in records_by_pair.items():
        source, _target = pair.split("->")
        labels = np.argmax(record.hs, axis=1)
        coords = np.asarray(record.coords_s, dtype=float)
        frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
        graph = _spatial_graph(frame, cfg.spatial_knn)
        count = len(labels)
        state_ei_values = state_level_ei(record.q_direct)
        span = max(float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))), 1e-12)
        for state in range(record.hs.shape[1]):
            indices = np.flatnonzero(labels == state)
            mask = np.zeros(count, dtype=bool)
            mask[indices] = True
            subgraph = graph[indices][:, indices]
            components = (
                connected_components(subgraph, directed=False, return_labels=False)
                if len(indices)
                else 0
            )
            selected_edges = graph[indices].tocoo()
            boundary = int(np.sum(labels[selected_edges.col] != state))
            edge_count = max(len(selected_edges.col), 1)
            center = coords[indices].mean(axis=0) if len(indices) else np.zeros(2)
            radius = (
                float(np.mean(np.linalg.norm(coords[indices] - center, axis=1)))
                if len(indices)
                else 0.0
            )
            values = mask.astype(float)
            centered = values - values.mean()
            weight_sum = float(graph.sum())
            denominator = float(np.sum(centered**2))
            moran = (
                float(count / weight_sum * (centered @ (graph @ centered)) / denominator)
                if weight_sum > 0 and denominator > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "time": source,
                    "state_index": state,
                    "spot_count": len(indices),
                    "connected_components": int(components),
                    "fragmentation": float(components / max(len(indices), 1)),
                    "boundary_ratio": float(boundary / edge_count),
                    "radius_norm": radius / span,
                    "moran_i": moran,
                    "state_ei": float(state_ei_values[state]),
                }
            )
    return pd.DataFrame(rows)


def build_unified_effective_states(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_index, time in enumerate(cfg.times):
        for mapping in cfg.mapping_names:
            if time_index < len(cfg.times) - 1:
                record = records_by_pair[(mapping, cfg.adjacent_pairs[time_index])]
                hard, soft = record.hs, record.soft_s
            else:
                record = records_by_pair[(mapping, cfg.adjacent_pairs[-1])]
                hard, soft = record.ht, record.soft_t
            labels = np.argmax(hard, axis=1)
            counts = np.bincount(labels, minlength=hard.shape[1]).astype(float)
            usage = counts / counts.sum()
            positive = usage[usage > 0]
            usage_entropy = float(-np.sum(positive * np.log2(positive)))
            nominal = int(record.summary.get("K", soft.shape[1])) if is_optimized(mapping) else hard.shape[1]
            rows.append(
                {
                    "mapping": mapping,
                    "time": time,
                    "active_k": int(np.count_nonzero(counts)),
                    "nominal_k": nominal,
                    "Keff": float(2**usage_entropy),
                    "usage_entropy_bits": usage_entropy,
                    "max_usage": float(usage.max()),
                    "min_usage": float(positive.min()),
                    "spot_count": int(counts.sum()),
                    "assignment_confidence": float(np.mean(np.max(soft, axis=1))),
                    "assignment_entropy_bits": float(np.mean(entropy_rows(soft))),
                }
            )
    return pd.DataFrame(rows)
