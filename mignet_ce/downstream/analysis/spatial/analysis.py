from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from ..dynamic_closure.analysis import entropy_rows, state_level_ei


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
            macro_ei = state_level_ei(record.q_direct)
            spot_ei = record.source_assignment @ macro_ei
            for spot, coords, value in zip(record.spots_s, record.coords_s, spot_ei):
                rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "source_time": source,
                        "target_time": target,
                        "spot_id": spot,
                        "state_ei": float(value),
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
        assignment = np.asarray(record.source_assignment, dtype=float)
        coords = np.asarray(record.coords_s, dtype=float)
        frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
        graph = _spatial_graph(frame, cfg.spatial_knn)
        count = assignment.shape[0]
        state_ei_values = state_level_ei(record.q_direct)
        span = max(float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))), 1e-12)
        for state in range(assignment.shape[1]):
            values = assignment[:, state]
            mass = float(values.sum())
            center = (values[:, None] * coords).sum(axis=0) / max(mass, 1e-12)
            radius = float(np.sum(values * np.linalg.norm(coords - center, axis=1)) / max(mass, 1e-12))
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
                    "soft_mass": mass,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "neighborhood_smoothness": float(np.mean(np.abs(values[graph.tocoo().row] - values[graph.tocoo().col]))) if graph.nnz else np.nan,
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
                assignment = record.source_assignment
            else:
                record = records_by_pair[(mapping, cfg.adjacent_pairs[-1])]
                assignment = record.target_assignment
            usage = assignment.mean(axis=0)
            positive = usage[usage > 0]
            usage_entropy = float(-np.sum(positive * np.log2(positive)))
            nominal = assignment.shape[1]
            rows.append(
                {
                    "mapping": mapping,
                    "time": time,
                    "nominal_k": nominal,
                    "Keff": float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0,
                    "soft_usage_entropy": usage_entropy,
                    "soft_usage_cv": float(np.std(usage) / max(np.mean(usage), 1e-12)),
                    "max_usage": float(usage.max()),
                    "min_usage": float(positive.min()),
                    "spot_count": int(assignment.shape[0]),
                    "assignment_confidence": float(np.mean(np.max(assignment, axis=1))),
                    "assignment_entropy_bits": float(np.mean(entropy_rows(assignment))),
                }
            )
    return pd.DataFrame(rows)
