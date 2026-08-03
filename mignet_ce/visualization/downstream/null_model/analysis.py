from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_domain_map, load_pij, load_units
from ..metrics import aggregate_transition_by_overlap, ei_decomposition


def build_random_null(
    cfg: DownstreamConfig,
    counts_by_time: dict[str, np.ndarray],
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, object]] = []
    map_cache: dict[str, pd.DataFrame] = {}
    units_cache: dict[str, tuple[list[str], list[str]]] = {}
    for time in cfg.times:
        lower_units = load_units(cfg.pair_archive, "lower", time)
        upper_units = load_units(cfg.pair_archive, "upper", time)
        lower_map = load_domain_map(cfg.data_root, cfg.lower_layer, time, cfg.organ)
        upper_map = load_domain_map(cfg.data_root, cfg.upper_layer, time, cfg.organ)
        map_cache[time] = lower_map[["spot_id", "domain_id"]].rename(columns={"domain_id": "lower"}).merge(
            upper_map[["spot_id", "domain_id"]].rename(columns={"domain_id": "upper"}),
            on="spot_id",
            how="inner",
        )
        units_cache[time] = (lower_units, upper_units)
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        p_lower = load_pij(cfg.pair_archive, pair, "lower")
        observed = aggregate_transition_by_overlap(p_lower, counts_by_time[source], counts_by_time[target])
        rows.append(
            {
                "time_pair": pair,
                "kind": "observed",
                "repeat": -1,
                "EI": ei_decomposition(observed)["EI"],
            }
        )
        for repeat in range(cfg.random_repeats):
            random_counts: list[np.ndarray] = []
            for time in (source, target):
                merged = map_cache[time]
                lower_units, upper_units = units_cache[time]
                labels = merged["upper"].to_numpy(copy=True)
                rng.shuffle(labels)
                random_frame = pd.DataFrame({"lower": merged["lower"].to_numpy(), "upper": labels})
                table = pd.crosstab(random_frame["lower"], random_frame["upper"]).reindex(
                    index=lower_units,
                    columns=upper_units,
                    fill_value=0,
                )
                random_counts.append(table.to_numpy(dtype=float))
            q_matrix = aggregate_transition_by_overlap(p_lower, random_counts[0], random_counts[1])
            rows.append(
                {
                    "time_pair": pair,
                    "kind": "matched_random",
                    "repeat": repeat,
                    "EI": ei_decomposition(q_matrix)["EI"],
                }
            )
    return pd.DataFrame(rows)
