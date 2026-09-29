from __future__ import annotations

import numpy as np
import pandas as pd

from ..dynamic_closure.analysis import (
    effective_information,
    information_closure_budget,
    information_closure_budget_from_observed,
)


def build_unified_matched_null(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    """State-size-matched source partition null for every formal mapping."""

    rng = np.random.default_rng(cfg.profile.random_seed)
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            baseline = information_closure_budget(record.p, record.source_assignment, record.target_assignment)
            observed = baseline["observed"]
            rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "kind": "observed",
                    "repeat": -1,
                    "seed": cfg.profile.random_seed,
                    "repeat_count": cfg.profile.matched_null_repeats,
                    "EI": effective_information(baseline["q_induced"]),
                    "closure_leakage_bits": baseline["closure_leakage_bits"],
                }
            )
            for repeat in range(cfg.profile.matched_null_repeats):
                shuffled_source = record.source_assignment[rng.permutation(record.source_assignment.shape[0])]
                shuffled_target = record.target_assignment[rng.permutation(record.target_assignment.shape[0])]
                random_observed = record.p @ shuffled_target
                random_budget = information_closure_budget_from_observed(random_observed, shuffled_source)
                rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "kind": "matched_random",
                        "repeat": repeat,
                        "seed": cfg.profile.random_seed,
                        "repeat_count": cfg.profile.matched_null_repeats,
                        "EI": effective_information(random_budget["q_induced"]),
                        "closure_leakage_bits": random_budget["closure_leakage_bits"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_unified_matched_null(null_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize a persisted null distribution without regenerating samples."""

    rows: list[dict[str, object]] = []
    for (mapping, pair), subset in null_table.groupby(["mapping", "time_pair"], sort=False):
        observed_rows = subset.loc[subset["kind"] == "observed"]
        random_rows = subset.loc[subset["kind"] == "matched_random"]
        if len(observed_rows) != 1 or random_rows.empty:
            raise ValueError(f"Incomplete matched-null table for {mapping} {pair}")
        observed = float(observed_rows["EI"].iloc[0])
        values = random_rows["EI"].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "mapping": mapping,
                "time_pair": pair,
                "observed_EI": observed,
                "null_mean_EI": mean,
                "null_std_EI": std,
                "effect_size_EI": observed - mean,
                "z_score": (observed - mean) / max(std, 1e-12),
                "p_empirical": float((1 + np.count_nonzero(values >= observed)) / (1 + len(values))),
                "seed": int(random_rows["seed"].iloc[0]) if "seed" in random_rows else np.nan,
                "repeat_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)
