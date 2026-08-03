from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DownstreamConfig


def summarize_findings(
    cfg: DownstreamConfig,
    metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    closure: pd.DataFrame,
    single_closure: pd.DataFrame,
    consistency: pd.DataFrame,
    random_null: pd.DataFrame,
    effective: pd.DataFrame,
    correlations: pd.DataFrame,
    perturbation: pd.DataFrame,
) -> dict[str, object]:
    gain = pd.to_numeric(metrics["EI_gain"])
    adjacent = metrics[metrics["lag"] == 1]
    pivot = decomposition.pivot(index="time_pair", columns="space", values=["H_effect", "H_noise"])
    noise_reduction = pivot[("H_noise", "lower")] - pivot[("H_noise", "upper")]
    effect_change = pivot[("H_effect", "upper")] - pivot[("H_effect", "lower")]
    null_summary = []
    for pair in random_null["time_pair"].unique():
        subset = random_null[random_null["time_pair"] == pair]
        observed = float(subset.loc[subset["kind"] == "observed", "EI"].iloc[0])
        values = subset.loc[subset["kind"] == "matched_random", "EI"].to_numpy()
        null_summary.append(
            {
                "time_pair": pair,
                "observed": observed,
                "null_mean": float(values.mean()),
                "z": float((observed - values.mean()) / max(values.std(ddof=1), 1e-12)),
                "p_empirical": float((1 + np.sum(values >= observed)) / (1 + len(values))),
            }
        )
    grn_rows = correlations[
        (correlations["outcome"] == "state_ei")
        & (correlations["predictor"] == "grn_concentration")
    ]
    max_dose = perturbation[np.isclose(perturbation["dose"], 1.0)]
    perturbation_summary = max_dose.groupby("target")["ei_drop_mean"].mean().to_dict()
    return {
        "run_scope": {
            "network_method": cfg.network_method,
            "pij_method": cfg.pij_method,
            "organ": cfg.organ,
            "layer_pair": f"{cfg.lower_layer}->{cfg.upper_layer}",
            "time_points": list(cfg.times),
        },
        "interpretation_boundaries": {
            "matched_random": "Preserves K40 spot counts; it is not a random learned WYT assignment.",
            "grn_cci": "Exploratory associations, not demonstrated biological causality.",
            "virtual_perturbation": "Pij row homogenization, not a full CCI/GRN intervention rerun.",
        },
        "delta_ei_mean": float(gain.mean()),
        "delta_ei_median": float(gain.median()),
        "delta_ei_min": float(gain.min()),
        "delta_ei_max": float(gain.max()),
        "adjacent_delta_ei_mean": float(adjacent["EI_gain"].mean()),
        "positive_fraction": float((gain > 0).mean()),
        "noise_reduction_mean": float(noise_reduction.mean()),
        "effect_diversity_change_mean": float(effect_change.mean()),
        "noise_reduction_dominant_pairs": int((noise_reduction > np.abs(effect_change)).sum()),
        "multistep_relative_error_mean_K40": float(
            closure.loc[closure["space"] == "upper", "relative_frobenius"].mean()
        ),
        "multistep_relative_error_mean_K150": float(
            closure.loc[closure["space"] == "lower", "relative_frobenius"].mean()
        ),
        "single_step_closure_mean_direct_K40": float(
            single_closure.loc[
                single_closure["macro_Q"] == "direct_K40_Pij",
                "relative_frobenius",
            ].mean()
        ),
        "single_step_closure_mean_overlap_Q": float(
            single_closure.loc[
                single_closure["macro_Q"] == "overlap_aggregated",
                "relative_frobenius",
            ].mean()
        ),
        "single_step_closure_mean_best_fit": float(
            single_closure.loc[
                single_closure["macro_Q"] == "clipped_least_squares",
                "relative_frobenius",
            ].mean()
        ),
        "multiscale_relative_error_mean": float(consistency["relative_frobenius"].mean()),
        "random_null": null_summary,
        "effective_states": effective.to_dict(orient="records"),
        "grn_concentration_state_ei_correlation": grn_rows.iloc[0].to_dict() if len(grn_rows) else {},
        "perturbation_mean_full_dose_drop": {
            key: float(value) for key, value in perturbation_summary.items()
        },
    }
