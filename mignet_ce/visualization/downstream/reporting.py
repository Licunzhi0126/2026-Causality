from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

from .config import DownstreamConfig
from .mappings import MAPPINGS


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


def audit_unified_outputs(
    cfg,
    tables: dict[str, pd.DataFrame],
    figure_pngs: Iterable[Path],
    *,
    deltaei_contract: pd.DataFrame | None = None,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    full_manifest = cfg.full_cache_root / "full_benchmark_manifest.json"
    add("full benchmark manifest exists", full_manifest.exists(), full_manifest)
    if full_manifest.exists():
        import json

        payload = json.loads(full_manifest.read_text(encoding="utf-8"))
        add("exactly six full DeltaEI jobs", payload.get("deltaei_job_count") == 6, payload.get("deltaei_job_count"))
        add("exactly nine natural NG_KLot caches", payload.get("natural_cache_count") == 9, payload.get("natural_cache_count"))
        add("locked full profile id", payload.get("profile_id") == cfg.profile.profile_id, payload.get("profile_id"))
        add(
            "full model-space cache protocol",
            payload.get("cache_protocol") == "full_model_space_v2",
            payload.get("cache_protocol"),
        )

    metrics = tables["metrics"]
    optimized = metrics[metrics["mapping"].astype(str).str.startswith("Optimized ")]
    add(
        "optimized metrics use complete K=40 model space",
        bool(
            (optimized["model_source_states"] == 40).all()
            and (optimized["model_target_states"] == 40).all()
        ),
        optimized[
            ["mapping", "time_pair", "model_source_states", "model_target_states"]
        ].to_dict(orient="records"),
    )
    add(
        "optimized downstream DeltaEI matches trainer summaries",
        bool(
            optimized["delta_EI_consistency_error"].notna().all()
            and (optimized["delta_EI_consistency_error"].abs() <= 1e-5).all()
        ),
        float(optimized["delta_EI_consistency_error"].abs().max()),
    )
    if deltaei_contract is not None:
        add(
            "six-row formal DeltaEI contract passes",
            len(deltaei_contract) == 6 and bool(deltaei_contract["passed"].all()),
            deltaei_contract.loc[
                ~deltaei_contract["passed"], ["mapping", "time_pair"]
            ].to_dict(orient="records"),
        )

    mapping_tables = (
        "metrics",
        "states",
        "spatial_spots",
        "closure",
        "spatial",
        "effective",
        "mechanism",
        "null",
        "fate",
        "perturbation",
    )
    expected_mappings = set(MAPPINGS)
    for name in mapping_tables:
        table = tables[name]
        present = set(table["mapping"].dropna().astype(str)) if "mapping" in table else set()
        add(f"{name}: all four mappings", expected_mappings.issubset(present), sorted(present))

    closure = tables["closure"]
    identity = (
        closure["I_available_bits"]
        - closure["I_retained_bits"]
        - closure["closure_leakage_bits"]
    ).abs()
    add("closure information identity", bool((identity < 1e-8).all()), float(identity.max()))
    add(
        "direct-induced Q consistency complete",
        "direct_induced_q_js" in closure and closure["direct_induced_q_js"].notna().all(),
    )

    consistency = tables["consistency"]
    pair_count = consistency[["mapping_a", "mapping_b"]].drop_duplicates().shape[0]
    add("all six representation pairs", pair_count == 6, pair_count)
    add(
        "consistency covers all four times",
        set(consistency["time"].astype(str)) == set(cfg.times),
        sorted(consistency["time"].astype(str).unique()),
    )

    pngs = tuple(map(Path, figure_pngs))
    add("exactly nine expected PNG figures", len(pngs) == 9 and all(path.exists() for path in pngs), [path.name for path in pngs])
    pdfs = tuple(path.with_suffix(".pdf") for path in pngs)
    add("exactly nine matching PDF figures", len(pdfs) == 9 and all(path.exists() for path in pdfs), [path.name for path in pdfs])
    return pd.DataFrame(checks)


def unified_run_manifest(
    cfg,
    tables: dict[str, pd.DataFrame],
    figure_pngs: Iterable[Path],
    *,
    deltaei_contract: pd.DataFrame | None = None,
) -> dict[str, object]:
    pngs = tuple(map(Path, figure_pngs))
    return {
        "workflow": "formal_full_unified_downstream",
        "cache_protocol": "full_model_space_v2",
        "model_state_contract": "full_soft_k",
        "profile_id": cfg.profile.profile_id,
        "full_profile": cfg.profile.__dict__,
        "organ": cfg.organ,
        "time_points": list(cfg.times),
        "mappings": list(MAPPINGS),
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
        "figures_png": [str(path) for path in pngs],
        "figures_pdf": [str(path.with_suffix('.pdf')) for path in pngs],
        "full_cache_root": str(cfg.full_cache_root),
        "deltaei_contract_passed": (
            bool(deltaei_contract["passed"].all()) if deltaei_contract is not None else None
        ),
    }
