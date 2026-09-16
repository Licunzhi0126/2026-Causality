from __future__ import annotations

import pandas as pd

from .mappings import MAPPINGS


def audit_unified_analysis_outputs(
    cfg,
    tables: dict[str, pd.DataFrame],
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
        expected_jobs = 3 * len(cfg.adjacent_pairs)
        add(
            "all optimized full DeltaEI jobs",
            payload.get("deltaei_job_count") == expected_jobs,
            payload.get("deltaei_job_count"),
        )
        add("exactly nine natural NG_KLot caches", payload.get("natural_cache_count") == 9, payload.get("natural_cache_count"))
        add("locked full profile id", payload.get("profile_id") == cfg.profile.profile_id, payload.get("profile_id"))
        add(
            "full model-space cache protocol",
            payload.get("cache_protocol") == "full_model_space_v3",
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
            "nine-row formal DeltaEI contract passes",
            len(deltaei_contract) == 9 and bool(deltaei_contract["passed"].all()),
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
        "null_distribution",
        "null_summary",
        "fate",
    )
    expected_mappings = set(MAPPINGS)
    for name in mapping_tables:
        table = tables[name]
        present = set(table["mapping"].dropna().astype(str)) if "mapping" in table else set()
        add(f"{name}: all five mappings", expected_mappings.issubset(present), sorted(present))

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
    expected_pair_count = len(MAPPINGS) * (len(MAPPINGS) - 1) // 2
    add("all representation pairs", pair_count == expected_pair_count, pair_count)
    add(
        "consistency covers all configured times",
        set(consistency["time"].astype(str)) == set(cfg.times),
        sorted(consistency["time"].astype(str).unique()),
    )

    null_summary = tables["null_summary"]
    add(
        "matched-null summaries retain reproducibility metadata",
        bool(
            {"observed_EI", "null_mean_EI", "effect_size_EI", "p_empirical", "z_score", "seed", "repeat_count"}
            <= set(null_summary.columns)
            and (null_summary["repeat_count"] == cfg.profile.matched_null_repeats).all()
        ),
        list(null_summary.columns),
    )
    return pd.DataFrame(checks)


def unified_run_manifest(
    cfg,
    tables: dict[str, pd.DataFrame],
    *,
    deltaei_contract: pd.DataFrame | None = None,
) -> dict[str, object]:
    return {
        "workflow": "formal_full_unified_downstream_analysis",
        "cache_protocol": "full_model_space_v3",
        "model_state_contract": "full_soft_k",
        "profile_id": cfg.profile.profile_id,
        "full_profile": cfg.profile.__dict__,
        "organ": cfg.organ,
        "time_points": list(cfg.times),
        "mappings": list(MAPPINGS),
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
        "full_cache_root": str(cfg.full_cache_root),
        "deltaei_contract_passed": (
            bool(deltaei_contract["passed"].all()) if deltaei_contract is not None else None
        ),
    }
