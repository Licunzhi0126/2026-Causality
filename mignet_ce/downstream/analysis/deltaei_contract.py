from __future__ import annotations

"""Strict agreement checks between DeltaEI training and formal downstream use."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .dynamic_closure.analysis import effective_information
from .mappings import (
    MAPPING_COMPLETE,
    MAPPING_MATURITY,
    MAPPING_TWO_STAGE,
    OPTIMIZED_METHOD_BY_MAPPING,
    is_optimized,
)
from .metrics import row_normalize


DELTAEI_CONTRACT_TOLERANCE = 1e-5
OPTIMIZED_MAPPINGS = (MAPPING_COMPLETE, MAPPING_MATURITY, MAPPING_TWO_STAGE)
FORMAL_TIME_PAIRS = ("11.5->12.5", "12.5->13.5", "13.5->14.5")
FORMAL_K = 40
FORMAL_EPOCHS = 1500
FORMAL_NMF_MAX_ITER = 300
FORMAL_CACHE_PROTOCOL = "full_model_space_v3"


def _required_summary_float(summary: dict[str, object], key: str, context: str) -> float:
    if key not in summary:
        raise ValueError(f"DeltaEI summary is missing {key!r} for {context}")
    value = float(summary[key])
    if not np.isfinite(value):
        raise ValueError(f"DeltaEI summary contains non-finite {key!r} for {context}")
    return value


def deltaei_contract_row(record, *, tolerance: float = DELTAEI_CONTRACT_TOLERANCE) -> dict[str, object]:
    """Recompute a record's full-model EI quantities and compare with training."""

    if not is_optimized(record.mapping):
        raise ValueError(f"DeltaEI trainer contract applies only to optimized mappings: {record.mapping}")
    context = f"{record.mapping} {record.pair}"
    micro = effective_information(row_normalize(record.p_model_micro))
    macro = effective_information(row_normalize(record.q_model_full))
    delta = macro - micro
    summary_micro = _required_summary_float(record.summary, "EI_micro_fixed", context)
    summary_macro = _required_summary_float(record.summary, "EI_macro_best_checkpoint", context)
    summary_delta = _required_summary_float(record.summary, "delta_EI_best_checkpoint", context)
    errors = {
        "EI_micro_abs_error": abs(micro - summary_micro),
        "EI_macro_abs_error": abs(macro - summary_macro),
        "delta_EI_abs_error": abs(delta - summary_delta),
    }
    passed = all(value <= tolerance for value in errors.values())
    return {
        "mapping": record.mapping,
        "method": record.method,
        "time_pair": record.pair,
        "K": int(record.summary.get("K", record.q_model_full.shape[0])),
        "best_epoch": record.summary.get("best_epoch", np.nan),
        "model_source_states": int(record.q_model_full.shape[0]),
        "model_target_states": int(record.q_model_full.shape[1]),
        "hardK_t": record.summary.get("hardK_t", np.nan),
        "hardK_tp": record.summary.get("hardK_tp", np.nan),
        "Keff_t": record.summary.get("Keff_t", np.nan),
        "Keff_tp": record.summary.get("Keff_tp", np.nan),
        "summary_EI_micro": summary_micro,
        "recomputed_EI_micro": micro,
        "summary_EI_macro": summary_macro,
        "recomputed_EI_macro": macro,
        "summary_delta_EI": summary_delta,
        "recomputed_delta_EI": delta,
        **errors,
        "tolerance": float(tolerance),
        "passed": passed,
    }


def build_deltaei_contract_table(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
    metrics: pd.DataFrame | None = None,
    *,
    tolerance: float = DELTAEI_CONTRACT_TOLERANCE,
    fail: bool = True,
) -> pd.DataFrame:
    rows = [
        deltaei_contract_row(records_by_pair[(mapping, pair)], tolerance=tolerance)
        for mapping in OPTIMIZED_MAPPINGS
        for pair in cfg.adjacent_pairs
    ]
    table = pd.DataFrame(rows)
    if metrics is not None:
        downstream = metrics[
            ["mapping", "time_pair", "delta_EI_matched_spot"]
        ].rename(columns={"delta_EI_matched_spot": "downstream_delta_EI"})
        table = table.merge(downstream, on=["mapping", "time_pair"], how="left", validate="one_to_one")
        table["downstream_delta_EI_abs_error"] = (
            table["downstream_delta_EI"] - table["summary_delta_EI"]
        ).abs()
        table["passed"] &= table["downstream_delta_EI_abs_error"] <= tolerance
    if fail and not bool(table["passed"].all()):
        failures = table.loc[~table["passed"]].to_dict(orient="records")
        raise RuntimeError(f"Full-model DeltaEI contract failed: {failures}")
    return table


def audit_full_cache_root(
    full_cache_root: Path,
    metrics_csv: Path,
    *,
    tolerance: float = DELTAEI_CONTRACT_TOLERANCE,
) -> pd.DataFrame:
    """Audit a completed full cache without constructing downstream records."""

    root = Path(full_cache_root).resolve()
    metrics = pd.read_csv(metrics_csv)
    rows: list[dict[str, object]] = []
    for mapping in OPTIMIZED_MAPPINGS:
        method = OPTIMIZED_METHOD_BY_MAPPING[mapping]
        method_root = root / "optimized_coarse" / method
        if not method_root.exists():
            raise FileNotFoundError(f"Missing optimized cache directory: {method_root}")
        observed_pairs = {
            path.name.replace("_to_", "->"): path
            for path in method_root.iterdir()
            if path.is_dir()
        }
        if set(observed_pairs) != set(FORMAL_TIME_PAIRS):
            raise RuntimeError(
                f"Formal cache pairs for {mapping} must be {FORMAL_TIME_PAIRS}; "
                f"found {sorted(observed_pairs)}"
            )
        for pair in FORMAL_TIME_PAIRS:
            pair_root = observed_pairs[pair]
            from .preparation import required_optimized_outputs

            missing = [
                name for name in required_optimized_outputs(method) if not (pair_root / name).exists()
            ]
            if missing:
                raise FileNotFoundError(f"Incomplete optimized cache {pair_root}: {missing}")
            summary = json.loads((pair_root / "summary.json").read_text(encoding="utf-8"))
            trainer = json.loads((pair_root / "config.json").read_text(encoding="utf-8"))
            feature_manifest = json.loads(
                (pair_root / "feature_manifest.json").read_text(encoding="utf-8")
            )
            full_manifest = json.loads(
                (pair_root / "downstream_full_manifest.json").read_text(encoding="utf-8")
            )
            micro = effective_information(np.load(pair_root / "PIJ_micro_train.npy"))
            macro_matrix = row_normalize(np.load(pair_root / "PIJ_macro_train.npy"))
            macro = effective_information(macro_matrix)
            delta = macro - micro
            soft_t = row_normalize(np.load(pair_root / "S_t.npy"))
            soft_tp = row_normalize(np.load(pair_root / "S_tp.npy"))
            summary_micro = _required_summary_float(summary, "EI_micro_fixed", f"{mapping} {pair}")
            summary_macro = _required_summary_float(summary, "EI_macro_best_checkpoint", f"{mapping} {pair}")
            summary_delta = _required_summary_float(summary, "delta_EI_best_checkpoint", f"{mapping} {pair}")
            selected = metrics[
                (metrics["mapping"] == mapping) & (metrics["time_pair"] == pair)
            ]
            downstream_delta = (
                float(selected["delta_EI_matched_spot"].iloc[0]) if len(selected) == 1 else np.nan
            )
            errors = {
                "EI_micro_abs_error": abs(micro - summary_micro),
                "EI_macro_abs_error": abs(macro - summary_macro),
                "delta_EI_abs_error": abs(delta - summary_delta),
                "downstream_delta_EI_abs_error": abs(downstream_delta - summary_delta),
            }
            k = int(summary.get("K", trainer.get("k", macro_matrix.shape[0])))
            epochs = int(trainer.get("epochs", -1))
            nmf_max_iter = feature_manifest.get("provenance", {}).get(
                "nmf_max_iter_used"
            )
            protocol = full_manifest.get("cache_protocol")
            from mignet_ce.coarse_frontends.method_specs import get_coarse_method_spec

            spec = get_coarse_method_spec(method)
            dimensions_valid = (
                k == FORMAL_K
                and macro_matrix.shape == (FORMAL_K, FORMAL_K)
                and soft_t.shape[1] == FORMAL_K
                and soft_tp.shape[1] == FORMAL_K
            )
            formal_profile_valid = (
                epochs == FORMAL_EPOCHS
                and nmf_max_iter == FORMAL_NMF_MAX_ITER
                and protocol == FORMAL_CACHE_PROTOCOL
                and full_manifest.get("model_state_contract") == "full_soft_k"
                and full_manifest.get("method") == method
                and full_manifest.get("frontend") == (spec.frontend or method)
                and full_manifest.get("training_mode") == spec.training_mode
                and full_manifest.get("objective_version") == spec.objective_version
            )
            rows.append(
                {
                    "mapping": mapping,
                    "method": method,
                    "time_pair": pair,
                    "epochs": epochs,
                    "best_epoch": summary.get("best_epoch", np.nan),
                    "K": k,
                    "nmf_max_iter_used": nmf_max_iter,
                    "cache_protocol": protocol,
                    "hardK_t": summary.get("hardK_t", np.nan),
                    "hardK_tp": summary.get("hardK_tp", np.nan),
                    "Keff_t": summary.get("Keff_t", np.nan),
                    "Keff_tp": summary.get("Keff_tp", np.nan),
                    "summary_EI_micro": summary_micro,
                    "recomputed_EI_micro": micro,
                    "summary_EI_macro": summary_macro,
                    "recomputed_EI_macro": macro,
                    "summary_delta_EI": summary_delta,
                    "recomputed_delta_EI": delta,
                    "downstream_delta_EI": downstream_delta,
                    **errors,
                    "dimensions_valid": dimensions_valid,
                    "formal_profile_valid": formal_profile_valid,
                    "tolerance": float(tolerance),
                    "passed": (
                        dimensions_valid
                        and formal_profile_valid
                        and all(value <= tolerance for value in errors.values())
                    ),
                }
            )
    table = pd.DataFrame(rows)
    expected_count = len(OPTIMIZED_MAPPINGS) * len(FORMAL_TIME_PAIRS)
    if len(table) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} optimized DeltaEI caches, found {len(table)} under {root}"
        )
    return table
