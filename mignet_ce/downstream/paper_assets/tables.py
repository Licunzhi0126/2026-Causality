from __future__ import annotations

"""Paper-facing tables assembled from persisted scientific measurements."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import INPUT_SCALES, K10_LEVELS, OPTIMAL_TIME_PAIRS
from .inputs import normalize_time_pair, select_coarse_run


def _one_ei(metrics: pd.DataFrame, lower: str, upper: str, pair: str, *, method: str | None = None) -> float:
    selected = metrics.loc[
        metrics["lower_layer"].eq(lower)
        & metrics["upper_layer"].eq(upper)
        & metrics["time_pair"].eq(pair)
    ]
    if method is not None:
        selected = selected.loc[selected["Method"].eq(method)]
    if len(selected) != 1:
        raise ValueError(f"Expected one EI row for {method or 'primary'} {lower}->{upper} {pair}; got {len(selected)}")
    value = float(selected.iloc[0]["EI_gain"])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite EI for {lower}->{upper} {pair}")
    return value


def build_ei_ablation(
    metrics: pd.DataFrame,
    levels: Mapping[str, tuple[str, str]],
    methods: Sequence[str],
    time_pairs: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, (lower, upper) in levels.items():
        for method in methods:
            rows.append({
                "Method": method,
                "Hierarchy": label,
                **{pair: _one_ei(metrics, lower, upper, pair, method=method) for pair in time_pairs},
            })
    return pd.DataFrame(rows)


def build_ei_hierarchy(
    metrics: pd.DataFrame,
    levels: Mapping[str, tuple[str, str]],
    time_pairs: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame([
        {"Time pair": pair, **{
            label: _one_ei(metrics, lower, upper, pair)
            for label, (lower, upper) in levels.items()
        }}
        for pair in time_pairs
    ])


def _closure_quality_cell(value: object, status: object) -> object:
    """Keep raw CQ visible; mark low-signal ratios rather than blanking them."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(numeric):
        return np.nan
    return f"{numeric:.4f}†" if str(status) != "informative" else numeric


def build_seurat_cg_hierarchy(metrics: pd.DataFrame, time_pairs: Sequence[str]) -> pd.DataFrame:
    required = {"macro_layer", "time_pair", "closure_quality", "signal_status"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Seurat CG metrics lack {sorted(required - set(metrics.columns))}")
    data = metrics.copy()
    data["time_pair"] = data["time_pair"].map(normalize_time_pair)
    rows: list[dict[str, object]] = []
    for pair in time_pairs:
        row: dict[str, object] = {"Time pair": pair}
        for layer, label in (
            ("seurat_k150", "Spot -> K150"),
            ("seurat_k40", "Spot -> K40"),
            ("seurat_k10", "Spot -> K10"),
        ):
            selected = data.loc[data["macro_layer"].eq(layer) & data["time_pair"].eq(pair)]
            if len(selected) != 1:
                raise ValueError(f"Expected one Seurat CG row for {layer} {pair}; got {len(selected)}")
            item = selected.iloc[0]
            row[label] = _closure_quality_cell(item["closure_quality"], item["signal_status"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_closure_quality_grid(
    selected: pd.DataFrame,
    time_pairs: Sequence[str] = OPTIMAL_TIME_PAIRS,
) -> pd.DataFrame:
    required = {"input_scale", "time_pair", "closure_quality", "signal_status"}
    if not required.issubset(selected.columns):
        raise ValueError(f"ClosureQuality grid lacks {sorted(required - set(selected.columns))}")
    rows: list[dict[str, object]] = []
    for scale in INPUT_SCALES:
        row: dict[str, object] = {"Input scale": scale}
        for pair in time_pairs:
            hits = selected.loc[selected["input_scale"].eq(scale) & selected["time_pair"].eq(pair)]
            if len(hits) != 1:
                raise ValueError(f"Expected one closure run for {scale} {pair}; got {len(hits)}")
            item = hits.iloc[0]
            row[pair] = _closure_quality_cell(item["closure_quality"], item["signal_status"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_legacy_optimal_table(
    runs: pd.DataFrame,
    methods: Sequence[str],
    time_pairs: Sequence[str],
    time_points: Sequence[str],
    *,
    k: int,
    seed: int,
) -> pd.DataFrame:
    if len(time_points) != 3:
        raise ValueError("The legacy table needs three time points")
    t0, t1, t2 = time_points
    rows: list[dict[str, object]] = []
    for method in methods:
        try:
            selected = {
                pair: select_coarse_run(runs, method, "spot", pair, k=k, seed=seed)
                for pair in time_pairs
            }
        except KeyError:
            # Table 3 is allowed to render a draft placeholder while a newly
            # registered method (e.g. two-stage) is waiting for its matched-K run.
            rows.append({
                "Method": method,
                **{f"DeltaEI {pair}": np.nan for pair in time_pairs},
                f"K@{t0}": np.nan, f"K@{t1}": np.nan, f"K@{t2}": np.nan,
            })
            continue
        first = selected[f"{t0}->{t1}"]
        second = selected[f"{t1}->{t2}"]
        rows.append({
            "Method": method,
            **{f"DeltaEI {pair}": float(selected[pair]["delta_EI_paper"]) for pair in time_pairs},
            f"K@{t0}": int(first["hardK_t_paper"]),
            f"K@{t1}": int(second["hardK_t_paper"]),
            f"K@{t2}": int(second["hardK_tp_paper"]),
        })
    return pd.DataFrame(rows)


def select_optimal_input_runs(
    runs: pd.DataFrame,
    *,
    method: str,
    k_by_scale: Mapping[str, int],
    seed: int,
    time_pairs: Sequence[str] = OPTIMAL_TIME_PAIRS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scale in INPUT_SCALES:
        for pair in time_pairs:
            selected = select_coarse_run(
                runs, method, scale, pair, k=int(k_by_scale[scale]), seed=seed, strict=True
            )
            delta = float(selected["delta_EI_best_checkpoint"])
            micro = float(selected["EI_micro_fixed"])
            macro = float(selected["EI_macro_best_checkpoint"])
            if not np.isclose(macro - micro, delta, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Optimized DeltaEI disagrees with best-checkpoint EI: {selected['run_dir']}")
            rows.append({
                "input_scale": scale,
                "time_pair": pair,
                "K": int(selected["K"]),
                "seed": seed,
                "method": method,
                "run_dir": str(selected["run_dir"]),
                "EI_micro_fixed": micro,
                "EI_macro_best_checkpoint": macro,
                "delta_EI_best_checkpoint": delta,
            })
    return pd.DataFrame(rows)


def build_optimal_grid(
    selected: pd.DataFrame,
    value_column: str,
    time_pairs: Sequence[str] = OPTIMAL_TIME_PAIRS,
) -> pd.DataFrame:
    required = {"input_scale", "time_pair", value_column}
    if not required.issubset(selected.columns):
        raise ValueError(f"Optimal grid lacks {sorted(required - set(selected.columns))}")
    rows: list[dict[str, object]] = []
    for scale in INPUT_SCALES:
        row: dict[str, object] = {"Input scale": scale}
        for pair in time_pairs:
            hits = selected.loc[
                selected["input_scale"].eq(scale) & selected["time_pair"].eq(pair)
            ]
            if len(hits) != 1:
                raise ValueError(f"Expected one optimized run for {scale} {pair}; got {len(hits)}")
            row[pair] = hits.iloc[0][value_column]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "K10_LEVELS", "build_ei_ablation", "build_ei_hierarchy", "build_seurat_cg_hierarchy",
    "build_legacy_optimal_table", "select_optimal_input_runs", "build_optimal_grid",
    "build_closure_quality_grid",
]


FEATURE_ABLATION_ROW_ORDER = (
    ("CCI", "NMF", "KL"),
    ("CCI", "NMF", "KL + OT"),
    ("CCI", "Laplacian", "KL"),
    ("CCI", "Laplacian", "KL + OT"),
    ("GRN", "Dual-end gating", "KL"),
    ("GRN", "Dual-end gating", "KL + OT"),
    ("CCI + GRN", "NMF + dual-end gating", "KL"),
    ("CCI + GRN", "NMF + dual-end gating", "KL + OT"),
    ("CCI + GRN", "Laplacian + dual-end gating", "KL"),
    ("CCI + GRN", "Laplacian + dual-end gating", "KL + OT"),
)


def build_feature_ablation_table(
    metrics: pd.DataFrame,
    levels: Mapping[str, tuple[str, str]],
    time_pairs: Sequence[str],
) -> pd.DataFrame:
    """Build one of the two paper-facing controlled PIJ-ablation tables."""

    required = {
        "input_group", "feature_method", "pij_construction",
        "lower_layer", "upper_layer", "time_pair", "delta_EI",
    }
    if not required.issubset(metrics.columns):
        raise ValueError(f"Feature ablation lacks {sorted(required - set(metrics.columns))}")
    rows: list[dict[str, object]] = []
    for hierarchy, (lower, upper) in levels.items():
        block = metrics.loc[
            metrics["lower_layer"].eq(lower) & metrics["upper_layer"].eq(upper)
        ]
        for input_group, feature_method, pij_construction in FEATURE_ABLATION_ROW_ORDER:
            row: dict[str, object] = {
                "Hierarchy": hierarchy,
                "Input": input_group,
                "Feature method": feature_method,
                "PIJ construction": pij_construction,
            }
            for pair in time_pairs:
                hit = block.loc[
                    block["input_group"].eq(input_group)
                    & block["feature_method"].eq(feature_method)
                    & block["pij_construction"].eq(pij_construction)
                    & block["time_pair"].eq(pair)
                ]
                if len(hit) != 1:
                    raise ValueError(
                        "Expected one controlled ablation row for "
                        f"{hierarchy} / {input_group} / {feature_method} / "
                        f"{pij_construction} / {pair}; got {len(hit)}"
                    )
                row[pair] = float(hit.iloc[0]["delta_EI"])
            rows.append(row)
    return pd.DataFrame(rows)


__all__.extend(["FEATURE_ABLATION_ROW_ORDER", "build_feature_ablation_table"])
