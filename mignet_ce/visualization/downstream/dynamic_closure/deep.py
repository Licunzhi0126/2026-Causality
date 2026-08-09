from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .analysis import (
    EPS,
    best_macro_q,
    effective_information,
    entropy_rows,
    js_rows,
    kl_rows,
    load_domain_assignment,
    normalize_assignment,
    read_h5ad_units_coords,
    row_normalize,
    weighted_information,
)
from .config import DEFAULT_TIME_POINTS
from .deep_plots import render_deep_figures
from .io import write_manifest
from .workflow import _layer_paths, _load_index

TIME_POINTS = DEFAULT_TIME_POINTS
TIME_PAIRS = tuple(f"{a}->{b}" for a, b in zip(TIME_POINTS[:-1], TIME_POINTS[1:]))
MAPPINGS = ("Seurat K150", "Seurat K40", "Optimized coarse-graining")


@dataclass
class MappingRecord:
    mapping: str
    time_pair: str
    source_time: str
    target_time: str
    p_micro: np.ndarray
    source_assignment: np.ndarray
    target_assignment: np.ndarray
    q_best: np.ndarray
    q_direct: np.ndarray
    source_names: list[str]

    @property
    def observed(self) -> np.ndarray:
        return row_normalize(self.p_micro @ normalize_assignment(self.target_assignment))


def load_npz_matrix(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        matrix = sp.load_npz(path)
        return np.asarray(matrix.toarray(), dtype=np.float64)
    except Exception:
        archive = np.load(path)
        if "pij" in archive.files:
            return np.asarray(archive["pij"], dtype=np.float64)
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]], dtype=np.float64)
        raise ValueError(f"Could not infer matrix key in {path}; keys={archive.files}")


def _spot_assignments(
    data_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIME_POINTS,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], list[str]]]:
    assignments: dict[tuple[str, str], np.ndarray] = {}
    units: dict[tuple[str, str], list[str]] = {}
    for time in time_points:
        spot_units = _load_index(_layer_paths(_ConfigProxy(data_root, organ), "spot", time)["index"])
        units[("spot", time)] = spot_units
        for layer in ("seurat_k150", "seurat_k40"):
            state_units = _load_index(_layer_paths(_ConfigProxy(data_root, organ), layer, time)["index"])
            units[(layer, time)] = state_units
            assignments[(layer, time)] = load_domain_assignment(
                _layer_paths(_ConfigProxy(data_root, organ), layer, time)["map"],
                spot_units,
                state_units,
            )
    return assignments, units


class _ConfigProxy:
    def __init__(self, data_root: Path, organ: str):
        self.data_root = Path(data_root)
        self.organ = organ


def load_mapping_records(
    data_root: Path,
    closure_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIME_POINTS,
) -> list[MappingRecord]:
    data_root = Path(data_root)
    closure_root = Path(closure_root)
    assignments, units = _spot_assignments(data_root, organ, time_points)
    records: list[MappingRecord] = []
    for source, target in zip(time_points[:-1], time_points[1:]):
        pair = f"{source}->{target}"
        p = row_normalize(load_npz_matrix(closure_root / "matrices" / f"pij_spot_{source}_to_{target}.npz"))
        for layer, mapping in (("seurat_k150", "Seurat K150"), ("seurat_k40", "Seurat K40")):
            q_best = row_normalize(np.load(closure_root / "tables" / f"q_best_{layer}_{source}_to_{target}.npy"))
            q_direct = row_normalize(np.load(closure_root / "tables" / f"q_direct_{layer}_{source}_to_{target}.npy"))
            records.append(
                MappingRecord(
                    mapping=mapping,
                    time_pair=pair,
                    source_time=source,
                    target_time=target,
                    p_micro=p,
                    source_assignment=assignments[(layer, source)],
                    target_assignment=assignments[(layer, target)],
                    q_best=q_best,
                    q_direct=q_direct,
                    source_names=units[(layer, source)],
                )
            )
        opt = closure_root / "optimal_coarse" / f"{source}_to_{target}"
        p_opt = row_normalize(load_npz_matrix(opt / "PIJ_micro.npz"))
        s_t = normalize_assignment(np.load(opt / "S_t.npy"))
        s_tp = normalize_assignment(np.load(opt / "S_tp.npy"))
        q_best = row_normalize(np.load(closure_root / "tables" / f"q_best_optimized_{source}_to_{target}.npy"))
        q_direct = row_normalize(np.load(closure_root / "tables" / f"q_direct_optimized_{source}_to_{target}.npy"))
        records.append(
            MappingRecord(
                mapping="Optimized coarse-graining",
                time_pair=pair,
                source_time=source,
                target_time=target,
                p_micro=p_opt,
                source_assignment=s_t,
                target_assignment=s_tp,
                q_best=q_best,
                q_direct=q_direct,
                source_names=[f"M{i:02d}" for i in range(s_t.shape[1])],
            )
        )
    return records


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if weights.sum() <= EPS:
        return float(np.quantile(values, quantile))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(quantile, cumulative, values))


def weighted_best_q(observed: np.ndarray, assignment: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = row_normalize(observed)
    assignment = normalize_assignment(assignment)
    weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    weights = weights / max(weights.sum(), EPS)
    weighted_assignment = assignment * weights[:, None]
    macro_mass = weighted_assignment.sum(axis=0)
    q = weighted_assignment.T @ observed
    q = np.divide(q, macro_mass[:, None], out=np.zeros_like(q), where=macro_mass[:, None] > EPS)
    return row_normalize(q), macro_mass


def source_weight_schemes(assignment: np.ndarray) -> dict[str, np.ndarray]:
    assignment = normalize_assignment(assignment)
    n = assignment.shape[0]
    uniform_spot = np.full(n, 1.0 / n, dtype=float)
    mass = assignment.sum(axis=0)
    active = mass > EPS
    inv_mass = np.zeros_like(mass)
    inv_mass[active] = 1.0 / mass[active]
    state_balanced = assignment @ inv_mass
    state_balanced /= max(float(state_balanced.sum()), EPS)
    confidence = np.max(assignment, axis=1)
    confidence_weighted = confidence / max(float(confidence.sum()), EPS)
    return {
        "Uniform spot": uniform_spot,
        "State balanced": state_balanced,
        "Confidence weighted": confidence_weighted,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    weights /= max(float(weights.sum()), EPS)
    return float(np.sum(values * weights))


def strong_lumpability_tables(records: Iterable[MappingRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for record in records:
        observed = record.observed
        assignment = normalize_assignment(record.source_assignment)
        hard = np.argmax(assignment, axis=1)
        predicted_best = row_normalize(assignment @ record.q_best)
        residual_js = js_rows(observed, predicted_best)
        residual_kl = kl_rows(observed, predicted_best)
        residual_tv = 0.5 * np.sum(np.abs(observed - predicted_best), axis=1)
        total_kl = float(np.sum(residual_kl))
        for state_index in range(assignment.shape[1]):
            weights = assignment[:, state_index]
            member = weights > 1e-8
            if not np.any(member):
                continue
            local_weights = weights[member]
            local_weights /= max(float(local_weights.sum()), EPS)
            q_disagreement = float(js_rows(record.q_best[state_index:state_index+1], record.q_direct[state_index:state_index+1])[0])
            tv_disagreement = float(0.5 * np.sum(np.abs(record.q_best[state_index] - record.q_direct[state_index])))
            contribution = float(np.sum(weights * residual_kl) / max(total_kl, EPS))
            state_rows.append({
                "mapping": record.mapping,
                "time_pair": record.time_pair,
                "state_index": state_index,
                "state": record.source_names[state_index] if state_index < len(record.source_names) else f"S{state_index}",
                "mass": float(weights.sum()),
                "hard_mass": int(np.sum(hard == state_index)),
                "mean_js": _weighted_mean(residual_js[member], local_weights),
                "p95_js": weighted_quantile(residual_js[member], 0.95, local_weights),
                "max_js": float(np.max(residual_js[member])),
                "mean_tv": _weighted_mean(residual_tv[member], local_weights),
                "p95_tv": weighted_quantile(residual_tv[member], 0.95, local_weights),
                "max_tv": float(np.max(residual_tv[member])),
                "mean_kl": _weighted_mean(residual_kl[member], local_weights),
                "p95_kl": weighted_quantile(residual_kl[member], 0.95, local_weights),
                "max_kl": float(np.max(residual_kl[member])),
                "residual_kl_share": contribution,
                "q_best_direct_js": q_disagreement,
                "q_best_direct_tv": tv_disagreement,
                "q_best_entropy": float(entropy_rows(record.q_best[state_index:state_index+1])[0]),
                "q_direct_entropy": float(entropy_rows(record.q_direct[state_index:state_index+1])[0]),
            })
        frame = pd.DataFrame([row for row in state_rows if row["mapping"] == record.mapping and row["time_pair"] == record.time_pair])
        summary_rows.append({
            "mapping": record.mapping,
            "time_pair": record.time_pair,
            "mean_js": float(np.mean(residual_js)),
            "p95_js": float(np.quantile(residual_js, 0.95)),
            "max_js": float(np.max(residual_js)),
            "mean_tv": float(np.mean(residual_tv)),
            "p95_tv": float(np.quantile(residual_tv, 0.95)),
            "max_tv": float(np.max(residual_tv)),
            "top10_state_residual_share": float(frame.nlargest(min(10, len(frame)), "residual_kl_share")["residual_kl_share"].sum()) if len(frame) else np.nan,
            "active_states": int((assignment.sum(axis=0) > EPS).sum()),
            "mean_q_best_direct_js": float(np.average(frame["q_best_direct_js"], weights=np.maximum(frame["mass"], EPS))) if len(frame) else np.nan,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(state_rows)


def intervention_weighting_table(records: Iterable[MappingRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for record in records:
        observed = record.observed
        assignment = normalize_assignment(record.source_assignment)
        for scheme, weights in source_weight_schemes(assignment).items():
            q_weighted, macro_mass = weighted_best_q(observed, assignment, weights)
            predicted = row_normalize(assignment @ q_weighted)
            predicted_direct = row_normalize(assignment @ record.q_direct)
            micro_info = weighted_information(observed, weights)
            macro_info = weighted_information(q_weighted, macro_mass)
            residual_info = max(0.0, micro_info - macro_info)
            rows.append({
                "mapping": record.mapping,
                "time_pair": record.time_pair,
                "weighting": scheme,
                "micro_information": micro_info,
                "macro_information": macro_info,
                "residual_information": residual_info,
                "macro_sufficiency": macro_info / max(micro_info, EPS),
                "best_mean_js": _weighted_mean(js_rows(observed, predicted), weights),
                "best_mean_kl": _weighted_mean(kl_rows(observed, predicted), weights),
                "direct_mean_js": _weighted_mean(js_rows(observed, predicted_direct), weights),
                "direct_excess_js": _weighted_mean(js_rows(observed, predicted_direct), weights) - _weighted_mean(js_rows(observed, predicted), weights),
                "effective_macro_states": float(np.exp(-np.sum((macro_mass[macro_mass > EPS] / macro_mass.sum()) * np.log(macro_mass[macro_mass > EPS] / macro_mass.sum())))),
            })
    return pd.DataFrame(rows)

def _write_json(path: Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def run_deep_closure_analysis(
    *,
    data_root: Path,
    closure_root: Path,
    output_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIME_POINTS,
    bootstrap_repeats: int = 200,
    seed: int = 20260804,
) -> dict[str, Any]:
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_mapping_records(data_root, closure_root, organ, time_points)
    lump_summary, lump_states = strong_lumpability_tables(records)
    weighting = intervention_weighting_table(records)
    horizon = horizon_closure_table(records, closure_root, time_points)
    direction = directional_closure_table(records)
    spectrum_summary, spectrum = q_spectrum_table(records)
    bootstrap = stratified_bootstrap_table(records, repeats=bootstrap_repeats, seed=seed)
    boot_summary = bootstrap_summary(bootstrap)

    tables = {
        "closure_lumpability_summary.csv": lump_summary,
        "closure_lumpability_states.csv": lump_states,
        "closure_intervention_weighting.csv": weighting,
        "closure_horizon_memory.csv": horizon,
        "closure_directionality.csv": direction,
        "closure_q_spectrum_summary.csv": spectrum_summary,
        "closure_q_spectrum_values.csv": spectrum,
        "closure_bootstrap_samples.csv": bootstrap,
        "closure_bootstrap_summary.csv": boot_summary,
    }
    for name, frame in tables.items():
        frame.to_csv(tables_dir / name, index=False)

    figures = render_deep_figures(
        lump_summary=lump_summary,
        lump_states=lump_states,
        weighting=weighting,
        horizon=horizon,
        direction=direction,
        spectrum_summary=spectrum_summary,
        spectrum=spectrum,
        bootstrap_summary=boot_summary,
        output_dir=figures_dir,
    )
    findings = {
        "scope": "Deep dynamical-closure audit: strong/weak lumpability, intervention weighting, horizon memory, forward/backward closure, Q-spectrum mismatch, and stratified uncertainty.",
        "records": len(records),
        "bootstrap_repeats": bootstrap_repeats,
        "figures": [str(path) for path in figures],
        "tables": list(tables),
    }
    _write_json(output_root / "findings.json", findings)
    manifest = write_manifest(
        output_root,
        {
            "data_root": str(Path(data_root).resolve()),
            "closure_root": str(Path(closure_root).resolve()),
            "organ": organ,
            "time_points": list(time_points),
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
        },
    )
    return {
        "output_root": output_root,
        "figures": figures,
        "tables": tables,
        "findings": findings,
        "manifest": manifest,
    }



def reverse_transition(p: np.ndarray, source_prior: np.ndarray | None = None) -> np.ndarray:
    p = row_normalize(p)
    if source_prior is None:
        source_prior = np.full(p.shape[0], 1.0 / p.shape[0])
    prior = np.maximum(np.asarray(source_prior, dtype=float), 0.0)
    prior /= max(float(prior.sum()), EPS)
    joint = prior[:, None] * p
    target_mass = joint.sum(axis=0)
    reverse = np.divide(joint.T, target_mass[:, None], out=np.zeros((p.shape[1], p.shape[0]), dtype=float), where=target_mass[:, None] > EPS)
    return row_normalize(reverse)


def directional_closure_table(records: Iterable[MappingRecord]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for record in records:
        for direction in ("Forward", "Backward"):
            if direction == "Forward":
                p = record.p_micro
                s_source = record.source_assignment
                s_target = record.target_assignment
            else:
                p = reverse_transition(record.p_micro)
                s_source = record.target_assignment
                s_target = record.source_assignment
            observed = row_normalize(p @ normalize_assignment(s_target))
            q, mass = best_macro_q(observed, s_source)
            predicted = row_normalize(normalize_assignment(s_source) @ q)
            micro_info = effective_information(observed)
            macro_weights = mass / max(float(mass.sum()), EPS)
            macro_info = weighted_information(q, macro_weights)
            rows.append({
                "mapping": record.mapping,
                "time_pair": record.time_pair,
                "direction": direction,
                "micro_information": micro_info,
                "macro_information": macro_info,
                "macro_sufficiency": macro_info / max(micro_info, EPS),
                "residual_fraction": max(0.0, micro_info - macro_info) / max(micro_info, EPS),
                "mean_js": float(np.mean(js_rows(observed, predicted))),
                "p95_js": float(np.quantile(js_rows(observed, predicted), 0.95)),
                "relative_frobenius": float(np.linalg.norm(observed - predicted) / max(np.linalg.norm(observed), EPS)),
            })
    frame = pd.DataFrame(rows)
    forward = frame[frame["direction"] == "Forward"].set_index(["mapping", "time_pair"])
    backward = frame[frame["direction"] == "Backward"].set_index(["mapping", "time_pair"])
    asym = forward[["macro_sufficiency", "mean_js", "micro_information"]].join(
        backward[["macro_sufficiency", "mean_js", "micro_information"]], lsuffix="_forward", rsuffix="_backward"
    ).reset_index()
    asym["sufficiency_asymmetry"] = asym["macro_sufficiency_forward"] - asym["macro_sufficiency_backward"]
    asym["closure_js_asymmetry"] = asym["mean_js_backward"] - asym["mean_js_forward"]
    frame = frame.merge(asym[["mapping", "time_pair", "sufficiency_asymmetry", "closure_js_asymmetry"]], on=["mapping", "time_pair"], how="left")
    return frame


def _compose_to_target(
    p_by_pair: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
    target_assignment: np.ndarray,
    time_points: tuple[str, ...] = TIME_POINTS,
) -> np.ndarray:
    representation = normalize_assignment(target_assignment)
    for index in range(end_index - 1, start_index - 1, -1):
        pair = f"{time_points[index]}->{time_points[index+1]}"
        representation = row_normalize(p_by_pair[pair] @ representation)
    return representation


def horizon_closure_table(
    records: Iterable[MappingRecord],
    closure_root: Path,
    time_points: tuple[str, ...] = TIME_POINTS,
) -> pd.DataFrame:
    records = list(records)
    by_key = {(r.mapping, r.time_pair): r for r in records}
    rows: list[dict[str, float | str]] = []
    for mapping in MAPPINGS:
        time_pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
        p_by_pair = {pair: by_key[(mapping, pair)].p_micro for pair in time_pairs}
        for start_index in range(len(time_points) - 1):
            for end_index in range(start_index + 1, len(time_points)):
                start = time_points[start_index]
                end = time_points[end_index]
                first_pair = f"{start}->{time_points[start_index+1]}"
                last_pair = f"{time_points[end_index-1]}->{end}"
                source_assignment = by_key[(mapping, first_pair)].source_assignment
                target_assignment = by_key[(mapping, last_pair)].target_assignment
                observed = _compose_to_target(
                    p_by_pair,
                    start_index,
                    end_index,
                    target_assignment,
                    time_points,
                )
                q_best, mass = best_macro_q(observed, source_assignment)
                predicted = row_normalize(normalize_assignment(source_assignment) @ q_best)
                micro_info = effective_information(observed)
                macro_weights = mass / max(float(mass.sum()), EPS)
                macro_info = weighted_information(q_best, macro_weights)
                rows.append({
                    "mapping": mapping,
                    "interval": f"{start}->{end}",
                    "start_time": start,
                    "end_time": end,
                    "horizon_steps": end_index - start_index,
                    "micro_information": micro_info,
                    "macro_information": macro_info,
                    "macro_sufficiency": macro_info / max(micro_info, EPS),
                    "residual_information": max(0.0, micro_info - macro_info),
                    "residual_fraction": max(0.0, micro_info - macro_info) / max(micro_info, EPS),
                    "best_mean_js": float(np.mean(js_rows(observed, predicted))),
                    "best_p95_js": float(np.quantile(js_rows(observed, predicted), 0.95)),
                    "best_q_ei": effective_information(q_best),
                })
    horizon = pd.DataFrame(rows)
    multi_path = Path(closure_root) / "tables" / "closure_multistep_summary.csv"
    if multi_path.exists():
        multi = pd.read_csv(multi_path).rename(columns={"interval": "interval"})
        horizon = horizon.merge(
            multi[["mapping", "interval", "composed_mean_js", "best_possible_mean_js", "semigroup_excess_js", "EI_composed_Q", "EI_best_Q"]],
            on=["mapping", "interval"], how="left"
        )
    return horizon


def q_spectrum_table(records: Iterable[MappingRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | str]] = []
    spectrum_rows: list[dict[str, float | str]] = []
    for record in records:
        singulars: dict[str, np.ndarray] = {}
        for q_type, matrix in (("Micro-induced Q", record.q_best), ("Independent Q", record.q_direct)):
            s = np.linalg.svd(matrix, compute_uv=False)
            s = np.maximum(s, 0.0)
            probability = s / max(float(s.sum()), EPS)
            positive = probability[probability > EPS]
            effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
            singulars[q_type] = s
            for rank, value in enumerate(s[: min(40, len(s))], start=1):
                spectrum_rows.append({
                    "mapping": record.mapping,
                    "time_pair": record.time_pair,
                    "q_type": q_type,
                    "rank": rank,
                    "singular_value": float(value),
                    "normalized_singular_value": float(probability[rank - 1]),
                })
            summary_rows.append({
                "mapping": record.mapping,
                "time_pair": record.time_pair,
                "q_type": q_type,
                "effective_rank": effective_rank,
                "top1_fraction": float(probability[0]),
                "top5_fraction": float(probability[:5].sum()),
                "spectral_entropy_bits": float(-np.sum(positive * np.log2(positive))),
            })
        a = singulars["Micro-induced Q"]
        b = singulars["Independent Q"]
        length = min(len(a), len(b))
        cosine = float(np.dot(a[:length], b[:length]) / max(np.linalg.norm(a[:length]) * np.linalg.norm(b[:length]), EPS))
        summary_rows.append({
            "mapping": record.mapping,
            "time_pair": record.time_pair,
            "q_type": "Comparison",
            "effective_rank": np.nan,
            "top1_fraction": np.nan,
            "top5_fraction": np.nan,
            "spectral_entropy_bits": np.nan,
            "spectrum_cosine": cosine,
            "relative_q_frobenius": float(np.linalg.norm(record.q_direct - record.q_best) / max(np.linalg.norm(record.q_best), EPS)),
            "mean_row_js": float(np.mean(js_rows(record.q_best, record.q_direct))),
            "p95_row_js": float(np.quantile(js_rows(record.q_best, record.q_direct), 0.95)),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(spectrum_rows)


def stratified_bootstrap_table(records: Iterable[MappingRecord], repeats: int = 200, seed: int = 20260804) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []
    for record in records:
        observed = record.observed
        assignment = normalize_assignment(record.source_assignment)
        hard = np.argmax(assignment, axis=1)
        groups = [np.where(hard == state)[0] for state in np.unique(hard)]
        for repeat in range(repeats):
            sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups if len(group)])
            obs = observed[sampled]
            assn = assignment[sampled]
            q, mass = best_macro_q(obs, assn)
            predicted = row_normalize(assn @ q)
            micro_info = effective_information(obs)
            macro_info = weighted_information(q, mass / max(float(mass.sum()), EPS))
            rows.append({
                "mapping": record.mapping,
                "time_pair": record.time_pair,
                "repeat": repeat,
                "macro_sufficiency": macro_info / max(micro_info, EPS),
                "mean_js": float(np.mean(js_rows(obs, predicted))),
                "residual_information": max(0.0, micro_info - macro_info),
            })
    return pd.DataFrame(rows)


def bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (mapping, pair), frame in bootstrap.groupby(["mapping", "time_pair"]):
        for metric in ("macro_sufficiency", "mean_js", "residual_information"):
            values = frame[metric].to_numpy(float)
            rows.append({
                "mapping": mapping,
                "time_pair": pair,
                "metric": metric,
                "mean": float(np.mean(values)),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "std": float(np.std(values, ddof=1)),
            })
    return pd.DataFrame(rows)
