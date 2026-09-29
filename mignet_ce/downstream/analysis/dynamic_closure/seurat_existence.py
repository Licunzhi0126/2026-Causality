from __future__ import annotations

"""Seurat coarse-graining closure evidence from exported native-unit PIJs."""

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..io import cci_index_path, domain_map_path, load_domain_map, read_index
from ..mappings import natural_assignment
from ..metrics import row_normalize
from .analysis import (
    EPS,
    crossfit_macro_q_error,
    effective_information,
    induced_macro_q,
    kl_rows,
    js_rows,
    normalize_assignment,
    weighted_information,
)
from .seurat_archive import (
    ArchivedTransition,
    file_descriptor,
    load_seurat_pair_archive,
)


SEURAT_LAYERS = ("seurat_k150", "seurat_k40", "seurat_k10")
EXPECTED_K = {"seurat_k150": 150, "seurat_k40": 40, "seurat_k10": 10}


@dataclass(frozen=True)
class SeuratClosureConfig:
    data_root: Path
    pij_archive_root: Path
    output_root: Path
    organ: str = "heart"
    times: tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5")
    micro_reference_layer: str = "seurat_k150"
    null_repeats: int = 999
    seed: int = 20260809
    crossfit_folds: int = 5
    network_method: str = "light_cci_grn"
    pij_method: str = "NG_KLot"

    def validate(self) -> None:
        if len(self.times) < 2 or len(set(self.times)) != len(self.times):
            raise ValueError("At least two unique ordered time points are required")
        if self.micro_reference_layer not in SEURAT_LAYERS:
            raise ValueError(f"Unknown micro reference layer {self.micro_reference_layer!r}")
        if self.null_repeats < 1 or self.crossfit_folds < 2:
            raise ValueError("null_repeats must be positive and crossfit_folds must be at least two")
        if self.network_method != "light_cci_grn" or self.pij_method != "NG_KLot":
            raise ValueError("The Seurat closure existence pipeline uses light_cci_grn / NG_KLot")
        if not Path(self.data_root).is_dir():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        if not Path(self.pij_archive_root).is_dir():
            raise FileNotFoundError(f"PIJ archive root does not exist: {self.pij_archive_root}")
        output = Path(self.output_root)
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Output root is not empty; use a new directory: {output}")


def _aligned_values(
    transition: ArchivedTransition,
    source_units: tuple[str, ...],
    target_units: tuple[str, ...],
) -> np.ndarray:
    if set(transition.source_units) != set(source_units) or set(transition.target_units) != set(target_units):
        raise ValueError(f"Spot unit sets differ from the reference PIJ in {transition.matrix_path}")
    source_index = {unit: index for index, unit in enumerate(transition.source_units)}
    target_index = {unit: index for index, unit in enumerate(transition.target_units)}
    return transition.values[
        np.ix_(
            [source_index[unit] for unit in source_units],
            [target_index[unit] for unit in target_units],
        )
    ]


def _assignment_in_archive_order(
    cfg: SeuratClosureConfig,
    layer: str,
    time: str,
    spot_units: tuple[str, ...],
    domain_units: tuple[str, ...],
) -> np.ndarray:
    if len(domain_units) != EXPECTED_K[layer]:
        raise ValueError(f"Expected {EXPECTED_K[layer]} {layer} domains at {time}; got {len(domain_units)}")
    expected_spots = read_index(cci_index_path(cfg.data_root, "spot", time, cfg.organ))
    if len(expected_spots) != len(spot_units) or set(expected_spots) != set(spot_units):
        raise ValueError(f"Spot PIJ units do not match the spot CCI index at {time}")
    frame = load_domain_map(cfg.data_root, layer, time, cfg.organ)
    if frame["spot_id"].duplicated().any() or set(frame["spot_id"]) != set(spot_units):
        raise ValueError(f"Domain map must contain each PIJ spot exactly once: {layer} {time}")
    assignment, _, cci_states = natural_assignment(
        cfg, layer, time, spot_order=spot_units
    )
    if set(cci_states) != set(domain_units):
        raise ValueError(f"Domain PIJ units do not match the domain CCI index: {layer} {time}")
    state_index = {state: index for index, state in enumerate(cci_states)}
    assignment = assignment[:, [state_index[unit] for unit in domain_units]]
    if np.any(assignment.sum(axis=0) == 0):
        raise ValueError(f"An archived {layer} domain has no mapped spots at {time}")
    return assignment


def _effective_states(assignment: np.ndarray) -> float:
    mass = assignment.mean(axis=0)
    return float(np.exp(-np.sum(mass * np.log(np.maximum(mass, EPS)))))


def _hard_information_closure_budget(
    p_micro: np.ndarray,
    source_assignment: np.ndarray,
    target_assignment: np.ndarray,
) -> dict[str, object]:
    """The shared uniform-spot budget with a two-dimensional hard-label KL."""

    p_matrix = row_normalize(p_micro)
    source = normalize_assignment(source_assignment)
    target = normalize_assignment(target_assignment)
    if p_matrix.shape != (source.shape[0], target.shape[0]):
        raise ValueError("Spot PIJ and Seurat assignment dimensions do not match")
    if not np.all((source == 0) | (source == 1)) or not np.all(source.sum(axis=1) == 1):
        raise ValueError("Seurat source assignment must be hard one-hot")
    observed = row_normalize(p_matrix @ target)
    weights = np.full(source.shape[0], 1.0 / source.shape[0])
    q_induced, macro_mass = induced_macro_q(observed, source, weights)
    predicted = row_normalize(source @ q_induced)
    available = weighted_information(observed, weights)
    retained = weighted_information(q_induced, macro_mass)
    leakage = float(np.mean(kl_rows(observed, q_induced[np.argmax(source, axis=1)])))
    identity_error = float(abs(available - retained - leakage))
    if identity_error > 1e-8:
        raise AssertionError(f"Seurat information-closure identity error {identity_error}")
    threshold = max(0.01, 0.01 * np.log2(max(target.shape[1], 2)))
    return {
        "observed": observed,
        "source_assignment": source,
        "target_assignment": target,
        "weights": weights,
        "q_induced": q_induced,
        "macro_mass": macro_mass,
        "predicted_induced": predicted,
        "I_available_bits": float(available),
        "I_macro_retained_bits": float(retained),
        "closure_leakage_bits": leakage,
        "closure_quality": float(retained / available) if available > EPS else np.nan,
        "information_identity_error": identity_error,
        "signal_status": "informative" if available >= threshold else "low-signal",
        "signal_threshold_bits": float(threshold),
        "weighting": "uniform_spot",
    }


def _retained_from_hard_labels(
    observed: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    target_mean: np.ndarray,
) -> float:
    """Fast one-hot equivalent of the shared information budget's retained term."""

    n_spots = len(labels)
    one_hot_t = sp.csr_matrix(
        (np.ones(n_spots), (labels, np.arange(n_spots))),
        shape=(len(counts), n_spots),
    )
    q_induced = (one_hot_t @ observed) / counts[:, None]
    mass = counts / n_spots
    terms = np.where(
        q_induced > EPS,
        q_induced * np.log2(np.maximum(q_induced, EPS) / np.maximum(target_mean[None, :], EPS)),
        0.0,
    )
    return float(np.sum(mass[:, None] * terms))


def _matched_source_partition_null(
    observed: np.ndarray,
    source_assignment: np.ndarray,
    available_bits: float,
    observed_quality: float,
    observed_retained_bits: float,
    *,
    repeats: int,
    seed: int,
    layer: str,
    pair: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not np.all((source_assignment == 0) | (source_assignment == 1)) or not np.all(source_assignment.sum(axis=1) == 1):
        raise ValueError("The Seurat matched null requires a hard one-hot source assignment")
    labels = np.argmax(source_assignment, axis=1)
    counts = np.bincount(labels, minlength=source_assignment.shape[1])
    if np.any(counts == 0):
        raise ValueError("The Seurat matched null cannot use empty source domains")
    target_mean = observed.mean(axis=0)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = [
        {
            "macro_layer": layer,
            "cg_pair": f"spot:{layer}",
            "time_pair": pair,
            "kind": "observed",
            "repeat": -1,
            "seed": seed,
            "I_retained_bits": observed_retained_bits,
            "closure_quality": observed_quality,
        }
    ]
    values = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        retained = _retained_from_hard_labels(observed, rng.permutation(labels), counts, target_mean)
        quality = retained / available_bits if available_bits > EPS else np.nan
        values[repeat] = quality
        rows.append(
            {
                "macro_layer": layer,
                "cg_pair": f"spot:{layer}",
                "time_pair": pair,
                "kind": "size_matched_source_shuffle",
                "repeat": repeat,
                "seed": seed,
                "I_retained_bits": retained,
                "closure_quality": quality,
            }
        )
    finite = values[np.isfinite(values)]
    valid = len(finite) == repeats and np.isfinite(observed_quality)
    summary = {
        "macro_layer": layer,
        "time_pair": pair,
        "null_repeats": repeats,
        "null_seed": seed,
        "null_mean_closure_quality": float(finite.mean()) if len(finite) else np.nan,
        "null_std_closure_quality": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
        "null_p95_closure_quality": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
        "effect_size_closure_quality": (
            float(observed_quality - finite.mean()) if valid else np.nan
        ),
        "p_empirical": (
            float((1 + np.count_nonzero(finite >= observed_quality)) / (1 + repeats))
            if valid else np.nan
        ),
    }
    return rows, summary


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return result
    order = valid[np.argsort(values[valid])]
    ranks = np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate((values[order] * len(order) / ranks)[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _metric_row(
    cfg: SeuratClosureConfig,
    pair: str,
    layer: str,
    p_micro: np.ndarray,
    source_assignment: np.ndarray,
    target_assignment: np.ndarray,
    q_direct: np.ndarray,
    *,
    pair_micro_ei_bits: float | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    source_time, target_time = pair.split("->")
    budget = _hard_information_closure_budget(p_micro, source_assignment, target_assignment)
    observed = budget["observed"]
    weights = budget["weights"]
    induced = budget["q_induced"]
    if q_direct.shape != induced.shape:
        raise ValueError(f"Direct and induced Q differ in shape for {layer} {pair}")
    direct_prediction = source_assignment @ q_direct
    induced_prediction = budget["predicted_induced"]
    crossfit = crossfit_macro_q_error(
        observed,
        source_assignment,
        folds=cfg.crossfit_folds,
        weights=weights,
        seed=cfg.seed,
    )
    source_counts = source_assignment.sum(axis=0)
    target_counts = target_assignment.sum(axis=0)
    row = {
        "organ": cfg.organ,
        "network_method": cfg.network_method,
        "pij_method": cfg.pij_method,
        "micro_layer": "spot",
        "macro_layer": layer,
        "cg_pair": f"spot:{layer}",
        "mapping": f"Seurat K{EXPECTED_K[layer]}",
        "time_pair": pair,
        "source_time": source_time,
        "target_time": target_time,
        "micro_reference_pair": f"spot:{cfg.micro_reference_layer}",
        "n_source_spots": p_micro.shape[0],
        "n_target_spots": p_micro.shape[1],
        "source_states": source_assignment.shape[1],
        "target_states": target_assignment.shape[1],
        "Keff_source": _effective_states(source_assignment),
        "Keff_target": _effective_states(target_assignment),
        "min_source_domain_spots": int(source_counts.min()),
        "min_target_domain_spots": int(target_counts.min()),
        "EI_micro_spot_bits": effective_information(p_micro),
        "EI_macro_direct_bits": effective_information(q_direct),
        "EI_macro_induced_bits": effective_information(induced),
        "I_available_bits": budget["I_available_bits"],
        "I_retained_bits": budget["I_macro_retained_bits"],
        "closure_leakage_bits": budget["closure_leakage_bits"],
        "closure_quality": budget["closure_quality"],
        "signal_status": budget["signal_status"],
        "signal_threshold_bits": budget["signal_threshold_bits"],
        "information_identity_error": budget["information_identity_error"],
        "direct_induced_q_js": float(np.sum(budget["macro_mass"] * js_rows(q_direct, induced))),
        "closure_js_direct": float(np.sum(weights * js_rows(observed, direct_prediction))),
        "closure_js_induced": float(np.sum(weights * js_rows(observed, induced_prediction))),
        **crossfit,
    }
    row["delta_EI_direct_vs_spot_bits"] = row["EI_macro_direct_bits"] - row["EI_micro_spot_bits"]
    row["delta_EI_induced_vs_spot_bits"] = row["EI_macro_induced_bits"] - row["EI_micro_spot_bits"]
    row["EI_micro_pair_spot_bits"] = (
        row["EI_micro_spot_bits"] if pair_micro_ei_bits is None else float(pair_micro_ei_bits)
    )
    row["delta_EI_vertical_pair_bits"] = row["EI_macro_direct_bits"] - row["EI_micro_pair_spot_bits"]
    return row, budget


def build_seurat_closure_tables(
    cfg: SeuratClosureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Compute six time pairs for each of the three independent Seurat partitions."""

    cfg.validate()
    archives = {
        layer: load_seurat_pair_archive(
            cfg.pij_archive_root,
            organ=cfg.organ,
            macro_layer=layer,
            network_method=cfg.network_method,
            pij_method=cfg.pij_method,
            times=cfg.times,
        )
        for layer in SEURAT_LAYERS
    }
    metrics: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    null_summaries: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    input_paths: set[Path] = {archive.directory / "kernel_metadata.json" for archive in archives.values()}
    for pair_index, (source_time, target_time) in enumerate(combinations(cfg.times, 2)):
        pair = f"{source_time}->{target_time}"
        reference = archives[cfg.micro_reference_layer].transition(source_time, target_time, "lower")
        input_paths.update((reference.matrix_path, reference.source_units_path, reference.target_units_path))
        p_micro = reference.values
        for layer_index, layer in enumerate(SEURAT_LAYERS):
            archive = archives[layer]
            direct = archive.transition(source_time, target_time, "upper")
            input_paths.update((direct.matrix_path, direct.source_units_path, direct.target_units_path))
            if layer == cfg.micro_reference_layer:
                spot_pij_max_abs_diff = 0.0
                pair_micro_ei = None
            else:
                other_spot = archive.transition(source_time, target_time, "lower")
                input_paths.update((other_spot.matrix_path, other_spot.source_units_path, other_spot.target_units_path))
                aligned = _aligned_values(other_spot, reference.source_units, reference.target_units)
                spot_pij_max_abs_diff = float(np.max(np.abs(aligned - p_micro)))
                pair_micro_ei = effective_information(aligned)
            source_assignment = _assignment_in_archive_order(
                cfg, layer, source_time, reference.source_units, direct.source_units
            )
            target_assignment = _assignment_in_archive_order(
                cfg, layer, target_time, reference.target_units, direct.target_units
            )
            for time in (source_time, target_time):
                input_paths.add(domain_map_path(cfg.data_root, layer, time, cfg.organ))
                input_paths.add(cci_index_path(cfg.data_root, layer, time, cfg.organ))
                input_paths.add(cci_index_path(cfg.data_root, "spot", time, cfg.organ))
            row, budget = _metric_row(
                cfg, pair, layer, p_micro, source_assignment, target_assignment, direct.values,
                pair_micro_ei_bits=pair_micro_ei,
            )
            row["spot_pij_max_abs_diff_vs_reference"] = spot_pij_max_abs_diff
            metrics.append(row)
            null_seed = cfg.seed + pair_index * len(SEURAT_LAYERS) + layer_index
            samples, summary = _matched_source_partition_null(
                budget["observed"],
                source_assignment,
                float(budget["I_available_bits"]),
                float(budget["closure_quality"]),
                float(budget["I_macro_retained_bits"]),
                repeats=cfg.null_repeats,
                seed=null_seed,
                layer=layer,
                pair=pair,
            )
            null_rows.extend(samples)
            null_summaries.append(summary)
            audit_rows.append(
                {
                    "macro_layer": layer,
                    "time_pair": pair,
                    "reference_spot_pij": str(reference.matrix_path.resolve()),
                    "comparison_spot_pij": str(
                        (archive.directory / f"{source_time}_to_{target_time}_lower_P.npz").resolve()
                    ),
                    "spot_pij_max_abs_diff_vs_reference": spot_pij_max_abs_diff,
                    "direct_macro_pij": str(direct.matrix_path.resolve()),
                    "source_domain_map": str(domain_map_path(cfg.data_root, layer, source_time, cfg.organ).resolve()),
                    "target_domain_map": str(domain_map_path(cfg.data_root, layer, target_time, cfg.organ).resolve()),
                    "spot_unit_order_checked": True,
                    "domain_unit_order_checked": True,
                }
            )
    metric_frame = pd.DataFrame(metrics)
    null_frame = pd.DataFrame(null_rows)
    summary_frame = pd.DataFrame(null_summaries)
    summary_frame["p_fdr_bh"] = _bh_adjust(summary_frame["p_empirical"].to_numpy(dtype=float))
    metric_frame = metric_frame.merge(
        summary_frame,
        on=["macro_layer", "time_pair"],
        how="left",
        validate="one_to_one",
    )
    informative = metric_frame["signal_status"].eq("informative")
    supported = (
        informative
        & metric_frame["effect_size_closure_quality"].gt(0)
        & metric_frame["p_fdr_bh"].lt(0.05)
    )
    metric_frame["closure_quality_for_claim"] = metric_frame["closure_quality"].where(informative)
    metric_frame["evidence_vs_size_matched_null"] = supported
    metric_frame["claim_status"] = np.where(
        ~informative,
        "low_signal",
        np.where(supported, "supported_vs_matched_null", "not_supported"),
    )
    return metric_frame, null_frame, summary_frame, pd.DataFrame(audit_rows), [
        file_descriptor(path) for path in sorted(input_paths)
    ]


def run_seurat_closure_existence(cfg: SeuratClosureConfig) -> dict[str, Path]:
    metrics, null_distribution, null_summary, input_audit, inputs = build_seurat_closure_tables(cfg)
    output_root = Path(cfg.output_root)
    tables_dir = output_root / "tables"
    audit_dir = output_root / "audit"
    tables_dir.mkdir(parents=True, exist_ok=False)
    audit_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "closure_metrics": tables_dir / "seurat_closure_metrics.csv",
        "null_distribution": tables_dir / "seurat_closure_matched_null_distribution.csv",
        "null_summary": tables_dir / "seurat_closure_matched_null_summary.csv",
        "input_audit": audit_dir / "seurat_closure_input_audit.csv",
        "manifest": audit_dir / "seurat_closure_manifest.json",
    }
    metrics.to_csv(paths["closure_metrics"], index=False)
    null_distribution.to_csv(paths["null_distribution"], index=False)
    null_summary.to_csv(paths["null_summary"], index=False)
    input_audit.to_csv(paths["input_audit"], index=False)
    manifest = {
        "workflow": "seurat_closure_existence_v1",
        "organ": cfg.organ,
        "network_method": cfg.network_method,
        "pij_method": cfg.pij_method,
        "times": list(cfg.times),
        "macro_layers": list(SEURAT_LAYERS),
        "micro_reference_pair": f"spot:{cfg.micro_reference_layer}",
        "spot_weighting": "uniform_spot",
        "null_model": "shuffle_source_domain_labels_preserving_domain_sizes_and_fixed_target_map",
        "null_repeats": cfg.null_repeats,
        "seed": cfg.seed,
        "crossfit_folds": cfg.crossfit_folds,
        "fdr_family": "all_finite_layer_time_pair_tests",
        "signal_threshold": "max(0.01, 0.01 * log2(max(K_target, 2))) bits",
        "closure_claim_rule": "informative and positive CQ effect and BH-adjusted p < 0.05",
        "row_count": len(metrics),
        "expected_row_count": len(SEURAT_LAYERS) * len(tuple(combinations(cfg.times, 2))),
        "inputs": inputs,
        "outputs": {name: str(path.resolve()) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
