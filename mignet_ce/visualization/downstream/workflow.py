from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    build_cci_metrics,
    build_ei_tables,
    build_fate_paths,
    build_grn_metrics,
    build_hierarchy_tables,
    build_mechanism_table,
    build_multiscale_consistency,
    build_multistep_closure,
    build_perturbation_curves,
    build_random_null,
    build_single_step_closure,
    build_spatial_metrics,
    build_spatial_state_maps,
    summarize_findings,
)
from .config import DownstreamConfig
from .io import load_domain_map, load_filtered_metrics, validate_pair_archive
from .plots import (
    plot_effective_spatial,
    plot_ei_overview,
    plot_fate_paths,
    plot_mechanism,
    plot_multiscale,
    plot_multistep_closure,
    plot_perturbation,
    plot_random_null,
    plot_single_step_closure,
    plot_spatial_state_ei,
)
from .determinism_degeneracy.analysis import build_unified_ei_tables
from .determinism_degeneracy.plots import plot_unified_ei_overview
from .dynamic_closure.analysis import (
    build_cross_representation_consistency,
    build_unified_closure_table,
)
from .dynamic_closure.plots import (
    plot_cross_representation_consistency,
    plot_dynamical_closure_three_panels,
)
from .fate_path.analysis import build_unified_fate_paths
from .fate_path.plots import plot_unified_fate_paths
from .grn_cci.analysis import build_unified_mechanism_table
from .grn_cci.plots import plot_unified_mechanism
from .mappings import load_all_mapping_records
from .null_model.analysis import build_unified_matched_null
from .null_model.plots import plot_unified_random_null
from .perturbation.analysis import build_unified_perturbation_curves
from .perturbation.plots import plot_unified_perturbation
from .preparation import ensure_full_unified_inputs
from .reporting import audit_unified_outputs, unified_run_manifest
from .spatial.analysis import (
    build_unified_effective_states,
    build_unified_spatial_metrics,
    build_unified_spatial_spots,
)
from .spatial.plots import (
    plot_unified_effective_spatial,
    plot_unified_spatial_state_ei,
)


TABLE_FILES = {
    "metrics": "01_metrics_input.csv",
    "decomposition": "02_ei_decomposition.csv",
    "state_table": "03_state_level_ei.csv",
    "multistep": "04_multistep_closure.csv",
    "single_step": "04a_single_step_closure.csv",
    "purity": "05_k150_k40_overlap_purity.csv",
    "effective": "06_effective_states.csv",
    "consistency": "07_multiscale_consistency.csv",
    "random_null": "08_matched_random_null.csv",
    "spatial": "09_spatial_state_metrics.csv",
    "grn": "10_grn_state_fingerprints.csv",
    "cci": "11_cci_state_roles.csv",
    "mechanism": "12_mechanism_merged.csv",
    "correlations": "13_mechanism_correlations.csv",
    "coefficients": "14_mechanism_regression.csv",
    "fate": "15_macro_fate_paths.csv",
    "perturbation": "16_virtual_perturbation.csv",
}

FIGURE_FILES = {
    "ei_overview": "fig01_causal_emergence_decomposition.png",
    "spatial_state_ei": "fig02_state_level_ei_spatial.png",
    "single_step": "fig03_single_step_closure.png",
    "multistep": "fig04_multistep_closure.png",
    "random_null": "fig05_matched_random_null.png",
    "multiscale": "fig06_multiscale_consistency.png",
    "effective_spatial": "fig07_effective_states_spatial.png",
    "mechanism": "fig08_grn_cci_mechanism.png",
    "fate": "fig09_macro_fate_paths.png",
    "perturbation": "fig10_virtual_perturbation.png",
}


UNIFIED_TABLE_FILES = {
    "metrics": "01_metrics.csv",
    "states": "02_states.csv",
    "spatial_spots": "03_spatial_spots.csv",
    "closure": "04_dynamical_closure.csv",
    "spatial": "05_spatial_state_metrics.csv",
    "effective": "06_effective_states.csv",
    "mechanism": "07_grn_cci_mechanism.csv",
    "null": "08_matched_random_null.csv",
    "consistency": "09_cross_representation_consistency.csv",
    "fate": "10_macro_fate_paths.csv",
    "perturbation": "11_virtual_perturbation.csv",
}


UNIFIED_FIGURE_FILES = {
    "ei": "fig01_causal_emergence_decomposition.png",
    "spatial_ei": "fig02_state_level_ei_spatial.png",
    "null": "fig03_matched_random_null.png",
    "consistency": "fig04_cross_representation_consistency.png",
    "effective": "fig05_effective_states_spatial.png",
    "mechanism": "fig06_grn_cci_mechanism.png",
    "fate": "fig07_macro_fate_paths.png",
    "perturbation": "fig08_virtual_perturbation.png",
    "closure": "fig09_dynamical_closure_three_panels.png",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _render(
    tables: dict[str, pd.DataFrame],
    spatial_frames: list[pd.DataFrame],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_ei_overview(tables["metrics"], tables["decomposition"], figures_dir / FIGURE_FILES["ei_overview"])
    plot_spatial_state_ei(spatial_frames, figures_dir / FIGURE_FILES["spatial_state_ei"])
    plot_single_step_closure(tables["single_step"], figures_dir / FIGURE_FILES["single_step"])
    plot_multistep_closure(tables["multistep"], figures_dir / FIGURE_FILES["multistep"])
    plot_random_null(tables["random_null"], figures_dir / FIGURE_FILES["random_null"])
    plot_multiscale(tables["consistency"], tables["purity"], figures_dir / FIGURE_FILES["multiscale"])
    plot_effective_spatial(tables["effective"], tables["spatial"], figures_dir / FIGURE_FILES["effective_spatial"])
    plot_mechanism(
        tables["mechanism"],
        tables["correlations"],
        tables["coefficients"],
        figures_dir / FIGURE_FILES["mechanism"],
    )
    plot_fate_paths(tables["fate"], figures_dir / FIGURE_FILES["fate"])
    plot_perturbation(tables["perturbation"], figures_dir / FIGURE_FILES["perturbation"])


def _manifest(output_dir: Path, cfg: DownstreamConfig | None = None) -> dict[str, object]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    payload: dict[str, object] = {
        "tables": [str(path.relative_to(output_dir)) for path in sorted(tables_dir.glob("*.csv"))],
        "figures": [
            str(path.relative_to(output_dir))
            for path in sorted(figures_dir.iterdir())
            if path.suffix.lower() in {".png", ".pdf"}
        ],
        "findings": "findings.json" if (output_dir / "findings.json").exists() else None,
    }
    if cfg is not None:
        payload["configuration"] = {
            "data_root": str(cfg.data_root),
            "metrics_csv": str(cfg.metrics_csv),
            "pair_archive": str(cfg.pair_archive),
            "organ": cfg.organ,
            "times": list(cfg.times),
            "network_method": cfg.network_method,
            "pij_method": cfg.pij_method,
            "layer_pair": f"{cfg.lower_layer}->{cfg.upper_layer}",
            "random_repeats": cfg.random_repeats,
            "random_seed": cfg.random_seed,
            "spatial_knn": cfg.spatial_knn,
            "perturb_random_repeats": cfg.perturb_random_repeats,
        }
    return payload


def run_downstream_analysis(config: DownstreamConfig) -> dict[str, Path]:
    cfg = config.normalized()
    cfg.validate()
    validate_pair_archive(cfg)
    metrics = load_filtered_metrics(cfg)
    output_dir = cfg.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    decomposition, state_table = build_ei_tables(cfg, metrics)
    spatial_frames = build_spatial_state_maps(cfg, state_table)
    multistep = build_multistep_closure(cfg)
    counts_by_time, purity, effective = build_hierarchy_tables(cfg)
    single_step = build_single_step_closure(cfg, counts_by_time)
    consistency = build_multiscale_consistency(cfg, counts_by_time)
    random_null = build_random_null(cfg, counts_by_time)
    spatial = build_spatial_metrics(cfg, state_table)
    grn = build_grn_metrics(cfg)
    cci = build_cci_metrics(cfg)
    mechanism, correlations, coefficients = build_mechanism_table(cfg, state_table, spatial, grn, cci)
    fate = build_fate_paths(cfg, state_table)
    perturbation = build_perturbation_curves(cfg, state_table, grn, cci)

    tables = {
        "metrics": metrics,
        "decomposition": decomposition,
        "state_table": state_table,
        "multistep": multistep,
        "single_step": single_step,
        "purity": purity,
        "effective": effective,
        "consistency": consistency,
        "random_null": random_null,
        "spatial": spatial,
        "grn": grn,
        "cci": cci,
        "mechanism": mechanism,
        "correlations": correlations,
        "coefficients": coefficients,
        "fate": fate,
        "perturbation": perturbation,
    }
    for key, frame in tables.items():
        _write_csv(frame, tables_dir / TABLE_FILES[key])

    _render(tables, spatial_frames, figures_dir)
    findings = summarize_findings(
        cfg,
        metrics,
        decomposition,
        multistep,
        single_step,
        consistency,
        random_null,
        effective,
        correlations,
        perturbation,
    )
    findings_path = output_dir / "findings.json"
    _write_json(findings, findings_path)
    manifest_path = output_dir / "manifest.json"
    _write_json(_manifest(output_dir, cfg), manifest_path)
    return {
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "findings": findings_path,
        "manifest": manifest_path,
    }


def render_downstream_figures(
    results_dir: Path,
    data_root: Path,
    *,
    organ: str = "heart",
    times: tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5"),
) -> dict[str, Path]:
    results_dir = Path(results_dir).resolve()
    data_root = Path(data_root).resolve()
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables = {
        key: pd.read_csv(tables_dir / filename)
        for key, filename in TABLE_FILES.items()
    }
    state_table = tables["state_table"]
    spatial_frames: list[pd.DataFrame] = []
    adjacent_pairs = tuple(f"{left}->{right}" for left, right in zip(times[:-1], times[1:]))
    for layer in ("seurat_k150", "seurat_k40"):
        for pair in adjacent_pairs:
            source, target = pair.split("->")
            domain_map = load_domain_map(data_root, layer, source, organ)
            state = state_table[
                (state_table["time_pair"] == pair) & (state_table["layer"] == layer)
            ][["state", "state_ei"]]
            frame = domain_map.merge(state, left_on="domain_id", right_on="state", how="left")
            frame["state_ei"] = frame["state_ei"].fillna(0.0)
            frame["source_time"] = source
            frame["target_time"] = target
            frame["layer"] = layer
            spatial_frames.append(frame)
    _render(tables, spatial_frames, figures_dir)
    manifest_path = results_dir / "manifest.json"
    _write_json(_manifest(results_dir), manifest_path)
    return {"output_dir": results_dir, "figures_dir": figures_dir, "manifest": manifest_path}


def _render_unified(
    tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: figures_dir / filename for name, filename in UNIFIED_FIGURE_FILES.items()}
    plot_unified_ei_overview(tables["metrics"], paths["ei"])
    plot_unified_spatial_state_ei(tables["spatial_spots"], paths["spatial_ei"])
    plot_unified_random_null(tables["null"], paths["null"])
    plot_cross_representation_consistency(tables["consistency"], paths["consistency"])
    plot_unified_effective_spatial(tables["effective"], tables["spatial"], paths["effective"])
    plot_unified_mechanism(tables["mechanism"], paths["mechanism"])
    plot_unified_fate_paths(tables["fate"], paths["fate"])
    plot_unified_perturbation(tables["perturbation"], paths["perturbation"])
    plot_dynamical_closure_three_panels(tables["closure"], tables["metrics"], paths["closure"])
    return [paths[name] for name in UNIFIED_FIGURE_FILES]


def run_unified_downstream_analysis(config) -> dict[str, Path]:
    """Run the formal four-representation workflow with locked full DeltaEI jobs."""

    cfg = config.normalized()
    cfg.validate()
    output_dir = cfg.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Formal output directory is not empty: {output_dir}. "
            "Use a new output directory; existing downstream results are never overwritten."
        )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    audit_dir = output_dir / "audit"

    ensure_full_unified_inputs(cfg)
    records = load_all_mapping_records(cfg)
    metrics, states = build_unified_ei_tables(cfg, records)
    spatial_spots = build_unified_spatial_spots(cfg, records)
    closure = build_unified_closure_table(cfg, records)
    spatial = build_unified_spatial_metrics(cfg, records)
    effective = build_unified_effective_states(cfg, records)
    mechanism = build_unified_mechanism_table(cfg, records)
    matched_null = build_unified_matched_null(cfg, records)
    consistency = build_cross_representation_consistency(cfg, records)
    fate = build_unified_fate_paths(cfg, records)
    perturbation = build_unified_perturbation_curves(cfg, records, mechanism)
    tables = {
        "metrics": metrics,
        "states": states,
        "spatial_spots": spatial_spots,
        "closure": closure,
        "spatial": spatial,
        "effective": effective,
        "mechanism": mechanism,
        "null": matched_null,
        "consistency": consistency,
        "fate": fate,
        "perturbation": perturbation,
    }
    # Delay materializing the formal results tree until all full-scale caches
    # and downstream tables are ready.  A failed DeltaEI job can then be
    # resumed with the same result path instead of leaving an empty tree that
    # the no-overwrite guard rejects.
    for name, frame in tables.items():
        _write_csv(frame, tables_dir / UNIFIED_TABLE_FILES[name])

    figure_pngs = _render_unified(tables, figures_dir)
    checks = audit_unified_outputs(cfg, tables, figure_pngs)
    checks_path = audit_dir / "validation_checks.csv"
    _write_csv(checks, checks_path)
    manifest = unified_run_manifest(cfg, tables, figure_pngs)
    manifest["validation_passed"] = bool(checks["passed"].all())
    manifest_path = audit_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    if not bool(checks["passed"].all()):
        failures = checks.loc[~checks["passed"], ["check", "detail"]].to_dict(orient="records")
        raise RuntimeError(f"Unified downstream audit failed: {failures}")
    return {
        "output_dir": output_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "validation": checks_path,
        "manifest": manifest_path,
    }


def render_unified_downstream_figures(results_dir: Path) -> dict[str, object]:
    """Re-render figures only from an existing formal table directory."""

    root = Path(results_dir).resolve()
    tables_dir = root / "tables"
    tables = {
        name: pd.read_csv(tables_dir / filename)
        for name, filename in UNIFIED_TABLE_FILES.items()
    }
    figures_dir = root / "figures"
    figure_pngs = _render_unified(tables, figures_dir)
    return {"output_dir": root, "figures_dir": figures_dir, "figure_count": len(figure_pngs)}
