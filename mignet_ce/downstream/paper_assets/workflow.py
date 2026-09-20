from __future__ import annotations

"""Single publication workflow: prepare checked tables, then render them."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd
import numpy as np

from mignet_ce.downstream.analysis.dynamic_closure.optimal_input_existence import (
    evaluate_optimal_runs,
)

from .config import AssetConfig, K10_LEVELS, method_path_slug
from .figure_plots import render_figure1, render_figure1_k10, render_figure2
from .inputs import (
    discover_coarse_roots,
    discover_coarse_runs,
    load_ablation_metrics,
    load_assignments,
    load_domain_map,
    load_full_slice,
    load_spot_coordinates,
    load_vertical_metrics,
    normalize_time_pair,
    select_coarse_run,
)
from .registry import FIGURE_FILES, TABLE_FILES, normalize_asset_names
from .table_plots import (
    render_metric_grid,
    render_table1_bundle,
    render_table2,
    render_table3,
)
from .tables import (
    build_ei_ablation,
    build_ei_hierarchy,
    build_legacy_optimal_table,
    build_optimal_grid,
    build_seurat_cg_hierarchy,
    select_optimal_input_runs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_rendered_asset(source: Path, destination: Path) -> None:
    _copy_file(source, destination)
    source_pdf = source.with_suffix(".pdf")
    if source_pdf.is_file():
        _copy_file(source_pdf, destination.with_suffix(".pdf"))


def _read_cg_metrics(path: Path) -> pd.DataFrame:
    resolved = path / "tables" / "seurat_closure_metrics.csv" if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return pd.read_csv(resolved)


def _primary_ei_metrics(cfg: AssetConfig, ablations: pd.DataFrame | None) -> pd.DataFrame:
    if cfg.ei_metrics_path is not None:
        frame = load_vertical_metrics(cfg.ei_metrics_path)
    else:
        if ablations is None:
            ablations = load_ablation_metrics((cfg.vertical_ablation_root,), dict(cfg.pij_ablations))
        frame = ablations.loc[ablations["Method"].eq("full_NG_KLot")].copy()
    frame = frame.loc[
        frame["network_method"].eq(cfg.network_method)
        & frame["pij_method"].eq(cfg.primary_pij_method)
        & frame["organ"].eq(cfg.organ)
    ].copy()
    if frame.empty:
        raise ValueError("The primary EI source has no matching light_cci_grn / NG_KLot rows")
    keys = ["lower_layer", "upper_layer", "time_pair"]
    if frame.duplicated(keys).any():
        raise ValueError("The primary EI source contains duplicate hierarchy/time-pair rows")
    return frame


def prepare_paper_tables(cfg: AssetConfig, assets: Iterable[str] | None = None) -> dict[str, object]:
    cfg = cfg.normalized()
    cfg.validate()
    requested = normalize_asset_names(list(assets) if assets is not None else None)
    tables_dir = cfg.output_root / "tables"
    audit_dir = cfg.output_root / "audit"
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    optimal_methods_manifest: dict[str, dict[str, object]] = {}
    source_paths: set[Path] = set()

    ablations = None
    if any(name in requested for name in ("table1", "table1_k10")):
        roots = [cfg.vertical_ablation_root]
        if cfg.k10_ablation_root is not None:
            roots.append(cfg.k10_ablation_root)
        ablations = load_ablation_metrics(roots, dict(cfg.pij_ablations))
        source_paths.update(Path(path) for path in ablations["source_metrics"].unique())
        if "table1_k10" in requested and cfg.ei_metrics_path is not None:
            primary_k10 = load_vertical_metrics(cfg.ei_metrics_path)
            primary_k10 = primary_k10.loc[
                primary_k10["upper_layer"].eq("seurat_k10")
                & primary_k10["network_method"].eq(cfg.network_method)
                & primary_k10["pij_method"].eq(cfg.primary_pij_method)
            ].copy()
            primary_k10["Method"] = "full_NG_KLot"
            ablations = pd.concat([ablations, primary_k10], ignore_index=True)
            keys = ["Method", "organ", "lower_layer", "upper_layer", "time_pair"]
            for _, group in ablations.groupby(keys, dropna=False):
                values = pd.to_numeric(group["EI_gain"], errors="coerce").to_numpy(float)
                if len(values) > 1 and not np.allclose(values, values[0], atol=1e-8, rtol=1e-8):
                    raise ValueError(f"Conflicting primary K10 ablation rows: {group[keys].iloc[0].to_dict()}")
            ablations = ablations.drop_duplicates(keys, keep="first")
    if "table1" in requested:
        frames["table1"] = build_ei_ablation(
            ablations, cfg.levels, tuple(cfg.pij_ablations), cfg.time_pairs,
        )
    if "table1_k10" in requested:
        frames["table1_k10"] = build_ei_ablation(
            ablations, K10_LEVELS, tuple(cfg.pij_ablations), cfg.time_pairs,
        )

    if any(name in requested for name in ("table2", "table2_k10", "figure1", "figure1_k10")):
        primary = _primary_ei_metrics(cfg, ablations)
        if cfg.ei_metrics_path is not None:
            source_paths.add(Path(cfg.ei_metrics_path))
        if "table2" in requested or "figure1" in requested:
            frames["table2"] = build_ei_hierarchy(primary, cfg.levels, cfg.time_pairs)
        if "table2_k10" in requested or "figure1_k10" in requested:
            frames["table2_k10"] = build_ei_hierarchy(
                primary, {**cfg.levels, **K10_LEVELS}, cfg.time_pairs,
            )
    if "table2_cg_k10" in requested:
        if cfg.seurat_cg_metrics_path is None:
            raise ValueError("Seurat CG table requires --seurat-cg-metrics")
        cg_path = cfg.seurat_cg_metrics_path
        source_paths.add(cg_path / "tables" / "seurat_closure_metrics.csv" if cg_path.is_dir() else cg_path)
        frames["table2_cg_k10"] = build_seurat_cg_hierarchy(
            _read_cg_metrics(cg_path), cfg.time_pairs,
        )

    if "table3" in requested or "figure2" in requested:
        legacy_runs = discover_coarse_runs(cfg.coarse_root, cfg.coarse_methods)
        if "table3" in requested:
            frames["table3"] = build_legacy_optimal_table(
                legacy_runs, cfg.coarse_methods, cfg.time_pairs, cfg.time_points,
                k=int(cfg.coarse_k), seed=int(cfg.coarse_seed),
            )
        if "figure2" in requested:
            selected = select_coarse_run(
                legacy_runs, cfg.primary_coarse_method, cfg.coarse_scale,
                cfg.figure_pair, k=cfg.coarse_k, seed=cfg.coarse_seed,
            )
            _write_json(audit_dir / "figure2_run.json", {
                "run_dir": str(selected["run_dir"]),
                "delta_ei": float(selected["delta_EI_paper"]),
                "method": cfg.primary_coarse_method,
            })
        source_paths.update(Path(path) for path in legacy_runs["summary_path"])

    if any(name in requested for name in ("table4", "table5", "figure2_optimal")):
        if not cfg.multiscale_roots:
            raise ValueError("Optimal multiscale assets require --multiscale-root")
        for method in cfg.resolved_optimal_coarse_methods():
            slug = method_path_slug(method)
            method_tables_dir = tables_dir / "optimal" / slug
            method_audit_dir = audit_dir / "optimal" / slug
            method_tables_dir.mkdir(parents=True, exist_ok=True)
            method_audit_dir.mkdir(parents=True, exist_ok=True)
            method_manifest: dict[str, object] = {
                "method": method,
                "path_slug": slug,
            }
            runs = discover_coarse_roots(cfg.multiscale_roots, (method,))
            selected: pd.DataFrame | None = None
            if "table4" in requested or "table5" in requested:
                selected = select_optimal_input_runs(
                    runs, method=method,
                    k_by_scale=cfg.optimal_k_by_scale, seed=int(cfg.coarse_seed),
                    time_pairs=cfg.optimal_time_pairs,
                )
                selected_path = method_audit_dir / "selected_optimal_input_runs.csv"
                selected.to_csv(selected_path, index=False)
                method_manifest["selected_runs"] = str(selected_path)
                source_paths.update(Path(path) / "summary.json" for path in selected["run_dir"])
                if method == cfg.primary_coarse_method:
                    _copy_file(selected_path, audit_dir / "selected_optimal_input_runs.csv")
            if "table4" in requested:
                if selected is None:
                    raise RuntimeError(
                        f"Optimal coarse run selection did not produce a result for method {method!r}."
                    )
                table4 = build_optimal_grid(
                    selected, "delta_EI_best_checkpoint", cfg.optimal_time_pairs,
                )
                table4_path = method_tables_dir / TABLE_FILES["table4"]
                table4.to_csv(table4_path, index=False)
                method_manifest["table4"] = str(table4_path)
                if method == cfg.primary_coarse_method:
                    frames["table4"] = table4
            if "table5" in requested:
                if selected is None:
                    raise RuntimeError(
                        f"Optimal coarse run selection did not produce a result for method {method!r}."
                    )
                cg_long = evaluate_optimal_runs(selected)
                cg_long_path = method_tables_dir / "optimal_input_cg_long.csv"
                cg_long.to_csv(cg_long_path, index=False)
                table5 = build_optimal_grid(
                    cg_long, "closure_quality_for_claim", cfg.optimal_time_pairs,
                )
                table5_path = method_tables_dir / TABLE_FILES["table5"]
                table5.to_csv(table5_path, index=False)
                method_manifest["table5"] = str(table5_path)
                method_manifest["closure_long"] = str(cg_long_path)
                for directory in selected["run_dir"]:
                    source_paths.update(
                        Path(directory) / name
                        for name in ("S_t.npy", "S_tp.npy", "PIJ_micro_train.npy")
                    )
                if method == cfg.primary_coarse_method:
                    frames["table5"] = table5
                    _copy_file(cg_long_path, tables_dir / "optimal_input_cg_long.csv")
            if "figure2_optimal" in requested:
                figure_run = select_coarse_run(
                    runs,
                    method,
                    "spot",
                    cfg.figure_pair,
                    k=int(cfg.optimal_k_by_scale["spot"]),
                    seed=int(cfg.coarse_seed),
                    strict=True,
                )
                figure_audit = method_audit_dir / "figure2_run.json"
                _write_json(
                    figure_audit,
                    {
                        "run_dir": str(figure_run["run_dir"]),
                        "delta_ei": float(figure_run["delta_EI_best_checkpoint"]),
                        "method": method,
                        "input_scale": "spot",
                        "K": int(cfg.optimal_k_by_scale["spot"]),
                        "time_pair": cfg.figure_pair,
                        "seed": int(cfg.coarse_seed),
                    },
                )
                method_manifest["figure2_run"] = str(figure_audit)
                figure_run_dir = Path(str(figure_run["run_dir"]))
                source_paths.update(
                    figure_run_dir / name
                    for name in ("summary.json", "assignments_t.csv", "assignments_tp.csv")
                )
            optimal_methods_manifest[method] = method_manifest

    for name, frame in frames.items():
        frame.to_csv(tables_dir / TABLE_FILES[name], index=False)
    manifest = {
        "workflow": "paper_assets_prepare_v1",
        "requested": list(requested),
        "config": asdict(cfg),
        "tables": {name: str(tables_dir / TABLE_FILES[name]) for name in frames},
        "optimal_methods": optimal_methods_manifest,
        "source_sha256": {str(path): _sha256(path) for path in sorted(source_paths) if path.is_file()},
    }
    _write_json(audit_dir / "paper_assets_prepare_manifest.json", manifest)
    return manifest


def _spatial_inputs(cfg: AssetConfig, k10: bool) -> dict[str, object]:
    source_stage, target_stage = normalize_time_pair(cfg.figure_pair).split("->", 1)
    spatial = {
        "source_stage": source_stage,
        "target_stage": target_stage,
        "source_spot": load_spot_coordinates(cfg.data_root, cfg.organ, source_stage),
        "target_spot": load_spot_coordinates(cfg.data_root, cfg.organ, target_stage),
        "full_slice_source": load_full_slice(cfg.slice_root, source_stage, cfg.organ),
    }
    for label, layer in (("k150", "seurat_k150"), ("k40", "seurat_k40")) + (
        (("k10", "seurat_k10"),) if k10 else ()
    ):
        spatial[f"source_{label}"] = load_domain_map(cfg.data_root, layer, cfg.organ, source_stage)
        spatial[f"target_{label}"] = load_domain_map(cfg.data_root, layer, cfg.organ, target_stage)
    return spatial


def render_paper_assets(cfg: AssetConfig, assets: Iterable[str] | None = None) -> dict[str, object]:
    cfg = cfg.normalized()
    cfg.validate()
    requested = normalize_asset_names(list(assets) if assets is not None else None)
    tables_dir = cfg.output_root / "tables"
    figures_dir = cfg.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    required_tables = {
        name for name in requested if name in TABLE_FILES and name not in {"table4", "table5"}
    }
    if "figure1" in requested:
        required_tables.add("table2")
    if "figure1_k10" in requested:
        required_tables.add("table2_k10")
    paths = {name: tables_dir / TABLE_FILES[name] for name in required_tables}
    for method in cfg.resolved_optimal_coarse_methods():
        slug = method_path_slug(method)
        for name in ("table4", "table5"):
            if name in requested:
                paths[f"optimal:{method}:{name}"] = (
                    tables_dir / "optimal" / slug / TABLE_FILES[name]
                )
    before = {name: _sha256(path) for name, path in paths.items()}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    outputs: dict[str, object] = {}

    table_renderers = {
        "table1": lambda frame, path: render_table1_bundle(frame, path, dpi=cfg.dpi),
        "table1_k10": lambda frame, path: render_table1_bundle(
            frame, path, dpi=cfg.dpi, title="Table 1 · PIJ method ablation across K10 hierarchy",
        ),
        "table2": lambda frame, path: render_table2(frame, path, dpi=cfg.dpi),
        "table2_k10": lambda frame, path: render_table2(
            frame, path, dpi=cfg.dpi, title="Table 2 · EI existence across K10 hierarchy",
            subtitle="Primary PIJ method; the established hierarchy is extended to K10.",
        ),
        "table2_cg_k10": lambda frame, path: render_table2(
            frame, path, dpi=cfg.dpi, title="Table 2 · Seurat ClosureQuality across hierarchy",
            subtitle="Common Spot input; low-signal cells are shown as missing.",
        ),
        "table3": lambda frame, path: render_table3(frame, path, dpi=cfg.dpi),
    }
    for name in requested:
        if name in table_renderers:
            path = figures_dir / TABLE_FILES[name].replace(".csv", ".png")
            table_renderers[name](frames[name], path)
            outputs[name] = str(path)

    optimal_outputs: dict[str, dict[str, str]] = {}
    for method in cfg.resolved_optimal_coarse_methods():
        slug = method_path_slug(method)
        method_outputs: dict[str, str] = {}
        method_figures_dir = figures_dir / "optimal" / slug
        if "table4" in requested:
            frame = frames[f"optimal:{method}:table4"]
            path = method_figures_dir / TABLE_FILES["table4"].replace(".csv", ".png")
            render_metric_grid(
                frame,
                path,
                dpi=cfg.dpi,
                title="Table 4 · Optimal coarse-graining DeltaEI by input scale",
                subtitle=(
                    f"Method: {method}. Each value compares optimized output EI "
                    "with its own input EI."
                ),
            )
            method_outputs["table4"] = str(path)
            if method == cfg.primary_coarse_method:
                alias = figures_dir / TABLE_FILES["table4"].replace(".csv", ".png")
                _copy_rendered_asset(path, alias)
                outputs["table4"] = str(alias)
        if "table5" in requested:
            frame = frames[f"optimal:{method}:table5"]
            path = method_figures_dir / TABLE_FILES["table5"].replace(".csv", ".png")
            render_metric_grid(
                frame,
                path,
                dpi=cfg.dpi,
                title="Table 5 · Optimal coarse-graining ClosureQuality by input scale",
                subtitle=(
                    f"Method: {method}. Matched runs and cells with Table 4; "
                    "low-signal cells are shown as missing."
                ),
            )
            method_outputs["table5"] = str(path)
            if method == cfg.primary_coarse_method:
                alias = figures_dir / TABLE_FILES["table5"].replace(".csv", ".png")
                _copy_rendered_asset(path, alias)
                outputs["table5"] = str(alias)
        if "figure2_optimal" in requested:
            source_stage, target_stage = normalize_time_pair(cfg.figure_pair).split("->", 1)
            info = json.loads(
                (
                    cfg.output_root / "audit" / "optimal" / slug / "figure2_run.json"
                ).read_text(encoding="utf-8")
            )
            run_dir = Path(info["run_dir"])
            path = method_figures_dir / FIGURE_FILES["figure2_optimal"]
            render_figure2(
                source_stage=source_stage,
                target_stage=target_stage,
                source_spot=load_spot_coordinates(cfg.data_root, cfg.organ, source_stage),
                target_spot=load_spot_coordinates(cfg.data_root, cfg.organ, target_stage),
                source_assignments=load_assignments(run_dir, "t"),
                target_assignments=load_assignments(run_dir, "tp"),
                delta_ei=float(info["delta_ei"]),
                method_label=method,
                output_path=path,
                full_slice_source=load_full_slice(cfg.slice_root, source_stage, cfg.organ),
                dpi=cfg.dpi,
            )
            method_outputs["figure2_optimal"] = str(path)
        if method_outputs:
            optimal_outputs[method] = method_outputs
    if optimal_outputs:
        outputs["optimal_methods"] = optimal_outputs

    if "figure1" in requested:
        row = frames["table2"].loc[frames["table2"]["Time pair"].eq(cfg.figure_pair)]
        if len(row) != 1:
            raise ValueError(f"Missing Figure 1 time pair {cfg.figure_pair}")
        path = figures_dir / FIGURE_FILES["figure1"]
        render_figure1(
            **_spatial_inputs(cfg, False),
            gains={label: float(row.iloc[0][label]) for label in cfg.levels},
            output_path=path, dpi=cfg.dpi,
        )
        outputs["figure1"] = str(path)
    if "figure1_k10" in requested:
        row = frames["table2_k10"].loc[frames["table2_k10"]["Time pair"].eq(cfg.figure_pair)]
        if len(row) != 1:
            raise ValueError(f"Missing K10 Figure 1 time pair {cfg.figure_pair}")
        path = figures_dir / FIGURE_FILES["figure1_k10"]
        render_figure1_k10(
            **_spatial_inputs(cfg, True),
            gains={label: float(row.iloc[0][label]) for label in {**cfg.levels, **K10_LEVELS}},
            output_path=path, dpi=cfg.dpi,
        )
        outputs["figure1_k10"] = str(path)
    if "figure2" in requested:
        source_stage, target_stage = normalize_time_pair(cfg.figure_pair).split("->", 1)
        info = json.loads((cfg.output_root / "audit" / "figure2_run.json").read_text(encoding="utf-8"))
        run_dir = Path(info["run_dir"])
        path = figures_dir / FIGURE_FILES["figure2"]
        render_figure2(
            source_stage=source_stage, target_stage=target_stage,
            source_spot=load_spot_coordinates(cfg.data_root, cfg.organ, source_stage),
            target_spot=load_spot_coordinates(cfg.data_root, cfg.organ, target_stage),
            source_assignments=load_assignments(run_dir, "t"),
            target_assignments=load_assignments(run_dir, "tp"),
            delta_ei=float(info["delta_ei"]), method_label=str(info["method"]),
            output_path=path,
            full_slice_source=load_full_slice(cfg.slice_root, source_stage, cfg.organ),
            dpi=cfg.dpi,
        )
        outputs["figure2"] = str(path)

    after = {name: _sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("Rendering modified one or more analysis CSV files")
    _write_json(cfg.output_root / "audit" / "paper_assets_render_manifest.json", {
        "workflow": "paper_assets_render_v1",
        "requested": list(requested),
        "table_sha256": before,
        "tables_unchanged": True,
        "outputs": outputs,
    })
    return {"outputs": outputs, "tables_unchanged": True}


def build_paper_assets(
    cfg: AssetConfig, assets: Iterable[str] | None = None, *, stage: str = "all",
) -> dict[str, object]:
    if stage not in {"prepare", "render", "all"}:
        raise ValueError("stage must be prepare, render, or all")
    result: dict[str, object] = {}
    if stage in {"prepare", "all"}:
        result["prepare"] = prepare_paper_tables(cfg, assets)
    if stage in {"render", "all"}:
        result["render"] = render_paper_assets(cfg, assets)
    return result
