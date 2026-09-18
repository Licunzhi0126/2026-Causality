from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import AssetConfig
from .figures import render_figure1, render_figure2
from .io import (
    discover_coarse_runs,
    load_assignments,
    load_domain_map,
    load_full_slice,
    load_spot_coordinates,
    load_vertical_metrics,
    normalize_time_pair,
    select_coarse_run,
)
from .registry import normalize_asset_names
from .tables import build_table1, build_table2, build_table3, render_table1_bundle, render_table2, render_table3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_paper_assets(cfg: AssetConfig, assets: Iterable[str] | None = None) -> dict[str, object]:
    """Build requested publication assets from existing pipeline outputs.

    Scientific calculations are not rerun here. The module only reads persisted PIJ
    metrics, coarse-graining summaries/assignments, and spatial mapping files.
    """

    cfg = cfg.normalized()
    cfg.validate()
    requested = normalize_asset_names(list(assets) if assets is not None else None)
    root = cfg.output_root
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    audit_dir = root / "audit"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, object] = {"output_root": str(root), "requested": list(requested)}
    source_files: list[Path] = []

    vertical_metrics: pd.DataFrame | None = None
    table2: pd.DataFrame | None = None
    if any(name in requested for name in ("table1", "table2", "figure1")):
        vertical_metrics = load_vertical_metrics(cfg.vertical_ablation_root)
        for candidate in (cfg.vertical_ablation_root / "all_metrics.csv", cfg.vertical_ablation_root / "metrics.csv"):
            if candidate.exists():
                source_files.append(candidate)
                break

    if "table1" in requested:
        table1 = build_table1(vertical_metrics, cfg)  # type: ignore[arg-type]
        csv_path = tables_dir / "table1_pij_ablation_merged.csv"
        table1.to_csv(csv_path, index=False)
        png_path = figures_dir / "table1_pij_ablation_merged.png"
        render_table1_bundle(table1, png_path, dpi=cfg.dpi)
        outputs["table1"] = {"csv": str(csv_path), "figure": str(png_path)}

    if "table2" in requested or "figure1" in requested:
        table2 = build_table2(vertical_metrics, cfg)  # type: ignore[arg-type]
        if "table2" in requested:
            csv_path = tables_dir / "table2_ei_hierarchy.csv"
            table2.to_csv(csv_path, index=False)
            png_path = figures_dir / "table2_ei_hierarchy.png"
            render_table2(table2, png_path, dpi=cfg.dpi)
            outputs["table2"] = {"csv": str(csv_path), "figure": str(png_path)}

    source_stage, target_stage = normalize_time_pair(cfg.figure_pair).split("->", 1)
    if "figure1" in requested:
        source_spot = load_spot_coordinates(cfg.data_root, cfg.organ, source_stage)
        target_spot = load_spot_coordinates(cfg.data_root, cfg.organ, target_stage)
        source_k150 = load_domain_map(cfg.data_root, "seurat_k150", cfg.organ, source_stage)
        target_k150 = load_domain_map(cfg.data_root, "seurat_k150", cfg.organ, target_stage)
        source_k40 = load_domain_map(cfg.data_root, "seurat_k40", cfg.organ, source_stage)
        target_k40 = load_domain_map(cfg.data_root, "seurat_k40", cfg.organ, target_stage)
        full_slice = load_full_slice(cfg.slice_root, source_stage, cfg.organ)
        assert table2 is not None
        row = table2[table2["Time pair"].astype(str) == cfg.figure_pair]
        if row.empty:
            raise KeyError(f"Table 2 has no row for figure pair {cfg.figure_pair}")
        gains = {label: float(row.iloc[0][label]) for label in cfg.levels}
        path = figures_dir / "figure1_ei_hierarchy_12p5_to_13p5.png"
        render_figure1(
            source_stage=source_stage,
            target_stage=target_stage,
            source_spot=source_spot,
            target_spot=target_spot,
            source_k150=source_k150,
            target_k150=target_k150,
            source_k40=source_k40,
            target_k40=target_k40,
            gains=gains,
            output_path=path,
            full_slice_source=full_slice,
            dpi=cfg.dpi,
        )
        outputs["figure1"] = str(path)

    coarse_runs: pd.DataFrame | None = None
    if any(name in requested for name in ("table3", "figure2")):
        coarse_runs = discover_coarse_runs(cfg.coarse_root, cfg.coarse_methods)

    if "table3" in requested:
        table3 = build_table3(coarse_runs, cfg)  # type: ignore[arg-type]
        csv_path = tables_dir / "table3_optimal_coarse_deltaei_and_k.csv"
        table3.to_csv(csv_path, index=False)
        png_path = figures_dir / "table3_optimal_coarse_deltaei_and_k.png"
        render_table3(table3, png_path, dpi=cfg.dpi)
        outputs["table3"] = {"csv": str(csv_path), "figure": str(png_path)}

    if "figure2" in requested:
        assert coarse_runs is not None
        run = select_coarse_run(
            coarse_runs, cfg.primary_coarse_method, cfg.coarse_scale, cfg.figure_pair,
            k=cfg.coarse_k, seed=cfg.coarse_seed, strict=cfg.strict,
        )
        run_dir = Path(str(run["run_dir"]))
        source_assignments = load_assignments(run_dir, "t")
        target_assignments = load_assignments(run_dir, "tp")
        source_spot = load_spot_coordinates(cfg.data_root, cfg.organ, source_stage)
        target_spot = load_spot_coordinates(cfg.data_root, cfg.organ, target_stage)
        full_slice = load_full_slice(cfg.slice_root, source_stage, cfg.organ)
        path = figures_dir / "figure2_optimal_coarse_12p5_to_13p5.png"
        render_figure2(
            source_stage=source_stage,
            target_stage=target_stage,
            source_spot=source_spot,
            target_spot=target_spot,
            source_assignments=source_assignments,
            target_assignments=target_assignments,
            delta_ei=float(run["delta_EI_paper"]),
            method_label=cfg.primary_coarse_method,
            output_path=path,
            full_slice_source=full_slice,
            dpi=cfg.dpi,
        )
        outputs["figure2"] = {"figure": str(path), "run_dir": str(run_dir)}
        source_files.extend([run_dir / "summary.json", run_dir / "assignments_t.csv", run_dir / "assignments_tp.csv"])

    source_checksums = {str(path): _sha256(path) for path in source_files if path.exists()}
    manifest = {
        "workflow": "downstream_visualization_asset_v3",
        "read_only_scientific_results": True,
        "config": cfg.__dict__,
        "requested": list(requested),
        "outputs": outputs,
        "source_sha256": source_checksums,
    }
    manifest_path = audit_dir / "paper_assets_manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = str(manifest_path)
    return outputs
