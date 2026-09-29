"""Thin Python-only wrappers around the existing exact-K Louvain tools."""

from __future__ import annotations

from dataclasses import asdict
import csv
from pathlib import Path
import subprocess
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CCI_SCRIPT = PROJECT_ROOT / "data_factory" / "scripts" / "run_cci_layer_commot.py"
GRN_SCRIPT = PROJECT_ROOT / "data_factory" / "scripts" / "run_grn_layer.py"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_louvain_layer(*, spot_root: Path, spot_cci_root: Path, layer_root: Path, k: int, sample_names: list[str]) -> dict[str, object]:
    """Adapt GSE sample paths while delegating all Louvain work to the existing builder.

    The repository's legacy Louvain CLIs require organ/stage file stems, whereas
    Plan 1 writes GSE sample IDs.  This adapter calls the public builder helpers
    directly and only supplies the GSE-specific output names and manifest.
    """
    if k not in (40, 150):
        raise ValueError(f"Unsupported Louvain target K={k}; expected 40 or 150.")
    from anndata import read_h5ad
    from data_factory.lib import domain_builder_louvain as builder

    domains_root = layer_root / "domains"
    manifest_path = layer_root / "manifests" / f"domain_manifest_louvain_k{k}.csv"
    domains_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config = builder.BuilderConfig(
        local_dir=spot_root,
        local_commot_dir=spot_cci_root,
        output_root=domains_root,
        k_values=(k,),
        sample_names=tuple(sample_names),
    )
    rows: list[dict[str, object]] = []
    for sample_name in sample_names:
        input_path = spot_root / f"{sample_name}.h5ad"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing Plan 1 prepared H5AD for Louvain: {input_path}")
        output_dir = domains_root / sample_name
        file_stem = f"louvain{k}_{sample_name}"
        output_path = output_dir / f"{file_stem}.h5ad"
        row: dict[str, object] = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "sample_name": sample_name,
            "k": k,
            "status": "planned",
        }
        if output_path.exists():
            row["status"] = "exists_skipped"
            rows.append(row)
            continue
        spot_adata = read_h5ad(input_path)
        try:
            builder.require_spatial(spot_adata, input_path)
            if spot_adata.n_obs < k:
                raise ValueError(f"Louvain K{k} requires at least {k} spots; {sample_name} has {spot_adata.n_obs}.")
            analysis_adata, count_matrix = builder.make_analysis_adata(spot_adata, config)
            cci_total = builder.load_cci_total(sample_name, spot_adata.obs_names.astype(str), config)
            expr_conn, expr_pca = builder.build_expression_connectivity(analysis_adata, config)
            cci_conn = builder.build_cci_connectivity(cci_total, config)
            fused_conn = builder.fuse_connectivities(expr_conn, cci_conn, config)
            merge_features = builder.build_merge_features(expr_pca, cci_total)
            labels, build_info = builder.fit_exact_k_partition(fused_conn, merge_features, target_k=k, cfg=config)
            builder.export_domain_result(spot_adata, count_matrix, labels, output_dir, file_stem, build_info)
            row.update({"status": "written", "n_spots": int(spot_adata.n_obs), "n_domains": int(builder.count_clusters(labels))})
        finally:
            del spot_adata
        rows.append(row)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_file", "output_file", "sample_name", "k", "status", "n_spots", "n_domains"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"builder": "data_factory.lib.domain_builder_louvain", "config": asdict(config), "manifest": str(manifest_path), "rows": rows}


def run_layer_networks(*, python: str, layer_name: str, domains_root: Path, cci_root: Path, grn_root: Path, sample_names: list[str], runner: Runner = subprocess.run) -> dict[str, object]:
    """Call the existing layer-level COMMOT and GRN wrappers for discovered domains."""
    commands = {
        "cci": [python, str(CCI_SCRIPT), "--layer", layer_name, "--input-root", str(domains_root), "--output-root", str(cci_root), "--sample-names", *sample_names, "--workers", "1", "--heartbeat-seconds", "0"],
        "grn": [python, str(GRN_SCRIPT), "--layer", layer_name, "--input-root", str(domains_root), "--output-root", str(grn_root), "--sample-names", *sample_names, "--threads", "1"],
    }
    records: dict[str, object] = {}
    for label, command in commands.items():
        completed = runner(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        records[label] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode != 0:
            raise RuntimeError(f"{label.upper()} for {layer_name} failed: {completed.stderr}")
    return records
