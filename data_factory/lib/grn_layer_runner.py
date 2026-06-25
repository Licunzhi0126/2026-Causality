from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad

from factory_common import append_csv, ensure_dir, iter_h5ad_files, write_csv

DEFAULT_THREADS = 32
DEFAULT_N_TREES = 500
DEFAULT_TOP_HVG = 2000
DEFAULT_TOP_EDGE_COUNT = 500_000


def configure_grn_runtime(
    grn,
    *,
    threads: int = DEFAULT_THREADS,
    n_trees: int = DEFAULT_N_TREES,
    top_hvg: int = DEFAULT_TOP_HVG,
    top_edge_count: int = DEFAULT_TOP_EDGE_COUNT,
    tf_list: Path | None = None,
) -> None:
    grn.N_THREADS = int(threads)
    grn.N_TREES = int(n_trees)
    grn.TOP_HVG = int(top_hvg)
    grn.TOP_EDGE_COUNT = int(top_edge_count)
    if tf_list is not None:
        grn.TF_CANDIDATE_FILES = [
            Path(tf_list),
            *[path for path in grn.TF_CANDIDATE_FILES if Path(path) != Path(tf_list)],
        ]
        grn.TF_SYMBOL_CACHE = None


def infer_grn_edges_from_adata(adata_raw, grn) -> tuple[object, dict[str, object]]:
    expression_source = prefer_count_layer(adata_raw)
    adata = grn.preprocess_adata(adata_raw)
    expr_data, genes = grn.extract_expression_matrix(adata)
    regulator_indices = grn.select_regulator_indices(genes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        vim = grn.run_genie3(expr_data, regulator_indices)
    edge_table = grn.build_edge_table(vim, genes)
    metadata = {
        "n_cells": int(adata.n_obs),
        "n_genes_used": int(len(genes)),
        "n_regulators_used": int(len(regulator_indices)),
        "n_edges": int(len(edge_table)),
        "expression_source": expression_source,
    }
    del adata, expr_data, genes, regulator_indices, vim
    gc.collect()
    return edge_table, metadata


def prefer_count_layer(adata) -> str:
    for key in ("count", "counts"):
        if key in adata.layers:
            value = adata.layers[key]
            adata.X = value.copy() if hasattr(value, "copy") else value
            return f"layers[{key!r}]"
    if adata.raw is not None:
        value = adata.raw.X
        adata.X = value.copy() if hasattr(value, "copy") else value
        return "raw.X"
    return "X"


def count_units(path: Path) -> int:
    adata = read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def run_one(file_path: Path, output_root: Path, grn) -> Dict[str, object]:
    dataset_name = file_path.stem
    dataset_output = output_root / dataset_name
    ensure_dir(dataset_output)

    adata_raw = grn.sc.read_h5ad(str(file_path))
    expression_source = prefer_count_layer(adata_raw)
    adata = grn.preprocess_adata(adata_raw)
    del adata_raw

    expr_data, genes = grn.extract_expression_matrix(adata)
    regulator_indices = grn.select_regulator_indices(genes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        vim = grn.run_genie3(expr_data, regulator_indices)

    matrix_path = dataset_output / "grn_vim.npy"
    edges_path = dataset_output / "grn_edges.csv"
    summary_path = dataset_output / "grn_summary.tsv"

    np.save(matrix_path, vim)
    edge_table = grn.build_edge_table(vim, genes)
    edge_table.to_csv(edges_path, index=False)
    summary_df = grn.summarize_edges(dataset_name, edge_table, len(genes), len(regulator_indices))
    summary_df.to_csv(summary_path, sep="\t", index=False)

    row = {
        "input_file": str(file_path),
        "dataset": dataset_name,
        "output_dir": str(dataset_output),
        "total_genes": len(genes),
        "regulators_used": len(regulator_indices),
        "edges_reported": int(edge_table.shape[0]),
        "expression_source": expression_source,
        "status": "written",
    }
    del adata, expr_data, genes, vim, edge_table, regulator_indices
    gc.collect()
    return row


def run_grn_layer(
    input_root: Path,
    output_root: Path,
    manifest_name: str,
    min_units: int = 2,
    sample_names: Sequence[str] = (),
    threads: int = DEFAULT_THREADS,
    n_trees: int = DEFAULT_N_TREES,
    top_hvg: int = DEFAULT_TOP_HVG,
    top_edge_count: int = DEFAULT_TOP_EDGE_COUNT,
    tf_list: Path | None = None,
    manifest_root: Path | None = None,
) -> None:
    import GRN_global as grn

    ensure_dir(output_root)
    configure_grn_runtime(
        grn,
        threads=threads,
        n_trees=n_trees,
        top_hvg=top_hvg,
        top_edge_count=top_edge_count,
        tf_list=tf_list,
    )

    allowed = set(map(str, sample_names))
    files = [
        path
        for path in iter_h5ad_files(input_root, exclude_auxiliary=True)
        if not allowed or path.stem in allowed
    ]
    if not files:
        raise FileNotFoundError(f"No h5ad files found under {input_root}")

    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for file_path in files:
        row: Dict[str, object] = {
            "input_file": str(file_path),
            "dataset": file_path.stem,
            "output_dir": str(output_root / file_path.stem),
            "status": "planned",
        }
        try:
            n_units = count_units(file_path)
            row["n_units"] = n_units
            if n_units < min_units:
                row["status"] = "too_few_units_skipped"
                row["reason"] = f"n_units={n_units} < min_units={min_units}"
                rows.append(row)
                skipped.append(row)
                continue
            row = run_one(file_path, output_root, grn)
            row["n_units"] = n_units
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = f"{type(exc).__name__}: {exc}"
            skipped.append(row)
        rows.append(row)

    manifest_dir = manifest_root if manifest_root is not None else output_root.parents[1] / "manifests"
    manifest = manifest_dir / manifest_name
    write_csv(manifest, rows)
    append_csv(manifest_dir / "skipped_jobs.csv", skipped)
    print(f"[GRN] Wrote manifest: {manifest}")


def build_argparser(description: str, default_input: Path, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--n-trees", type=int, default=DEFAULT_N_TREES)
    parser.add_argument("--top-hvg", type=int, default=DEFAULT_TOP_HVG)
    parser.add_argument("--top-edge-count", type=int, default=DEFAULT_TOP_EDGE_COUNT)
    parser.add_argument("--tf-list", type=Path, default=None)
    return parser
