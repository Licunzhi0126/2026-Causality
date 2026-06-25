from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad

from factory_common import append_csv, ensure_dir, write_csv
from grn_layer_runner import (
    DEFAULT_N_TREES,
    DEFAULT_THREADS,
    DEFAULT_TOP_EDGE_COUNT,
    DEFAULT_TOP_HVG,
    configure_grn_runtime,
    infer_grn_edges_from_adata,
)


UNIT_AUXILIARY_SUFFIX = "_spots_with_domain.h5ad"


def normalize_edge_weights(edge_table: pd.DataFrame) -> pd.DataFrame:
    out = edge_table.copy()
    if out.empty:
        out["weight_norm"] = pd.Series(dtype=float)
        return out
    weights = pd.to_numeric(out["weight"], errors="coerce").to_numpy(dtype=float)
    vmin = float(np.nanmin(weights))
    vmax = float(np.nanmax(weights))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        out["weight_norm"] = 1.0
    else:
        out["weight_norm"] = 1e-6 + (1.0 - 1e-6) * (weights - vmin) / (vmax - vmin)
    return out


def infer_unit_grn_tables(
    adata,
    *,
    unit_column: str = "domain_id",
    min_cells_per_unit: int = 30,
    infer_fn: Callable = infer_grn_edges_from_adata,
    grn=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unit_column not in adata.obs.columns:
        raise ValueError(f"Input h5ad is missing obs[{unit_column!r}].")
    edge_parts: List[pd.DataFrame] = []
    summary_rows: List[dict[str, object]] = []
    unit_values = adata.obs[unit_column].astype(str)
    for unit_id in sorted(unit_values.unique()):
        mask = unit_values.to_numpy() == unit_id
        n_cells = int(mask.sum())
        if n_cells < min_cells_per_unit:
            summary_rows.append(
                {
                    "unit_id": unit_id,
                    "n_cells": n_cells,
                    "n_genes_used": 0,
                    "n_regulators_used": 0,
                    "n_edges": 0,
                    "status": "skipped",
                    "reason": f"n_cells={n_cells} < min_cells_per_unit={min_cells_per_unit}",
                }
            )
            continue
        try:
            unit_adata = adata[mask].copy()
            edges, metadata = infer_fn(unit_adata, grn)
            edges = normalize_edge_weights(edges)
            edges.insert(0, "unit_id", unit_id)
            edges["n_cells"] = n_cells
            edges["grn_status"] = "written"
            edge_parts.append(edges)
            summary_rows.append(
                {
                    "unit_id": unit_id,
                    **metadata,
                    "status": "written",
                    "reason": "",
                }
            )
            del unit_adata, edges
            gc.collect()
        except Exception as exc:
            summary_rows.append(
                {
                    "unit_id": unit_id,
                    "n_cells": n_cells,
                    "n_genes_used": 0,
                    "n_regulators_used": 0,
                    "n_edges": 0,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    edge_columns = [
        "unit_id",
        "regulator",
        "target",
        "weight",
        "weight_norm",
        "n_cells",
        "grn_status",
    ]
    edges = (
        pd.concat(edge_parts, ignore_index=True).reindex(columns=edge_columns)
        if edge_parts
        else pd.DataFrame(columns=edge_columns)
    )
    return edges, pd.DataFrame(summary_rows)


def run_unit_grn_layer(
    *,
    input_root: Path,
    output_root: Path,
    manifest_name: str,
    sample_names: Sequence[str] = (),
    unit_column: str = "domain_id",
    min_cells_per_unit: int = 30,
    threads: int = DEFAULT_THREADS,
    n_trees: int = DEFAULT_N_TREES,
    top_hvg: int = DEFAULT_TOP_HVG,
    top_edge_count: int = DEFAULT_TOP_EDGE_COUNT,
    tf_list: Path | None = None,
    manifest_root: Path | None = None,
) -> None:
    import GRN_global as grn

    configure_grn_runtime(
        grn,
        threads=threads,
        n_trees=n_trees,
        top_hvg=top_hvg,
        top_edge_count=top_edge_count,
        tf_list=tf_list,
    )
    allowed = set(map(str, sample_names))
    files = sorted(input_root.rglob(f"*{UNIT_AUXILIARY_SUFFIX}"))
    files = [
        path
        for path in files
        if not allowed or path.name.removesuffix(UNIT_AUXILIARY_SUFFIX) in allowed
    ]
    if not files:
        raise FileNotFoundError(
            f"No {UNIT_AUXILIARY_SUFFIX} files found under {input_root}."
        )

    manifest_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    for path in files:
        sample = path.name.removesuffix(UNIT_AUXILIARY_SUFFIX)
        sample_output = output_root / sample
        ensure_dir(sample_output)
        row: Dict[str, object] = {
            "input_file": str(path),
            "sample": sample,
            "output_dir": str(sample_output),
            "status": "planned",
        }
        try:
            adata = read_h5ad(path)
            edges, summary = infer_unit_grn_tables(
                adata,
                unit_column=unit_column,
                min_cells_per_unit=min_cells_per_unit,
                grn=grn,
            )
            edges.to_csv(sample_output / "unit_grn_edges.csv", index=False)
            summary.to_csv(sample_output / "unit_grn_summary.csv", index=False)
            row.update(
                {
                    "status": "written",
                    "units_total": int(len(summary)),
                    "units_written": int((summary["status"] == "written").sum()),
                    "units_skipped_or_error": int((summary["status"] != "written").sum()),
                    "edges_reported": int(len(edges)),
                }
            )
            if (summary["status"] != "written").any():
                skipped_rows.extend(
                    summary.loc[summary["status"] != "written"]
                    .assign(sample=sample, input_file=str(path))
                    .to_dict(orient="records")
                )
            del adata, edges, summary
            gc.collect()
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = f"{type(exc).__name__}: {exc}"
            skipped_rows.append(dict(row))
        manifest_rows.append(row)

    default_manifest_base = output_root.parents[1] if len(output_root.parents) > 1 else output_root.parent
    manifest_dir = manifest_root if manifest_root is not None else default_manifest_base / "manifests"
    write_csv(manifest_dir / manifest_name, manifest_rows)
    append_csv(manifest_dir / "skipped_unit_grn_jobs.csv", skipped_rows)
