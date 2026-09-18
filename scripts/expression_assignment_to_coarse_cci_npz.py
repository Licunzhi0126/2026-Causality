#!/usr/bin/env python3
"""Build one coarse-node COMMOT CCI matrix from spot expression + assignment.

This script is standalone with respect to this project: it does not import
anything from ``mignet_ce`` or ``data_factory``.  It expects:

1. A wide spot-expression CSV whose leading columns are::

       spot_id,spatial_x,spatial_y,GeneA,GeneB,...

2. A local-to-global assignment CSV containing at least::

       local_id,global_id

``local_id`` is interpreted as the zero-based row number of the expression
CSV.  Coarse expression follows the domain-factory contract: raw counts are
summed over spots assigned to the same global node, while spatial coordinates
are averaged.  COMMOT is then run jointly over all retained ligand-receptor
pairs on the coarse nodes.

A successful run writes exactly two final data files::

    <output-dir>/<sample-name>_coarse_CCI_total.npz
    <output-dir>/<sample-name>_coarse_CCI_edges.csv

The NPZ remains compatible with ``scipy.sparse.load_npz`` and additionally
stores ``node_ids``, ``node_sizes``, and ``coarse_coordinates`` so that matrix
row/column positions can be mapped back to the original global IDs.  The CSV
contains one row per nonzero source-target-ligand-receptor contribution, with
columns::

    source_id,target_id,ligand,receptor,lr_pair,lr_weight,lr_count,lr_weight_sum

For each source-target pair, ``lr_count`` is the number of LR pairs with a
positive COMMOT weight and ``lr_weight_sum`` is the sum of those LR weights.
Consequently, ``lr_weight_sum`` equals the corresponding entry of the NPZ CCI
matrix, up to floating-point roundoff.

Use ``--dry-run`` to validate inputs and dependencies without creating output.
Existing final outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
import scipy.sparse as sp


SCRIPT_VERSION = "1.1.0"
EXPRESSION_METADATA_COLUMNS = ("spot_id", "spatial_x", "spatial_y")
ASSIGNMENT_REQUIRED_COLUMNS = ("local_id", "global_id")
EDGE_COLUMNS = (
    "source_id",
    "target_id",
    "ligand",
    "receptor",
    "lr_pair",
    "lr_weight",
    "lr_count",
    "lr_weight_sum",
)
COORDINATE_CORRELATION_WARNING = 0.99


@dataclass(frozen=True)
class CCIConfig:
    csv_chunk_rows: int = 128
    database_name: str = "cellphonedb_v4_mouse"
    lr_database: str = "CellPhoneDB_v4.0"
    species: str = "mouse"
    min_cell_pct: float = 0.05
    normalize_target_sum: float = 1.0e4
    distance_threshold: float = 200.0
    cot_eps_p: float = 1.0e-1
    cot_rho: float = 1.0e1
    cot_nitermax: int = 10_000

    def validate(self) -> None:
        if self.csv_chunk_rows <= 0:
            raise ValueError("csv_chunk_rows must be positive.")
        if not 0.0 < self.min_cell_pct <= 1.0:
            raise ValueError("min_cell_pct must be in (0, 1].")
        if self.normalize_target_sum <= 0:
            raise ValueError("normalize_target_sum must be positive.")
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")
        if self.cot_eps_p <= 0:
            raise ValueError("cot_eps_p must be positive.")
        if self.cot_rho <= 0:
            raise ValueError("cot_rho must be positive.")
        if self.cot_nitermax <= 0:
            raise ValueError("cot_nitermax must be positive.")


@dataclass
class MicroExpression:
    sample_name: str
    spot_ids: list[str]
    genes: list[str]
    coords: np.ndarray
    counts: sp.csr_matrix


@dataclass
class CoarseExpression:
    sample_name: str
    node_ids: np.ndarray
    node_sizes: np.ndarray
    genes: list[str]
    coords: np.ndarray
    counts: sp.csr_matrix


@dataclass
class CCIResult:
    total: sp.csr_matrix
    edges: pd.DataFrame


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def infer_sample_name(path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        name = str(explicit_name).strip()
    else:
        name = path.stem
        if name.endswith("_expression"):
            name = name[: -len("_expression")]
    if not name:
        raise ValueError("Sample name is empty.")
    if any(character in name for character in ("/", "\\", "\0")):
        raise ValueError(f"Invalid sample name: {name!r}")
    return name


def read_expression_header(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [
        column for column in EXPRESSION_METADATA_COLUMNS if column not in header
    ]
    if missing:
        raise ValueError(f"{path} is missing expression columns: {missing}")
    genes = [
        column for column in header if column not in EXPRESSION_METADATA_COLUMNS
    ]
    if not genes:
        raise ValueError(f"{path} contains no gene columns.")
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path} contains duplicate gene columns.")
    return header, genes


def read_expression_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        path,
        usecols=list(EXPRESSION_METADATA_COLUMNS),
        dtype={"spot_id": str, "spatial_x": np.float64, "spatial_y": np.float64},
    )
    if metadata.empty:
        raise ValueError(f"{path} has no expression rows.")
    if metadata["spot_id"].isna().any() or (
        metadata["spot_id"].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{path} contains empty spot_id values.")
    if metadata["spot_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate spot_id values.")
    coords = metadata.loc[:, ["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{path} contains nonfinite spatial coordinates.")
    return metadata


def _coerce_integer_column(
    values: pd.Series,
    *,
    path: Path,
    column: str,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{path} contains missing or nonnumeric {column!r}.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{path} contains noninteger {column!r}.")
    if numeric.size and float(numeric.min()) < 0:
        raise ValueError(f"{path} contains negative {column!r}.")
    return numeric.astype(np.int64)


def read_assignment_csv(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [
        column for column in ASSIGNMENT_REQUIRED_COLUMNS if column not in frame
    ]
    if missing:
        raise ValueError(f"{path} is missing assignment columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} has no assignment rows.")

    local_ids = _coerce_integer_column(
        frame["local_id"], path=path, column="local_id"
    )
    global_ids = _coerce_integer_column(
        frame["global_id"], path=path, column="global_id"
    )
    if len(np.unique(local_ids)) != len(local_ids):
        raise ValueError(f"{path} contains duplicate local_id values.")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(
            f"Expression/assignment row mismatch: expression={expected_rows}, "
            f"assignment={len(frame)}."
        )
    expected_local_ids = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(np.sort(local_ids), expected_local_ids):
        raise ValueError(
            f"{path} local_id values must be exactly 0..{len(frame) - 1}."
        )

    work = frame.copy()
    work["local_id"] = local_ids
    work["global_id"] = global_ids
    work = work.sort_values("local_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(work["local_id"].to_numpy(), expected_local_ids):
        raise RuntimeError("Internal assignment ordering error.")
    return work


def coordinate_alignment_report(
    expression_metadata: pd.DataFrame,
    assignment: pd.DataFrame,
) -> dict[str, object]:
    report: dict[str, object] = {"available": False}
    if not {"x", "y"}.issubset(assignment.columns):
        return report
    expression_coords = expression_metadata.loc[
        :, ["spatial_x", "spatial_y"]
    ].to_numpy(dtype=float)
    assignment_coords = (
        assignment.loc[:, ["x", "y"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(assignment_coords)):
        raise ValueError("Assignment x/y contains nonfinite values.")

    correlations: list[float] = []
    affine_max_residuals: list[float] = []
    for axis in range(2):
        source = expression_coords[:, axis]
        target = assignment_coords[:, axis]
        if float(np.std(source)) == 0.0 or float(np.std(target)) == 0.0:
            correlation = 1.0 if np.allclose(source, target) else 0.0
            residual = float(np.max(np.abs(source - target)))
        else:
            correlation = float(np.corrcoef(source, target)[0, 1])
            slope, intercept = np.polyfit(source, target, 1)
            residual = float(
                np.max(np.abs((slope * source + intercept) - target))
            )
        correlations.append(correlation)
        affine_max_residuals.append(residual)

    report = {
        "available": True,
        "pearson_x": correlations[0],
        "pearson_y": correlations[1],
        "affine_max_abs_residual_x": affine_max_residuals[0],
        "affine_max_abs_residual_y": affine_max_residuals[1],
    }
    weak_axes = [
        axis
        for axis, correlation in zip(("x", "y"), correlations)
        if abs(correlation) < COORDINATE_CORRELATION_WARNING
    ]
    if weak_axes:
        raise ValueError(
            "Expression and assignment coordinates do not appear row-aligned; "
            f"weak coordinate correlations on axes {weak_axes}: {report}"
        )
    return report


def inspect_inputs(
    expression_path: Path,
    assignment_path: Path,
) -> dict[str, object]:
    header, genes = read_expression_header(expression_path)
    metadata = read_expression_metadata(expression_path)
    assignment = read_assignment_csv(assignment_path, expected_rows=len(metadata))
    alignment = coordinate_alignment_report(metadata, assignment)

    sample = pd.read_csv(expression_path, nrows=3)
    if sample.shape[1] != len(header):
        raise ValueError(
            f"{expression_path} sample rows do not match the header width."
        )
    sample_expression = (
        sample.loc[:, genes].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    if not np.all(np.isfinite(sample_expression)):
        raise ValueError(f"{expression_path} sample has nonnumeric expression.")
    if sample_expression.size and float(sample_expression.min()) < 0:
        raise ValueError(f"{expression_path} sample has negative expression.")

    global_ids = np.sort(assignment["global_id"].unique().astype(np.int64))
    node_sizes = (
        assignment.groupby("global_id", sort=True)
        .size()
        .reindex(global_ids)
        .to_numpy(dtype=np.int64)
    )
    return {
        "expression_path": str(expression_path.resolve()),
        "assignment_path": str(assignment_path.resolve()),
        "expression_bytes": int(expression_path.stat().st_size),
        "assignment_bytes": int(assignment_path.stat().st_size),
        "micro_nodes": int(len(metadata)),
        "genes": int(len(genes)),
        "macro_nodes": int(len(global_ids)),
        "node_ids": global_ids.tolist(),
        "node_sizes": node_sizes.tolist(),
        "coordinate_alignment": alignment,
        "aggregation": "hard assignment; sum raw counts; mean original coordinates",
    }


def read_expression_csv(
    path: Path,
    sample_name: str,
    chunk_rows: int,
) -> MicroExpression:
    _, genes = read_expression_header(path)
    dtype_map: dict[str, object] = {
        "spot_id": str,
        "spatial_x": np.float64,
        "spatial_y": np.float64,
    }
    dtype_map.update({gene: np.float32 for gene in genes})

    spot_ids: list[str] = []
    coord_blocks: list[np.ndarray] = []
    count_blocks: list[sp.csr_matrix] = []
    reader = pd.read_csv(
        path,
        dtype=dtype_map,
        chunksize=max(1, int(chunk_rows)),
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk_ids = chunk["spot_id"].astype(str).tolist()
        coords = chunk.loc[:, ["spatial_x", "spatial_y"]].to_numpy(
            dtype=np.float64
        )
        values = chunk.loc[:, genes].to_numpy(dtype=np.float32, copy=False)
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"{path} contains nonfinite spatial coordinates.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains nonfinite expression values.")
        if values.size and float(values.min()) < 0:
            raise ValueError(f"{path} contains negative expression values.")
        spot_ids.extend(chunk_ids)
        coord_blocks.append(coords)
        count_blocks.append(sp.csr_matrix(values))
        if chunk_number % 10 == 0:
            log(f"Loaded {len(spot_ids)} expression rows from {path.name}")

    if not spot_ids:
        raise ValueError(f"{path} contains no expression rows.")
    if any(not spot_id.strip() for spot_id in spot_ids):
        raise ValueError(f"{path} contains empty spot_id values.")
    if len(set(spot_ids)) != len(spot_ids):
        raise ValueError(f"{path} contains duplicate spot_id values.")

    counts = sp.vstack(count_blocks, format="csr", dtype=np.float32)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    return MicroExpression(
        sample_name=sample_name,
        spot_ids=spot_ids,
        genes=genes,
        coords=np.vstack(coord_blocks),
        counts=counts,
    )


def aggregate_to_coarse(
    expression: MicroExpression,
    assignment: pd.DataFrame,
) -> CoarseExpression:
    n_micro = len(expression.spot_ids)
    if expression.counts.shape[0] != n_micro or expression.coords.shape[0] != n_micro:
        raise ValueError("Expression rows, counts, and coordinates are inconsistent.")
    if len(assignment) != n_micro:
        raise ValueError(
            f"Expression/assignment row mismatch: {n_micro} != {len(assignment)}."
        )

    node_ids = np.sort(assignment["global_id"].unique().astype(np.int64))
    group_index = np.searchsorted(
        node_ids,
        assignment["global_id"].to_numpy(dtype=np.int64),
    )
    membership = sp.csr_matrix(
        (
            np.ones(n_micro, dtype=np.float32),
            (np.arange(n_micro, dtype=np.int64), group_index),
        ),
        shape=(n_micro, len(node_ids)),
    )
    node_sizes = np.asarray(membership.sum(axis=0)).ravel().astype(np.int64)
    if np.any(node_sizes <= 0):
        raise ValueError("At least one coarse node has no assigned micro nodes.")

    coarse_counts = (membership.T @ expression.counts).tocsr().astype(np.float32)
    coarse_counts.sum_duplicates()
    coarse_counts.eliminate_zeros()
    coordinate_sum = membership.T @ np.asarray(expression.coords, dtype=np.float64)
    coarse_coords = np.asarray(
        coordinate_sum / node_sizes.astype(np.float64)[:, None],
        dtype=np.float64,
    )
    if coarse_counts.shape != (len(node_ids), len(expression.genes)):
        raise RuntimeError(
            f"Unexpected coarse expression shape: {coarse_counts.shape}."
        )
    if not np.all(np.isfinite(coarse_coords)):
        raise ValueError("Coarse spatial coordinates contain nonfinite values.")

    return CoarseExpression(
        sample_name=expression.sample_name,
        node_ids=node_ids,
        node_sizes=node_sizes,
        genes=expression.genes,
        coords=coarse_coords,
        counts=coarse_counts,
    )


def load_commot(commot_path: Path | None):
    if commot_path is not None:
        resolved = commot_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"COMMOT source path does not exist: {resolved}")
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
        try:
            return importlib.import_module("commot")
        except Exception as exc:
            raise ImportError(
                f"Unable to import COMMOT from {resolved}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    installed_error: Exception | None = None
    try:
        return importlib.import_module("commot")
    except Exception as exc:
        installed_error = exc
    raise ImportError(
        "Unable to import COMMOT. Install COMMOT and POT (`ot`) or pass "
        f"--commot-path. Original error: {type(installed_error).__name__}: "
        f"{installed_error}"
    ) from installed_error


def dependency_report(
    commot_path: Path | None,
) -> tuple[list[dict[str, str]], bool]:
    requirements = (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("anndata", "anndata"),
        ("scanpy", "scanpy"),
        ("POT", "ot"),
    )
    rows: list[dict[str, str]] = []
    ready = True
    for display_name, module_name in requirements:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            rows.append(
                {
                    "dependency": display_name,
                    "status": "ok",
                    "version": str(version),
                }
            )
        except Exception as exc:
            ready = False
            rows.append(
                {
                    "dependency": display_name,
                    "status": "missing_or_broken",
                    "version": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        commot = load_commot(commot_path)
        rows.append(
            {
                "dependency": "COMMOT",
                "status": "ok",
                "version": str(getattr(commot, "__version__", "unknown")),
            }
        )
    except Exception as exc:
        ready = False
        rows.append(
            {
                "dependency": "COMMOT",
                "status": "missing_or_broken",
                "version": f"{type(exc).__name__}: {exc}",
            }
        )
    return rows, ready


def normalize_ligrec_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(frame).copy()
    if {"ligand", "receptor", "pathway"}.issubset(
        set(map(str, work.columns))
    ):
        work = work.loc[:, ["ligand", "receptor", "pathway"]]
    elif work.shape[1] >= 3:
        work = work.iloc[:, :3]
        work.columns = ["ligand", "receptor", "pathway"]
    else:
        raise ValueError(
            f"COMMOT ligand-receptor table has invalid shape: {work.shape}"
        )
    for column in ("ligand", "receptor", "pathway"):
        work[column] = work[column].astype(str)
    return work.drop_duplicates().reset_index(drop=True)


def prepare_commot_input(
    expression: CoarseExpression,
    config: CCIConfig,
    ct,
):
    import anndata as ad
    import scanpy as sc

    node_names = [str(value) for value in expression.node_ids.tolist()]
    obs = pd.DataFrame(
        {
            "global_id": expression.node_ids,
            "micro_node_count": expression.node_sizes,
        },
        index=pd.Index(node_names, name="global_id"),
    )
    work = ad.AnnData(
        X=expression.counts.copy().astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(expression.genes, name="gene")),
    )
    work.obsm["spatial"] = np.asarray(expression.coords, dtype=np.float32)
    work.var_names_make_unique()
    sc.pp.normalize_total(
        work,
        target_sum=config.normalize_target_sum,
        inplace=True,
    )
    sc.pp.log1p(work)

    ligrec = ct.pp.ligand_receptor_database(
        database=config.lr_database,
        species=config.species,
        heteromeric_delimiter="_",
        signaling_type=None,
    )
    ligrec = ct.pp.filter_lr_database(
        ligrec,
        work,
        heteromeric=True,
        heteromeric_delimiter="_",
        heteromeric_rule="min",
        filter_criteria="min_cell_pct",
        min_cell=100,
        min_cell_pct=config.min_cell_pct,
    )
    ligrec = normalize_ligrec_table(ligrec)
    if ligrec.empty:
        raise ValueError(
            f"No ligand-receptor pairs remain for {expression.sample_name}."
        )
    return work, ligrec


def _validated_communication_matrix(
    matrix,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    if result.shape != expected_shape:
        raise ValueError(
            f"{label} shape {result.shape} does not match {expected_shape}."
        )
    if result.nnz:
        if not np.all(np.isfinite(result.data)):
            raise ValueError(f"{label} contains nonfinite weights.")
        if np.any(result.data < -1.0e-8):
            raise ValueError(f"{label} contains negative weights.")
        result.data[result.data < 0] = 0
        result.eliminate_zeros()
    return result


def build_lr_edge_table(
    work,
    ligrec: pd.DataFrame,
    expression: CoarseExpression,
    config: CCIConfig,
) -> pd.DataFrame:
    expected_shape = (len(expression.node_ids), len(expression.node_ids))
    pair_table = (
        ligrec.loc[:, ["ligand", "receptor"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    blocks: list[pd.DataFrame] = []
    for ligand, receptor in pair_table.itertuples(index=False, name=None):
        ligand = str(ligand)
        receptor = str(receptor)
        key = f"commot-{config.database_name}-{ligand}-{receptor}"
        if key not in work.obsp:
            raise KeyError(f"COMMOT output is missing LR matrix: {key}")
        matrix = _validated_communication_matrix(
            work.obsp[key],
            expected_shape=expected_shape,
            label=f"LR matrix {ligand};{receptor}",
        )
        if matrix.nnz == 0:
            continue
        coo = matrix.tocoo(copy=False)
        blocks.append(
            pd.DataFrame(
                {
                    "source_id": expression.node_ids[coo.row],
                    "target_id": expression.node_ids[coo.col],
                    "ligand": ligand,
                    "receptor": receptor,
                    "lr_pair": f"{ligand};{receptor}",
                    "lr_weight": coo.data,
                }
            )
        )

    if not blocks:
        return pd.DataFrame(columns=list(EDGE_COLUMNS))

    edges = pd.concat(blocks, ignore_index=True)
    summary = (
        edges.groupby(["source_id", "target_id"], sort=False)["lr_weight"]
        .agg(lr_count="size", lr_weight_sum="sum")
        .reset_index()
    )
    edges = edges.merge(
        summary,
        on=["source_id", "target_id"],
        how="left",
        validate="many_to_one",
    )
    edges["source_id"] = edges["source_id"].astype(np.int64)
    edges["target_id"] = edges["target_id"].astype(np.int64)
    edges["lr_weight"] = edges["lr_weight"].astype(np.float64)
    edges["lr_count"] = edges["lr_count"].astype(np.int64)
    edges["lr_weight_sum"] = edges["lr_weight_sum"].astype(np.float64)
    edges = edges.sort_values(
        ["source_id", "target_id", "ligand", "receptor"],
        kind="mergesort",
    ).reset_index(drop=True)
    return edges.loc[:, list(EDGE_COLUMNS)]


def validate_edge_table_against_total(
    edges: pd.DataFrame,
    total: sp.spmatrix,
    node_ids: np.ndarray,
) -> None:
    if list(edges.columns) != list(EDGE_COLUMNS):
        raise ValueError(
            f"Unexpected edge-table columns: {edges.columns.tolist()}"
        )
    total_csr = sp.csr_matrix(total, dtype=np.float64)
    total_csr.sum_duplicates()
    total_csr.eliminate_zeros()
    expected_shape = (len(node_ids), len(node_ids))
    if total_csr.shape != expected_shape:
        raise ValueError(
            f"Total CCI shape {total_csr.shape} does not match {expected_shape}."
        )
    if edges.empty:
        if total_csr.nnz:
            raise ValueError("Edge table is empty but total CCI is nonempty.")
        return

    if edges.duplicated(
        ["source_id", "target_id", "ligand", "receptor"]
    ).any():
        raise ValueError(
            "Edge table contains duplicate source-target-ligand-receptor rows."
        )
    expected_lr_pair = (
        edges["ligand"].astype(str) + ";" + edges["receptor"].astype(str)
    )
    if not np.array_equal(
        edges["lr_pair"].astype(str).to_numpy(),
        expected_lr_pair.to_numpy(),
    ):
        raise ValueError("Edge-table lr_pair must equal ligand + ';' + receptor.")
    weights = edges["lr_weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Edge-table lr_weight values must be finite and positive.")

    calculated = (
        edges.groupby(["source_id", "target_id"], sort=False)["lr_weight"]
        .agg(calculated_count="size", calculated_sum="sum")
        .reset_index()
    )
    advertised = edges.loc[
        :, ["source_id", "target_id", "lr_count", "lr_weight_sum"]
    ].drop_duplicates()
    if len(advertised) != len(calculated):
        raise ValueError(
            "lr_count or lr_weight_sum is inconsistent within an edge."
        )
    checked = advertised.merge(
        calculated,
        on=["source_id", "target_id"],
        how="outer",
        validate="one_to_one",
    )
    if not np.array_equal(
        checked["lr_count"].to_numpy(dtype=np.int64),
        checked["calculated_count"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Edge-table lr_count values are inconsistent.")
    if not np.allclose(
        checked["lr_weight_sum"].to_numpy(dtype=np.float64),
        checked["calculated_sum"].to_numpy(dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("Edge-table lr_weight_sum values are inconsistent.")

    node_to_position = {
        int(node_id): position for position, node_id in enumerate(node_ids)
    }
    row_positions = edges["source_id"].map(node_to_position)
    column_positions = edges["target_id"].map(node_to_position)
    if row_positions.isna().any() or column_positions.isna().any():
        raise ValueError("Edge table contains a source_id or target_id not in node_ids.")
    reconstructed = sp.csr_matrix(
        (
            weights,
            (
                row_positions.to_numpy(dtype=np.int64),
                column_positions.to_numpy(dtype=np.int64),
            ),
        ),
        shape=expected_shape,
    )
    reconstructed.sum_duplicates()
    difference = (reconstructed - total_csr).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        max_error = float(np.max(np.abs(difference.data)))
        scale = (
            max(1.0, float(np.max(np.abs(total_csr.data))))
            if total_csr.nnz
            else 1.0
        )
        if max_error > 1.0e-10 * scale:
            raise ValueError(
                "Sum of per-LR edge weights does not match total CCI; "
                f"max_error={max_error:.12g}."
            )


def run_joint_commot(
    expression: CoarseExpression,
    config: CCIConfig,
    ct,
) -> CCIResult:
    work, ligrec = prepare_commot_input(expression, config, ct)
    log(
        f"Joint COMMOT input: macro_nodes={work.n_obs}, genes={work.n_vars}, "
        f"LR_pairs={len(ligrec)}, node_ids={expression.node_ids.tolist()}"
    )
    ct.tl.spatial_communication(
        work,
        database_name=config.database_name,
        df_ligrec=ligrec,
        pathway_sum=False,
        heteromeric=True,
        heteromeric_rule="min",
        heteromeric_delimiter="_",
        dis_thr=config.distance_threshold,
        cost_scale=None,
        cost_type="euc",
        cot_eps_p=config.cot_eps_p,
        cot_eps_mu=None,
        cot_eps_nu=None,
        cot_rho=config.cot_rho,
        cot_nitermax=config.cot_nitermax,
        cot_weights=(0.25, 0.25, 0.25, 0.25),
        smooth=False,
        smth_eta=None,
        smth_nu=None,
        smth_kernel="exp",
        copy=False,
    )

    total_key = f"commot-{config.database_name}-total-total"
    if total_key not in work.obsp:
        raise KeyError(f"COMMOT output is missing total matrix: {total_key}")
    expected_shape = (len(expression.node_ids), len(expression.node_ids))
    total = _validated_communication_matrix(
        work.obsp[total_key],
        expected_shape=expected_shape,
        label="Final CCI matrix",
    )
    if total.nnz == 0:
        raise ValueError("Final coarse CCI matrix is empty.")
    edges = build_lr_edge_table(work, ligrec, expression, config)
    validate_edge_table_against_total(edges, total, expression.node_ids)
    log(
        f"Joint COMMOT complete: shape={total.shape}, nnz={total.nnz}, "
        f"weight_sum={float(total.data.sum()):.12g}, "
        f"LR_edge_rows={len(edges)}, "
        f"source_target_edges={edges[['source_id', 'target_id']].drop_duplicates().shape[0]}"
    )
    return CCIResult(total=total, edges=edges)


def _verify_npz_payload(
    path: Path,
    expected_matrix: sp.csr_matrix,
    expected_node_ids: np.ndarray,
    expected_node_sizes: np.ndarray,
    expected_coords: np.ndarray,
) -> None:
    loaded_matrix = sp.load_npz(path).tocsr()
    if loaded_matrix.shape != expected_matrix.shape:
        raise ValueError(
            f"Reloaded NPZ shape mismatch: {loaded_matrix.shape} != "
            f"{expected_matrix.shape}."
        )
    difference = (loaded_matrix - expected_matrix).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        max_error = float(np.max(np.abs(difference.data)))
        if max_error > 1.0e-7:
            raise ValueError(f"Reloaded NPZ matrix differs; max_error={max_error}.")

    with np.load(path, allow_pickle=False) as payload:
        for key in ("node_ids", "node_sizes", "coarse_coordinates"):
            if key not in payload.files:
                raise KeyError(f"Saved NPZ is missing metadata key {key!r}.")
        if not np.array_equal(payload["node_ids"], expected_node_ids):
            raise ValueError("Saved node_ids do not match the coarse matrix order.")
        if not np.array_equal(payload["node_sizes"], expected_node_sizes):
            raise ValueError("Saved node_sizes do not match the aggregation.")
        if not np.allclose(
            payload["coarse_coordinates"],
            expected_coords,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("Saved coarse_coordinates do not match aggregation.")


def save_final_npz(
    output_path: Path,
    matrix: sp.spmatrix,
    expression: CoarseExpression,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(
        f"{output_path.stem}.partial.{os.getpid()}{output_path.suffix}"
    )
    if partial.exists():
        raise FileExistsError(
            f"Refusing to reuse existing partial output: {partial}"
        )

    csr = matrix.tocsr().astype(np.float64)
    try:
        np.savez_compressed(
            partial,
            indices=csr.indices,
            indptr=csr.indptr,
            format=np.asarray(b"csr"),
            shape=np.asarray(csr.shape, dtype=np.int64),
            data=csr.data,
            node_ids=np.asarray(expression.node_ids, dtype=np.int64),
            node_sizes=np.asarray(expression.node_sizes, dtype=np.int64),
            coarse_coordinates=np.asarray(expression.coords, dtype=np.float64),
            sample_name=np.asarray(expression.sample_name),
            aggregation_method=np.asarray(
                "hard_assignment_sum_counts_mean_original_coordinates"
            ),
        )
        _verify_npz_payload(
            partial,
            csr,
            expression.node_ids,
            expression.node_sizes,
            expression.coords,
        )
        if output_path.exists():
            raise FileExistsError(
                f"Output appeared during the run: {output_path}"
            )
        partial.rename(output_path)
    except Exception:
        log(f"Write failed; partial output was retained for diagnosis: {partial}")
        raise


def save_edge_csv(
    output_path: Path,
    edges: pd.DataFrame,
    total: sp.spmatrix,
    node_ids: np.ndarray,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(
        f"{output_path.stem}.partial.{os.getpid()}{output_path.suffix}"
    )
    if partial.exists():
        raise FileExistsError(
            f"Refusing to reuse existing partial output: {partial}"
        )

    validate_edge_table_against_total(edges, total, node_ids)
    try:
        edges.to_csv(
            partial,
            index=False,
            columns=list(EDGE_COLUMNS),
            float_format="%.17g",
            lineterminator="\n",
        )
        loaded = pd.read_csv(partial)
        pd.testing.assert_frame_equal(
            loaded,
            edges.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        validate_edge_table_against_total(loaded, total, node_ids)
        if output_path.exists():
            raise FileExistsError(
                f"Output appeared during the run: {output_path}"
            )
        partial.rename(output_path)
    except Exception:
        log(f"Write failed; partial output was retained for diagnosis: {partial}")
        raise


def build_config(args: argparse.Namespace) -> CCIConfig:
    config = CCIConfig(
        csv_chunk_rows=args.csv_chunk_rows,
        min_cell_pct=args.min_cell_pct,
        normalize_target_sum=args.normalize_target_sum,
        distance_threshold=args.distance_threshold,
        cot_nitermax=args.cot_nitermax,
    )
    config.validate()
    return config


def planned_output_paths(
    output_dir: Path,
    sample_name: str,
) -> tuple[Path, Path]:
    resolved = output_dir.resolve()
    return (
        resolved / f"{sample_name}_coarse_CCI_total.npz",
        resolved / f"{sample_name}_coarse_CCI_edges.csv",
    )


def dry_run(
    args: argparse.Namespace,
    config: CCIConfig,
    sample_name: str,
) -> int:
    input_summary = inspect_inputs(args.input_csv, args.assignment_csv)
    npz_output, csv_output = planned_output_paths(
        args.output_dir,
        sample_name,
    )
    dependency_rows, dependencies_ready = dependency_report(args.commot_path)
    print(pd.DataFrame([input_summary]).to_string(index=False))
    print(pd.DataFrame(dependency_rows).to_string(index=False))
    print(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "input_ready": True,
                "dependencies_ready": dependencies_ready,
                "sample_name": sample_name,
                "planned_outputs": {
                    "cci_npz": str(npz_output),
                    "lr_edge_csv": str(csv_output),
                },
                "output_exists": {
                    "cci_npz": npz_output.exists(),
                    "lr_edge_csv": csv_output.exists(),
                },
                "macro_node_order": input_summary["node_ids"],
                "macro_node_sizes": input_summary["node_sizes"],
                "distance_threshold": config.distance_threshold,
                "joint_commot": True,
                "note": (
                    "Dry-run did not create a directory, NPZ, CSV, "
                    "or start COMMOT."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    outputs_available = not npz_output.exists() and not csv_output.exists()
    return 0 if dependencies_ready and outputs_available else 2


def run(args: argparse.Namespace) -> int:
    config = build_config(args)
    input_csv = args.input_csv.resolve()
    assignment_csv = args.assignment_csv.resolve()
    sample_name = infer_sample_name(input_csv, args.sample_name)
    if args.dry_run:
        return dry_run(args, config, sample_name)

    npz_output, csv_output = planned_output_paths(args.output_dir, sample_name)
    existing_outputs = [
        path for path in (npz_output, csv_output) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(map(str, existing_outputs))
        )
    dependency_rows, dependencies_ready = dependency_report(args.commot_path)
    if not dependencies_ready:
        raise RuntimeError(
            "Full-run dependencies are not ready:\n"
            + pd.DataFrame(dependency_rows).to_string(index=False)
        )
    ct = load_commot(args.commot_path)

    log(f"Loading micro expression CSV: {input_csv}")
    expression = read_expression_csv(
        input_csv,
        sample_name,
        chunk_rows=config.csv_chunk_rows,
    )
    assignment = read_assignment_csv(
        assignment_csv,
        expected_rows=len(expression.spot_ids),
    )
    coordinate_alignment_report(
        pd.DataFrame(
            {
                "spot_id": expression.spot_ids,
                "spatial_x": expression.coords[:, 0],
                "spatial_y": expression.coords[:, 1],
            }
        ),
        assignment,
    )
    log(
        f"Micro expression loaded: nodes={len(expression.spot_ids)}, "
        f"genes={len(expression.genes)}, nnz={expression.counts.nnz}"
    )

    coarse = aggregate_to_coarse(expression, assignment)
    log(
        f"Coarse aggregation complete: nodes={len(coarse.node_ids)}, "
        f"genes={len(coarse.genes)}, nnz={coarse.counts.nnz}, "
        f"node_ids={coarse.node_ids.tolist()}, "
        f"node_sizes={coarse.node_sizes.tolist()}"
    )

    started = time.perf_counter()
    result = run_joint_commot(coarse, config, ct)
    save_final_npz(npz_output, result.total, coarse)
    save_edge_csv(
        csv_output,
        result.edges,
        result.total,
        coarse.node_ids,
    )
    elapsed = time.perf_counter() - started
    log(
        f"Wrote {npz_output} and {csv_output} "
        f"in {elapsed / 60.0:.2f} minutes"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--commot-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv-chunk-rows", type=int, default=128)
    parser.add_argument("--min-cell-pct", type=float, default=0.05)
    parser.add_argument("--normalize-target-sum", type=float, default=1.0e4)
    parser.add_argument("--distance-threshold", type=float, default=200.0)
    parser.add_argument("--cot-nitermax", type=int, default=10_000)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        log("Interrupted by user; completed final outputs were left untouched.")
        return 130
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
