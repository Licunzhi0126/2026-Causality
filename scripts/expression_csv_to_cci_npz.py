#!/usr/bin/env python3
"""Convert one count+spatial expression CSV into one COMMOT CCI NPZ.

The script is standalone with respect to this project: it does not import any
module from ``mignet_ce`` or ``data_factory``.  It expects a wide CSV with
these leading columns::

    spot_id,spatial_x,spatial_y,GeneA,GeneB,...

All columns after the three metadata columns are treated as nonnegative gene
expression counts.  A successful full run writes exactly one final data file:

    <output-dir>/<sample-name>_CCI_total.npz

Use ``--dry-run`` to validate the input and runtime without creating output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
import scipy.sparse as sp
from scipy.sparse import save_npz


SCRIPT_VERSION = "1.0.0"
METADATA_COLUMNS = ("spot_id", "spatial_x", "spatial_y")


@dataclass(frozen=True)
class CCIConfig:
    csv_chunk_rows: int = 128
    workers: int = 8
    lr_chunk_size: int = 1
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
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if self.lr_chunk_size <= 0:
            raise ValueError("lr_chunk_size must be positive.")
        if not 0.0 < self.min_cell_pct <= 1.0:
            raise ValueError("min_cell_pct must be in (0, 1].")
        if self.normalize_target_sum <= 0:
            raise ValueError("normalize_target_sum must be positive.")
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")


@dataclass
class ExpressionInput:
    sample_name: str
    spot_ids: list[str]
    genes: list[str]
    coords: np.ndarray
    counts: sp.csr_matrix


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


def inspect_expression_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [column for column in METADATA_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    genes = [column for column in header if column not in METADATA_COLUMNS]
    if not genes:
        raise ValueError(f"{path} contains no gene columns.")
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path} contains duplicate gene columns.")
    sample = pd.read_csv(path, nrows=3)
    if sample.shape[1] != len(header):
        raise ValueError(f"{path} sample rows do not match the header width.")
    if sample.empty:
        raise ValueError(f"{path} has no expression rows.")
    coords = sample.loc[:, ["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    expression = sample.loc[:, genes].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{path} sample contains invalid coordinates.")
    if not np.all(np.isfinite(expression)) or float(expression.min()) < 0:
        raise ValueError(f"{path} sample contains invalid or negative expression.")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "columns": int(len(header)),
        "genes": int(len(genes)),
        "sample_rows_checked": int(len(sample)),
        "first_spot": str(sample.iloc[0]["spot_id"]),
    }


def read_expression_csv(path: Path, sample_name: str, chunk_rows: int) -> ExpressionInput:
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [column for column in METADATA_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    genes = [column for column in header if column not in METADATA_COLUMNS]
    if not genes:
        raise ValueError(f"{path} contains no gene columns.")
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path} contains duplicate gene columns.")

    dtype_map: dict[str, object] = {
        "spot_id": str,
        "spatial_x": np.float64,
        "spatial_y": np.float64,
    }
    dtype_map.update({gene: np.float32 for gene in genes})
    spot_ids: list[str] = []
    coord_blocks: list[np.ndarray] = []
    count_blocks: list[sp.csr_matrix] = []
    reader = pd.read_csv(path, dtype=dtype_map, chunksize=max(1, int(chunk_rows)))
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk_ids = chunk["spot_id"].astype(str).tolist()
        coords = chunk.loc[:, ["spatial_x", "spatial_y"]].to_numpy(dtype=np.float64)
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
            log(f"Loaded {len(spot_ids)} rows from {path.name}")
    if not spot_ids:
        raise ValueError(f"{path} contains no expression rows.")
    if len(set(spot_ids)) != len(spot_ids):
        raise ValueError(f"{path} contains duplicate spot_id values.")
    counts = sp.vstack(count_blocks, format="csr")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    coords = np.vstack(coord_blocks)
    return ExpressionInput(
        sample_name=sample_name,
        spot_ids=spot_ids,
        genes=genes,
        coords=coords,
        counts=counts,
    )


def load_commot(commot_path: Path | None):
    installed_error: Exception | None = None
    try:
        return importlib.import_module("commot")
    except Exception as exc:
        installed_error = exc
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
                f"Unable to import COMMOT from installed packages or {resolved}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    raise ImportError(
        "Unable to import COMMOT. Install COMMOT and POT (`ot`) or pass --commot-path. "
        f"Original error: {type(installed_error).__name__}: {installed_error}"
    ) from installed_error


def dependency_report(commot_path: Path | None) -> tuple[list[dict[str, str]], bool]:
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
            rows.append({"dependency": display_name, "status": "ok", "version": str(version)})
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
        rows.append({"dependency": "COMMOT", "status": "ok", "version": str(getattr(commot, "__version__", "unknown"))})
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
    if {"ligand", "receptor", "pathway"}.issubset(set(map(str, work.columns))):
        work = work.loc[:, ["ligand", "receptor", "pathway"]]
    elif work.shape[1] >= 3:
        work = work.iloc[:, :3]
        work.columns = ["ligand", "receptor", "pathway"]
    else:
        raise ValueError(f"COMMOT ligand-receptor table has invalid shape: {work.shape}")
    for column in ("ligand", "receptor", "pathway"):
        work[column] = work[column].astype(str)
    return work.drop_duplicates().reset_index(drop=True)


def prepare_commot_input(expression: ExpressionInput, config: CCIConfig, ct):
    import anndata as ad
    import scanpy as sc

    work = ad.AnnData(
        X=expression.counts.copy().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(expression.spot_ids, name="spot_id")),
        var=pd.DataFrame(index=pd.Index(expression.genes, name="gene")),
    )
    work.obsm["spatial"] = np.asarray(expression.coords, dtype=np.float32)
    work.var_names_make_unique()
    sc.pp.normalize_total(work, target_sum=config.normalize_target_sum, inplace=True)
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
        raise ValueError(f"No ligand-receptor pairs remain for {expression.sample_name}.")
    return work, ligrec


def run_spatial_communication(work, ligrec: pd.DataFrame, config: CCIConfig, ct) -> None:
    ct.tl.spatial_communication(
        work,
        database_name=config.database_name,
        df_ligrec=ligrec,
        pathway_sum=False,
        heteromeric=True,
        heteromeric_rule="min",
        heteromeric_delimiter="_",
        dis_thr=config.distance_threshold,
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


_FORK_WORK = None
_FORK_CONFIG: CCIConfig | None = None
_FORK_COMMOT = None


def cci_fork_worker(task: tuple[int, list[list[str]]]) -> tuple[int, sp.csr_matrix, int]:
    chunk_index, records = task
    if _FORK_WORK is None or _FORK_CONFIG is None or _FORK_COMMOT is None:
        raise RuntimeError("COMMOT fork worker was not initialized.")
    ligrec = pd.DataFrame(records, columns=["ligand", "receptor", "pathway"])
    work = _FORK_WORK.copy()
    run_spatial_communication(work, ligrec, _FORK_CONFIG, _FORK_COMMOT)
    total = sp.csr_matrix((work.n_obs, work.n_obs), dtype=np.float32)
    found = 0
    for row in ligrec.itertuples(index=False):
        lr_key = f"{row.ligand}-{row.receptor}"
        key = f"commot-{_FORK_CONFIG.database_name}-{lr_key}"
        if key in work.obsp:
            total = total + sp.csr_matrix(work.obsp[key], dtype=np.float32)
            found += 1
    return chunk_index, total.tocsr(), found


def split_ligrec(ligrec: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [
        ligrec.iloc[start : start + chunk_size].copy()
        for start in range(0, len(ligrec), chunk_size)
    ]


def infer_cci(expression: ExpressionInput, config: CCIConfig, ct) -> sp.csr_matrix:
    global _FORK_WORK, _FORK_CONFIG, _FORK_COMMOT

    work, ligrec = prepare_commot_input(expression, config, ct)
    chunks = split_ligrec(ligrec, config.lr_chunk_size)
    requested_workers = min(config.workers, len(chunks))
    can_fork = platform.system() != "Windows" and "fork" in mp.get_all_start_methods()
    workers = requested_workers if can_fork else 1
    if requested_workers > 1 and not can_fork:
        log("POSIX fork is unavailable; COMMOT is falling back to one worker.")
    log(
        f"COMMOT input: spots={work.n_obs}, genes={work.n_vars}, "
        f"LR pairs={len(ligrec)}, chunks={len(chunks)}, workers={workers}"
    )

    total = sp.csr_matrix((work.n_obs, work.n_obs), dtype=np.float32)
    found_pairs = 0
    if workers > 1:
        _FORK_WORK = work
        _FORK_CONFIG = config
        _FORK_COMMOT = ct
        tasks = [
            (
                chunk_index,
                chunk.loc[:, ["ligand", "receptor", "pathway"]]
                .astype(str)
                .values.tolist(),
            )
            for chunk_index, chunk in enumerate(chunks)
        ]
        context = mp.get_context("fork")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(cci_fork_worker, task) for task in tasks]
                completed = 0
                for future in as_completed(futures):
                    _, matrix, found = future.result()
                    total = total + matrix
                    found_pairs += found
                    completed += 1
                    log(f"COMMOT chunks completed: {completed}/{len(chunks)}")
        finally:
            _FORK_WORK = None
            _FORK_CONFIG = None
            _FORK_COMMOT = None
    else:
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_work = work.copy()
            run_spatial_communication(chunk_work, chunk, config, ct)
            for row in chunk.itertuples(index=False):
                lr_key = f"{row.ligand}-{row.receptor}"
                key = f"commot-{config.database_name}-{lr_key}"
                if key in chunk_work.obsp:
                    total = total + sp.csr_matrix(chunk_work.obsp[key], dtype=np.float32)
                    found_pairs += 1
            log(f"COMMOT chunks completed: {chunk_index}/{len(chunks)}")

    total = total.tocsr()
    total.sum_duplicates()
    if total.nnz:
        total.data = np.nan_to_num(total.data, nan=0.0, posinf=0.0, neginf=0.0)
        total.data[total.data < 0] = 0
        total.eliminate_zeros()
    if total.shape != (len(expression.spot_ids), len(expression.spot_ids)):
        raise ValueError(
            f"Final CCI shape {total.shape} does not match {len(expression.spot_ids)} spots."
        )
    if total.nnz == 0:
        raise ValueError("Final CCI matrix is empty.")
    log(
        f"COMMOT aggregation complete: found_pairs={found_pairs}, "
        f"shape={total.shape}, nnz={total.nnz}"
    )
    return total


def save_final_npz(output_path: Path, matrix: sp.spmatrix) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(
        f"{output_path.stem}.partial.{os.getpid()}{output_path.suffix}"
    )
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse existing partial output: {partial}")
    try:
        save_npz(partial, matrix.tocsr())
        if output_path.exists():
            raise FileExistsError(f"Output appeared during the run: {output_path}")
        partial.rename(output_path)
    except Exception:
        log(f"Write failed; any partial output was left untouched: {partial}")
        raise


def build_config(args: argparse.Namespace) -> CCIConfig:
    config = CCIConfig(
        csv_chunk_rows=args.csv_chunk_rows,
        workers=args.workers,
        lr_chunk_size=args.lr_chunk_size,
        min_cell_pct=args.min_cell_pct,
        normalize_target_sum=args.normalize_target_sum,
        distance_threshold=args.distance_threshold,
    )
    config.validate()
    return config


def dry_run(args: argparse.Namespace, config: CCIConfig, sample_name: str) -> int:
    input_summary = inspect_expression_csv(args.input_csv)
    output_path = args.output_dir.resolve() / f"{sample_name}_CCI_total.npz"
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
                "planned_output": str(output_path),
                "output_exists": output_path.exists(),
                "workers": config.workers,
                "lr_chunk_size": config.lr_chunk_size,
                "note": "Dry-run did not create a directory, NPZ, or start COMMOT.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if dependencies_ready and not output_path.exists() else 2


def run(args: argparse.Namespace) -> int:
    config = build_config(args)
    input_csv = args.input_csv.resolve()
    sample_name = infer_sample_name(input_csv, args.sample_name)
    if args.dry_run:
        return dry_run(args, config, sample_name)

    output_path = args.output_dir.resolve() / f"{sample_name}_CCI_total.npz"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    dependency_rows, dependencies_ready = dependency_report(args.commot_path)
    if not dependencies_ready:
        raise RuntimeError(
            "Full-run dependencies are not ready:\n"
            + pd.DataFrame(dependency_rows).to_string(index=False)
        )
    ct = load_commot(args.commot_path)
    log(f"Loading expression CSV: {input_csv}")
    expression = read_expression_csv(
        input_csv,
        sample_name,
        chunk_rows=config.csv_chunk_rows,
    )
    log(
        f"Expression loaded: spots={len(expression.spot_ids)}, "
        f"genes={len(expression.genes)}, nnz={expression.counts.nnz}"
    )
    started = time.perf_counter()
    cci = infer_cci(expression, config, ct)
    save_final_npz(output_path, cci)
    elapsed = time.perf_counter() - started
    log(f"Wrote {output_path} in {elapsed / 60.0:.2f} minutes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--commot-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv-chunk-rows", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--lr-chunk-size", type=int, default=1)
    parser.add_argument("--min-cell-pct", type=float, default=0.05)
    parser.add_argument("--normalize-target-sum", type=float, default=1.0e4)
    parser.add_argument("--distance-threshold", type=float, default=200.0)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        log("Interrupted by user; existing final files were left untouched.")
        return 130
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
