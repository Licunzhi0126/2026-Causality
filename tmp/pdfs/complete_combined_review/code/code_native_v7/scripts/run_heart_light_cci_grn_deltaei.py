#!/usr/bin/env python3
"""Standalone heart-expression -> LightCCI-GRN -> compare_N_kl -> deltaEI pipeline.

This file deliberately does not import any project-local Python module.  It
contains the relevant data extraction, COMMOT CCI, GENIE3-style GRN,
LightCCI-GRN feature, compare_N_kl Pij, and effective-information logic in one
place.  Third-party scientific packages are still required.

Typical usage
-------------

Extract count matrices plus spatial coordinates from the four h5ad files::

    python scripts/run_heart_light_cci_grn_deltaei.py extract \
        --h5ad-dir data/mouse_embyro/E1S1_domain_factory/spot/heart

Inspect inputs and runtime dependencies without running expensive work::

    python scripts/run_heart_light_cci_grn_deltaei.py run \
        --input-dir data/mouse_embyro/E1S1_domain_factory/spot/heart \
        --dry-run

Run the complete pipeline::

    python scripts/run_heart_light_cci_grn_deltaei.py run \
        --input-dir data/mouse_embyro/E1S1_domain_factory/spot/heart

The default hierarchy is gene -> spot, and therefore
deltaEI = EI_spot(light_cci_grn) - EI_gene(GRN).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# anndata releases predating NumPy 2 still access this removed alias.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from sklearn.ensemble import ExtraTreesRegressor


SCRIPT_VERSION = "1.0.0"
STAGE_PATTERN = re.compile(r"^spot_heart_(?P<stage>\d+(?:\.\d+)?)_expression\.csv$")
H5AD_PATTERN = re.compile(r"^spot_heart_(?P<stage>\d+(?:\.\d+)?)\.h5ad$")
REQUIRED_METADATA_COLUMNS = ("spot_id", "spatial_x", "spatial_y")
DEFAULT_STAGES = ("11.5", "12.5", "13.5", "14.5")


@dataclass(frozen=True)
class PipelineConfig:
    stages: tuple[str, ...] = DEFAULT_STAGES
    csv_chunk_rows: int = 128
    cci_workers: int = 8
    cci_lr_chunk_size: int = 1
    cci_database_name: str = "cellphonedb_v4_mouse"
    cci_lr_database: str = "CellPhoneDB_v4.0"
    cci_species: str = "mouse"
    cci_min_cell_pct: float = 0.05
    cci_distance_threshold: float = 200.0
    cci_normalize_target_sum: float = 1.0e4
    cci_cot_eps_p: float = 1.0e-1
    cci_cot_rho: float = 1.0e1
    cci_cot_nitermax: int = 10_000
    grn_workers: int = 32
    grn_n_trees: int = 500
    grn_top_hvg: int = 2_000
    grn_top_edge_count: int = 500_000
    grn_seed: int = 2025
    grn_topk_targets: int = 50
    grn_state_dim: int = 64
    grn_projection_seed: int = 20260713
    nmf_components: int = 5
    nmf_max_iter: int = 300
    nmf_seed: int = 42
    kl_block_weight_n: float = 0.5
    kl_block_weight_g: float = 0.5
    pij_entropy_epsilon: float = 0.05
    pij_temperature: float = 1.0

    def validate(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("At least two stages are required.")
        if self.csv_chunk_rows <= 0:
            raise ValueError("csv_chunk_rows must be positive.")
        if self.cci_workers <= 0 or self.grn_workers <= 0:
            raise ValueError("Worker counts must be positive.")
        if self.cci_lr_chunk_size <= 0:
            raise ValueError("cci_lr_chunk_size must be positive.")
        if self.grn_n_trees <= 0 or self.grn_top_hvg <= 0:
            raise ValueError("GRN tree and HVG counts must be positive.")
        if self.grn_top_edge_count <= 0 or self.grn_topk_targets <= 0:
            raise ValueError("GRN edge limits must be positive.")
        if self.grn_state_dim <= 0 or self.nmf_components <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if self.nmf_max_iter < 0:
            raise ValueError("nmf_max_iter must be nonnegative.")
        if self.kl_block_weight_n < 0 or self.kl_block_weight_g < 0:
            raise ValueError("KL block weights must be nonnegative.")
        if not math.isclose(self.kl_block_weight_n + self.kl_block_weight_g, 1.0, abs_tol=1e-12):
            raise ValueError("KL block weights must sum to 1.")
        if self.pij_entropy_epsilon <= 0 or self.pij_temperature <= 0:
            raise ValueError("Pij epsilon and temperature must be positive.")


@dataclass
class ExpressionMatrix:
    stage: str
    source_path: Path
    spot_ids: list[str]
    genes: list[str]
    coords: np.ndarray
    counts: sp.csr_matrix


@dataclass
class StageNetwork:
    stage: str
    gene_units: list[str]
    gene_adjacency: sp.csr_matrix
    spot_units: list[str]
    spot_cci_adjacency: sp.csr_matrix
    spot_grn_state: np.ndarray


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def write_dataframe_new(path: Path, frame: pd.DataFrame, *, sep: str = ",") -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False, sep=sep)


def save_sparse_new(path: Path, matrix: sp.spmatrix) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, matrix.tocsr())


def save_array_new(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(array))


def natural_sort(values: Iterable[str]) -> list[str]:
    def key(value: str):
        return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]

    return sorted(map(str, values), key=key)


def all_time_pairs(stages: Sequence[str]) -> list[tuple[int, int]]:
    return [(left, right) for left in range(len(stages)) for right in range(left + 1, len(stages))]


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def _read_h5ad_index(handle, group_name: str) -> list[str]:
    group = handle[group_name]
    index_key = group.attrs.get("_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if index_key is None:
        index_key = "_index"
    if str(index_key) not in group:
        raise KeyError(f"Cannot find {group_name} index dataset {index_key!r}.")
    return _decode_strings(group[str(index_key)][:])


def _read_h5ad_matrix(handle, key: str) -> sp.csr_matrix:
    obj = handle[key]
    if hasattr(obj, "keys") and {"data", "indices", "indptr"}.issubset(obj.keys()):
        shape = tuple(int(value) for value in obj.attrs["shape"])
        return sp.csr_matrix((obj["data"][:], obj["indices"][:], obj["indptr"][:]), shape=shape)
    values = np.asarray(obj[:])
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D matrix at {key}, got {values.shape}.")
    return sp.csr_matrix(values)


def read_h5ad_count_and_spatial(path: Path) -> tuple[list[str], list[str], np.ndarray, sp.csr_matrix, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        matrix_key = None
        for candidate in ("layers/count", "layers/counts", "raw/X", "X"):
            if candidate in handle:
                matrix_key = candidate
                break
        if matrix_key is None:
            raise KeyError(f"{path} contains no count/counts/raw.X/X matrix.")
        if "obsm/spatial" not in handle:
            raise KeyError(f"{path} is missing obsm['spatial'].")
        spot_ids = _read_h5ad_index(handle, "obs")
        genes = _read_h5ad_index(handle, "var")
        coords = np.asarray(handle["obsm/spatial"][:], dtype=float)
        counts = _read_h5ad_matrix(handle, matrix_key).tocsr()
    if counts.shape != (len(spot_ids), len(genes)):
        raise ValueError(f"Matrix shape {counts.shape} does not match obs/var lengths in {path}.")
    if coords.ndim != 2 or coords.shape[0] != len(spot_ids) or coords.shape[1] < 2:
        raise ValueError(f"Invalid spatial shape {coords.shape} in {path}.")
    counts.sum_duplicates()
    if counts.nnz and (not np.all(np.isfinite(counts.data)) or float(counts.data.min()) < 0):
        raise ValueError(f"Expression matrix in {path} contains nonfinite or negative values.")
    return spot_ids, genes, coords[:, :2], counts, matrix_key


def export_expression_csv(h5ad_path: Path, output_path: Path, *, chunk_rows: int) -> dict[str, object]:
    if output_path.exists():
        log(f"Expression CSV already exists; leaving it untouched: {output_path}")
        return {"status": "exists_skipped", "output": str(output_path)}
    spot_ids, genes, coords, counts, matrix_key = read_h5ad_count_and_spatial(h5ad_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse existing partial file: {partial}")
    log(
        f"Exporting {h5ad_path.name}: shape={counts.shape}, nnz={counts.nnz}, "
        f"source={matrix_key} -> {output_path.name}"
    )
    try:
        with partial.open("x", newline="", encoding="utf-8") as handle:
            first = True
            for start in range(0, counts.shape[0], max(1, int(chunk_rows))):
                stop = min(start + max(1, int(chunk_rows)), counts.shape[0])
                block = counts[start:stop].toarray()
                if np.allclose(block, np.rint(block), atol=0.0, rtol=0.0):
                    block = np.rint(block).astype(np.int64, copy=False)
                frame = pd.DataFrame(block, columns=genes)
                frame.insert(0, "spatial_y", coords[start:stop, 1])
                frame.insert(0, "spatial_x", coords[start:stop, 0])
                frame.insert(0, "spot_id", spot_ids[start:stop])
                frame.to_csv(handle, index=False, header=first)
                first = False
                log(f"  rows {start + 1}-{stop}/{counts.shape[0]}")
        partial.rename(output_path)
    except Exception:
        log(f"Export failed; partial file was retained for diagnosis: {partial}")
        raise
    return {
        "status": "written",
        "input": str(h5ad_path),
        "output": str(output_path),
        "matrix_source": matrix_key,
        "n_spots": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "nnz": int(counts.nnz),
        "count_sum": float(counts.sum()),
    }


def extract_command(args: argparse.Namespace) -> int:
    h5ad_dir = args.h5ad_dir.resolve()
    files: dict[str, Path] = {}
    for path in sorted(h5ad_dir.glob("*.h5ad")):
        match = H5AD_PATTERN.match(path.name)
        if match:
            files[match.group("stage")] = path
    missing = [stage for stage in args.stages if stage not in files]
    if missing:
        raise FileNotFoundError(f"Missing h5ad stages under {h5ad_dir}: {missing}")
    summaries = []
    for stage in args.stages:
        output = h5ad_dir / f"spot_heart_{stage}_expression.csv"
        summaries.append(export_expression_csv(files[stage], output, chunk_rows=args.chunk_rows))
    log("Expression extraction finished.")
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


def discover_expression_csvs(input_dir: Path, stages: Sequence[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.csv")):
        match = STAGE_PATTERN.match(path.name)
        if match:
            found[match.group("stage")] = path
    missing = [stage for stage in stages if stage not in found]
    if missing:
        raise FileNotFoundError(f"Missing expression CSV stages under {input_dir}: {missing}")
    return {stage: found[stage] for stage in stages}


def inspect_expression_csv(path: Path, *, sample_rows: int = 3) -> dict[str, object]:
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}.")
    genes = [column for column in header if column not in REQUIRED_METADATA_COLUMNS]
    if not genes:
        raise ValueError(f"{path} contains no gene columns.")
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path} contains duplicate gene columns.")
    sample = pd.read_csv(path, nrows=sample_rows)
    if sample.shape[1] != len(header):
        raise ValueError(f"{path} sample width differs from its header.")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "columns": int(len(header)),
        "genes": int(len(genes)),
        "sample_rows_checked": int(len(sample)),
        "first_spot": str(sample.iloc[0]["spot_id"]) if not sample.empty else None,
    }


def read_expression_csv(path: Path, stage: str, *, chunk_rows: int) -> ExpressionMatrix:
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}.")
    genes = [column for column in header if column not in REQUIRED_METADATA_COLUMNS]
    if len(set(genes)) != len(genes):
        raise ValueError(f"{path} contains duplicate gene columns.")
    dtype_map: dict[str, object] = {"spot_id": str, "spatial_x": np.float64, "spatial_y": np.float64}
    dtype_map.update({gene: np.float32 for gene in genes})
    spot_ids: list[str] = []
    coord_blocks: list[np.ndarray] = []
    count_blocks: list[sp.csr_matrix] = []
    reader = pd.read_csv(path, dtype=dtype_map, chunksize=max(1, int(chunk_rows)))
    for chunk_index, chunk in enumerate(reader, start=1):
        spot_ids.extend(chunk["spot_id"].astype(str).tolist())
        coords = chunk.loc[:, ["spatial_x", "spatial_y"]].to_numpy(dtype=float)
        values = chunk.loc[:, genes].to_numpy(dtype=np.float32, copy=False)
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains nonfinite coordinates or expression values.")
        if values.size and float(values.min()) < 0:
            raise ValueError(f"{path} contains negative expression values.")
        coord_blocks.append(coords)
        count_blocks.append(sp.csr_matrix(values))
        if chunk_index % 10 == 0:
            log(f"  loaded {len(spot_ids)} rows from {path.name}")
    if len(set(spot_ids)) != len(spot_ids):
        raise ValueError(f"{path} contains duplicate spot_id values.")
    counts = sp.vstack(count_blocks, format="csr") if count_blocks else sp.csr_matrix((0, len(genes)))
    coords = np.vstack(coord_blocks) if coord_blocks else np.zeros((0, 2), dtype=float)
    return ExpressionMatrix(stage=stage, source_path=path, spot_ids=spot_ids, genes=genes, coords=coords, counts=counts)


def load_commot(commot_path: Path | None = None):
    try:
        return importlib.import_module("commot")
    except Exception as installed_error:
        if commot_path is not None:
            resolved = commot_path.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"COMMOT source path does not exist: {resolved}") from installed_error
            sys.path.insert(0, str(resolved))
            try:
                return importlib.import_module("commot")
            except Exception as source_error:
                raise ImportError(
                    f"Unable to import COMMOT from installed packages or {resolved}: "
                    f"{type(source_error).__name__}: {source_error}"
                ) from source_error
        raise ImportError(
            "Unable to import COMMOT. Install COMMOT and POT (`ot`), or pass --commot-path. "
            f"Original error: {type(installed_error).__name__}: {installed_error}"
        ) from installed_error


def dependency_report(commot_path: Path | None = None) -> tuple[list[dict[str, object]], bool]:
    checks: list[tuple[str, str]] = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("h5py", "h5py"),
        ("anndata", "anndata"),
        ("scanpy", "scanpy"),
        ("scikit-misc", "skmisc"),
        ("POT", "ot"),
    ]
    rows: list[dict[str, object]] = []
    ready = True
    for display, module_name in checks:
        try:
            module = importlib.import_module(module_name)
            rows.append({"dependency": display, "status": "ok", "version": getattr(module, "__version__", "unknown")})
        except Exception as exc:
            ready = False
            rows.append({"dependency": display, "status": "missing_or_broken", "version": f"{type(exc).__name__}: {exc}"})
    try:
        module = load_commot(commot_path)
        rows.append({"dependency": "COMMOT", "status": "ok", "version": getattr(module, "__version__", "unknown")})
    except Exception as exc:
        ready = False
        rows.append({"dependency": "COMMOT", "status": "missing_or_broken", "version": f"{type(exc).__name__}: {exc}"})
    return rows, ready


def _normalize_ligrec_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(frame).copy()
    if {"ligand", "receptor", "pathway"}.issubset(set(map(str, work.columns))):
        work = work.loc[:, ["ligand", "receptor", "pathway"]]
    elif work.shape[1] >= 3:
        work = work.iloc[:, :3]
        work.columns = ["ligand", "receptor", "pathway"]
    else:
        raise ValueError(f"Ligand-receptor table has invalid shape: {work.shape}")
    for column in ("ligand", "receptor", "pathway"):
        work[column] = work[column].astype(str)
    return work.drop_duplicates().reset_index(drop=True)


def prepare_commot_input(expression: ExpressionMatrix, cfg: PipelineConfig, ct):
    import anndata as ad
    import scanpy as sc

    work = ad.AnnData(
        X=expression.counts.copy().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(expression.spot_ids, name="spot_id")),
        var=pd.DataFrame(index=pd.Index(expression.genes, name="gene")),
    )
    work.obsm["spatial"] = np.asarray(expression.coords, dtype=np.float32)
    work.var_names_make_unique()
    sc.pp.normalize_total(work, target_sum=cfg.cci_normalize_target_sum, inplace=True)
    sc.pp.log1p(work)
    ligrec = ct.pp.ligand_receptor_database(
        database=cfg.cci_lr_database,
        species=cfg.cci_species,
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
        min_cell_pct=cfg.cci_min_cell_pct,
    )
    ligrec = _normalize_ligrec_table(ligrec)
    if ligrec.empty:
        raise ValueError(f"No ligand-receptor pairs remain for stage {expression.stage}.")
    return work, ligrec


def _run_commot_spatial(work, ligrec: pd.DataFrame, cfg: PipelineConfig, ct) -> None:
    ct.tl.spatial_communication(
        work,
        database_name=cfg.cci_database_name,
        df_ligrec=ligrec,
        pathway_sum=False,
        heteromeric=True,
        heteromeric_rule="min",
        heteromeric_delimiter="_",
        dis_thr=cfg.cci_distance_threshold,
        cost_type="euc",
        cot_eps_p=cfg.cci_cot_eps_p,
        cot_eps_mu=None,
        cot_eps_nu=None,
        cot_rho=cfg.cci_cot_rho,
        cot_nitermax=cfg.cci_cot_nitermax,
        cot_weights=(0.25, 0.25, 0.25, 0.25),
        smooth=False,
        smth_eta=None,
        smth_nu=None,
        smth_kernel="exp",
        copy=False,
    )


_CCI_FORK_WORK = None
_CCI_FORK_CONFIG: PipelineConfig | None = None
_CCI_FORK_CT = None


def _cci_fork_worker(task: tuple[int, list[list[str]]]) -> tuple[int, sp.csr_matrix, list[str]]:
    chunk_index, records = task
    if _CCI_FORK_WORK is None or _CCI_FORK_CONFIG is None or _CCI_FORK_CT is None:
        raise RuntimeError("CCI fork worker is not initialized.")
    ligrec = pd.DataFrame(records, columns=["ligand", "receptor", "pathway"])
    work = _CCI_FORK_WORK.copy()
    _run_commot_spatial(work, ligrec, _CCI_FORK_CONFIG, _CCI_FORK_CT)
    total = sp.csr_matrix((work.n_obs, work.n_obs), dtype=np.float32)
    found: list[str] = []
    for row in ligrec.itertuples(index=False):
        lr_key = f"{row.ligand}-{row.receptor}"
        key = f"commot-{_CCI_FORK_CONFIG.cci_database_name}-{lr_key}"
        if key in work.obsp:
            total = total + sp.csr_matrix(work.obsp[key], dtype=np.float32)
            found.append(lr_key)
    return chunk_index, total.tocsr(), found


def _ligrec_chunks(ligrec: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [ligrec.iloc[start : start + chunk_size].copy() for start in range(0, len(ligrec), chunk_size)]


def infer_cci(expression: ExpressionMatrix, cfg: PipelineConfig, ct) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, object]]:
    global _CCI_FORK_WORK, _CCI_FORK_CONFIG, _CCI_FORK_CT

    work, ligrec = prepare_commot_input(expression, cfg, ct)
    chunks = _ligrec_chunks(ligrec, cfg.cci_lr_chunk_size)
    requested_workers = min(cfg.cci_workers, len(chunks))
    can_fork = platform.system() != "Windows" and "fork" in mp.get_all_start_methods()
    workers = requested_workers if can_fork else 1
    if requested_workers > 1 and not can_fork:
        log("COMMOT multiprocessing requires POSIX fork; falling back to one worker on this platform.")
    log(f"Stage {expression.stage}: COMMOT {len(ligrec)} LR pairs, {len(chunks)} chunks, workers={workers}")
    total = sp.csr_matrix((len(expression.spot_ids), len(expression.spot_ids)), dtype=np.float32)
    found_keys: list[str] = []
    if workers > 1:
        _CCI_FORK_WORK = work
        _CCI_FORK_CONFIG = cfg
        _CCI_FORK_CT = ct
        tasks = [
            (index, chunk.loc[:, ["ligand", "receptor", "pathway"]].astype(str).values.tolist())
            for index, chunk in enumerate(chunks)
        ]
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_cci_fork_worker, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                _, matrix, keys = future.result()
                total = total + matrix
                found_keys.extend(keys)
                completed += 1
                log(f"  COMMOT chunks {completed}/{len(chunks)}")
        _CCI_FORK_WORK = None
        _CCI_FORK_CONFIG = None
        _CCI_FORK_CT = None
    else:
        for index, chunk in enumerate(chunks, start=1):
            chunk_work = work.copy()
            _run_commot_spatial(chunk_work, chunk, cfg, ct)
            for row in chunk.itertuples(index=False):
                lr_key = f"{row.ligand}-{row.receptor}"
                key = f"commot-{cfg.cci_database_name}-{lr_key}"
                if key in chunk_work.obsp:
                    total = total + sp.csr_matrix(chunk_work.obsp[key], dtype=np.float32)
                    found_keys.append(lr_key)
            log(f"  COMMOT chunks {index}/{len(chunks)}")
    total = total.tocsr()
    total.sum_duplicates()
    total.eliminate_zeros()
    metadata = {
        "method": "official_commot_lr_chunk_aggregate",
        "stage": expression.stage,
        "n_units": len(expression.spot_ids),
        "n_genes": len(expression.genes),
        "n_lr_pairs_filtered": len(ligrec),
        "n_lr_pairs_found": len(found_keys),
        "cci_shape": list(total.shape),
        "cci_nnz": int(total.nnz),
        "workers": int(workers),
        "lr_chunk_size": int(cfg.cci_lr_chunk_size),
    }
    return total, ligrec, metadata


_GRN_EXPR: np.ndarray | None = None
_GRN_REGULATORS: list[int] | None = None
_GRN_N_TREES: int | None = None
_GRN_SEED: int | None = None


def _init_grn_worker(expression: np.ndarray, regulators: Sequence[int], n_trees: int, seed: int) -> None:
    global _GRN_EXPR, _GRN_REGULATORS, _GRN_N_TREES, _GRN_SEED
    _GRN_EXPR = expression
    _GRN_REGULATORS = list(regulators)
    _GRN_N_TREES = int(n_trees)
    _GRN_SEED = int(seed)


def _grn_target_worker(target_index: int) -> tuple[int, np.ndarray]:
    if _GRN_EXPR is None or _GRN_REGULATORS is None or _GRN_N_TREES is None or _GRN_SEED is None:
        raise RuntimeError("GRN worker is not initialized.")
    n_genes = _GRN_EXPR.shape[1]
    regulators = [index for index in _GRN_REGULATORS if index != target_index]
    if not regulators:
        return target_index, np.zeros(n_genes, dtype=np.float32)
    output = _GRN_EXPR[:, target_index]
    standard_deviation = float(np.std(output))
    if standard_deviation == 0.0:
        return target_index, np.zeros(n_genes, dtype=np.float32)
    model = ExtraTreesRegressor(
        n_estimators=_GRN_N_TREES,
        max_features="sqrt",
        random_state=_GRN_SEED + target_index,
        n_jobs=1,
    )
    model.fit(_GRN_EXPR[:, regulators], output / standard_deviation)
    values = np.zeros(n_genes, dtype=np.float32)
    values[np.asarray(regulators, dtype=int)] = model.feature_importances_.astype(np.float32, copy=False)
    return target_index, values


def preprocess_grn_expression(expression: ExpressionMatrix, cfg: PipelineConfig) -> tuple[np.ndarray, list[str]]:
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(
        X=expression.counts.copy().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(expression.spot_ids, name="spot_id")),
        var=pd.DataFrame(index=pd.Index(expression.genes, name="gene")),
    )
    sc.pp.normalize_total(adata, target_sum=1.0e4)
    sc.pp.log1p(adata)
    adata.var_names_make_unique()
    if cfg.grn_top_hvg < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=cfg.grn_top_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()
    values = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    return values.astype(np.float32, copy=False), adata.var_names.astype(str).tolist()


def run_genie3(expression: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    regulators = list(range(expression.shape[1]))
    targets = list(range(expression.shape[1]))
    vim = np.zeros((expression.shape[1], expression.shape[1]), dtype=np.float32)
    log(
        f"GENIE3 targets={len(targets)}, regulators={len(regulators)}, "
        f"trees={cfg.grn_n_trees}, workers={cfg.grn_workers}"
    )
    if cfg.grn_workers == 1:
        _init_grn_worker(expression, regulators, cfg.grn_n_trees, cfg.grn_seed)
        for completed, target in enumerate(targets, start=1):
            target_index, values = _grn_target_worker(target)
            vim[target_index, :] = values
            if completed % 50 == 0 or completed == len(targets):
                log(f"  GENIE3 targets {completed}/{len(targets)}")
    else:
        with mp.get_context("spawn" if platform.system() == "Windows" else "fork").Pool(
            processes=cfg.grn_workers,
            initializer=_init_grn_worker,
            initargs=(expression, regulators, cfg.grn_n_trees, cfg.grn_seed),
        ) as pool:
            completed = 0
            for target_index, values in pool.imap_unordered(_grn_target_worker, targets):
                vim[target_index, :] = values
                completed += 1
                if completed % 50 == 0 or completed == len(targets):
                    log(f"  GENIE3 targets {completed}/{len(targets)}")
    return vim.T


def grn_edge_table(vim: np.ndarray, genes: Sequence[str], top_edge_count: int) -> pd.DataFrame:
    flat = np.asarray(vim, dtype=np.float32).ravel()
    positive = np.flatnonzero(flat > 0)
    if positive.size == 0:
        return pd.DataFrame(columns=["regulator", "target", "weight"])
    if positive.size > top_edge_count:
        local = np.argpartition(flat[positive], -top_edge_count)[-top_edge_count:]
        selected = positive[local]
    else:
        selected = positive
    selected = selected[np.argsort(flat[selected], kind="mergesort")[::-1]]
    regulators, targets = np.unravel_index(selected, vim.shape)
    gene_array = np.asarray(list(map(str, genes)), dtype=object)
    return pd.DataFrame(
        {
            "regulator": gene_array[regulators],
            "target": gene_array[targets],
            "weight": flat[selected],
        }
    )


def infer_grn(expression: ExpressionMatrix, cfg: PipelineConfig) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    values, genes = preprocess_grn_expression(expression, cfg)
    vim = run_genie3(values, cfg)
    edges = grn_edge_table(vim, genes, cfg.grn_top_edge_count)
    metadata = {
        "method": "GENIE3_style_ExtraTrees",
        "stage": expression.stage,
        "n_spots": len(expression.spot_ids),
        "n_genes_used": len(genes),
        "n_regulators_used": len(genes),
        "n_edges": len(edges),
        "n_trees": cfg.grn_n_trees,
        "workers": cfg.grn_workers,
        "seed": cfg.grn_seed,
        "expression_source": "CSV raw count columns",
        "tf_policy": "all_HVGs_due_to_no_external_TF_file",
    }
    return vim, edges, metadata


def normalize_nonnegative_adjacency(matrix: sp.spmatrix) -> sp.csr_matrix:
    adjacency = matrix.tocsr().astype(float)
    if adjacency.nnz:
        adjacency.data = np.maximum(np.nan_to_num(adjacency.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        adjacency.eliminate_zeros()
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return (sp.diags(inverse, format="csr") @ adjacency).tocsr()


def prepare_grn_adjacency(
    edges: pd.DataFrame,
    expression_genes: Sequence[str],
    *,
    top_k_targets: int,
) -> tuple[list[str], sp.csr_matrix, pd.DataFrame]:
    required = {"regulator", "target", "weight"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"GRN edges are missing columns: {sorted(missing)}")
    expression_gene_set = set(map(str, expression_genes))
    work = edges.loc[:, ["regulator", "target", "weight"]].copy()
    work["regulator"] = work["regulator"].astype(str)
    work["target"] = work["target"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").abs()
    work = work.dropna(subset=["weight"])
    work = work.loc[
        (work["weight"] > 0)
        & work["regulator"].isin(expression_gene_set)
        & work["target"].isin(expression_gene_set)
    ]
    work = (
        work.groupby(["regulator", "target"], as_index=False, sort=False)["weight"]
        .sum()
        .sort_values(["regulator", "weight", "target"], ascending=[True, False, True], kind="mergesort")
        .groupby("regulator", group_keys=False, sort=False)
        .head(top_k_targets)
        .reset_index(drop=True)
    )
    if work.empty:
        raise ValueError("No usable positive GRN edges remain after alignment.")
    genes = natural_sort(set(work["regulator"]) | set(work["target"]))
    lookup = {gene: index for index, gene in enumerate(genes)}
    rows = work["regulator"].map(lookup).to_numpy(dtype=int)
    cols = work["target"].map(lookup).to_numpy(dtype=int)
    matrix = sp.coo_matrix((work["weight"].to_numpy(dtype=float), (rows, cols)), shape=(len(genes), len(genes)))
    return genes, normalize_nonnegative_adjacency(matrix), work


def align_expression_to_genes(expression: ExpressionMatrix, genes: Sequence[str]) -> np.ndarray:
    lookup = {gene: index for index, gene in enumerate(expression.genes)}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise ValueError(f"Expression is missing {len(missing)} GRN genes; examples: {missing[:5]}")
    columns = [lookup[gene] for gene in genes]
    return expression.counts[:, columns].toarray().astype(float, copy=False)


def double_end_grn_state(expression: np.ndarray, adjacency: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.nan_to_num(np.asarray(expression, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    matrix = adjacency.tocsr().astype(float)
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError(f"GRN adjacency {matrix.shape} does not match expression {values.shape}.")
    regulator_program = np.asarray(matrix @ values.T).T
    target_program = np.asarray(matrix.T @ values.T).T
    return values * regulator_program, values * target_program


def deterministic_projection_matrix(genes: Sequence[str], role: str, output_dim: int, seed: int) -> np.ndarray:
    if role not in {"reg", "tar"}:
        raise ValueError("role must be 'reg' or 'tar'.")
    scale = 1.0 / math.sqrt(float(output_dim))
    rows: list[np.ndarray] = []
    for gene in map(str, genes):
        digest = hashlib.sha256(f"{seed}\0{role}\0{gene}".encode("utf-8")).digest()
        row_seed = int.from_bytes(digest[:16], byteorder="little", signed=False)
        rows.append(np.random.default_rng(row_seed).standard_normal(output_dim) * scale)
    return np.vstack(rows) if rows else np.zeros((0, output_dim), dtype=float)


def projected_grn_state(expression: np.ndarray, adjacency: sp.spmatrix, genes: Sequence[str], output_dim: int, seed: int) -> np.ndarray:
    regulator_state, target_state = double_end_grn_state(expression, adjacency)
    reg_projection = deterministic_projection_matrix(genes, "reg", output_dim, seed)
    tar_projection = deterministic_projection_matrix(genes, "tar", output_dim, seed)
    result = regulator_state @ reg_projection + target_state @ tar_projection
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def build_stage_network(
    expression: ExpressionMatrix,
    cci: sp.spmatrix,
    edges: pd.DataFrame,
    cfg: PipelineConfig,
) -> StageNetwork:
    if cci.shape != (len(expression.spot_ids), len(expression.spot_ids)):
        raise ValueError(f"CCI shape {cci.shape} does not match {len(expression.spot_ids)} spots.")
    gene_units, gene_adjacency, _ = prepare_grn_adjacency(
        edges,
        expression.genes,
        top_k_targets=cfg.grn_topk_targets,
    )
    aligned_expression = align_expression_to_genes(expression, gene_units)
    state = projected_grn_state(
        aligned_expression,
        gene_adjacency,
        gene_units,
        output_dim=cfg.grn_state_dim,
        seed=cfg.grn_projection_seed,
    )
    return StageNetwork(
        stage=expression.stage,
        gene_units=gene_units,
        gene_adjacency=gene_adjacency,
        spot_units=list(expression.spot_ids),
        spot_cci_adjacency=sp.csr_matrix(cci),
        spot_grn_state=state,
    )


def pairwise_joint_nmf(
    source: np.ndarray,
    target: np.ndarray,
    *,
    components: int,
    max_iter: int,
    seed: int,
    eps: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.maximum(np.nan_to_num(np.asarray(source, dtype=float)), 0.0)
    target = np.maximum(np.nan_to_num(np.asarray(target, dtype=float)), 0.0)
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"Pairwise joint NMF needs equal column counts, got {source.shape} and {target.shape}.")
    rng = np.random.default_rng(seed)
    w_source = rng.random((source.shape[0], components)) + eps
    w_target = rng.random((target.shape[0], components)) + eps
    h_matrix = rng.random((components, source.shape[1])) + eps
    for _ in range(max_iter):
        w_source *= (source @ h_matrix.T) / (w_source @ h_matrix @ h_matrix.T + eps)
        w_target *= (target @ h_matrix.T) / (w_target @ h_matrix @ h_matrix.T + eps)
        w_source = np.maximum(np.nan_to_num(w_source, nan=eps, posinf=eps, neginf=eps), eps)
        w_target = np.maximum(np.nan_to_num(w_target, nan=eps, posinf=eps, neginf=eps), eps)
        numerator = w_source.T @ source + w_target.T @ target
        denominator = (w_source.T @ w_source + w_target.T @ w_target) @ h_matrix + eps
        h_matrix *= numerator / denominator
        h_matrix = np.maximum(np.nan_to_num(h_matrix, nan=eps, posinf=eps, neginf=eps), eps)
    return w_source, w_target


def shared_core_directed_nmf(
    source: np.ndarray,
    target: np.ndarray,
    *,
    components: int,
    max_iter: int,
    seed: int,
    eps: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.maximum(np.nan_to_num(np.asarray(source, dtype=float)), 0.0)
    target = np.maximum(np.nan_to_num(np.asarray(target, dtype=float)), 0.0)
    if source.shape[0] != source.shape[1] or target.shape[0] != target.shape[1]:
        raise ValueError("Shared-core directed NMF requires square adjacency matrices.")
    rng = np.random.default_rng(seed)
    u_source = rng.random((source.shape[0], components)) + eps
    v_source = rng.random((source.shape[0], components)) + eps
    u_target = rng.random((target.shape[0], components)) + eps
    v_target = rng.random((target.shape[0], components)) + eps
    core = rng.random((components, components)) + eps
    for _ in range(max_iter):
        source_vtv = v_source.T @ v_source
        source_utu = u_source.T @ u_source
        target_vtv = v_target.T @ v_target
        target_utu = u_target.T @ u_target
        u_source *= (source @ v_source @ core.T) / (u_source @ core @ source_vtv @ core.T + eps)
        u_target *= (target @ v_target @ core.T) / (u_target @ core @ target_vtv @ core.T + eps)
        u_source = np.maximum(np.nan_to_num(u_source, nan=eps, posinf=eps, neginf=eps), eps)
        u_target = np.maximum(np.nan_to_num(u_target, nan=eps, posinf=eps, neginf=eps), eps)
        source_utu = u_source.T @ u_source
        target_utu = u_target.T @ u_target
        v_source *= (source.T @ u_source @ core) / (v_source @ core.T @ source_utu @ core + eps)
        v_target *= (target.T @ u_target @ core) / (v_target @ core.T @ target_utu @ core + eps)
        v_source = np.maximum(np.nan_to_num(v_source, nan=eps, posinf=eps, neginf=eps), eps)
        v_target = np.maximum(np.nan_to_num(v_target, nan=eps, posinf=eps, neginf=eps), eps)
        source_vtv = v_source.T @ v_source
        target_vtv = v_target.T @ v_target
        numerator = u_source.T @ source @ v_source + u_target.T @ target @ v_target
        denominator = source_utu @ core @ source_vtv + target_utu @ core @ target_vtv + eps
        core *= numerator / denominator
        core = np.maximum(np.nan_to_num(core, nan=eps, posinf=eps, neginf=eps), eps)
    return np.hstack([u_source, v_source]), np.hstack([u_target, v_target])


def pairwise_zscore(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"Cannot z-score different feature dimensions: {source.shape}, {target.shape}")
    stacked = np.vstack([source, target])
    means = np.mean(stacked, axis=0, keepdims=True)
    standard_deviations = np.std(stacked, axis=0, keepdims=True)
    source_out = np.divide(source - means, standard_deviations, out=np.zeros_like(source), where=standard_deviations > 0)
    target_out = np.divide(target - means, standard_deviations, out=np.zeros_like(target), where=standard_deviations > 0)
    return source_out, target_out


def safe_row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    out = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    zero_rows = np.squeeze(row_sums <= 0, axis=1)
    if np.any(zero_rows) and values.shape[1] > 0:
        out[zero_rows, :] = 1.0 / values.shape[1]
    return out


def row_softmax_features(features: np.ndarray, beta: float) -> np.ndarray:
    values = np.asarray(features, dtype=float) / float(beta)
    values -= np.max(values, axis=1, keepdims=True)
    return safe_row_normalize(np.exp(np.clip(values, -700.0, 700.0)))


def pairwise_feature_kl(
    source: np.ndarray,
    target: np.ndarray,
    *,
    beta: float,
    eps: float = 1.0e-12,
    block_size: int = 512,
) -> np.ndarray:
    source_probability = np.clip(row_softmax_features(source, beta), eps, 1.0)
    target_probability = np.clip(row_softmax_features(target, beta), eps, 1.0)
    if source_probability.shape[1] != target_probability.shape[1]:
        raise ValueError("KL feature dimensions differ.")
    log_source = np.log(source_probability)
    log_target = np.log(target_probability)
    source_entropy = np.sum(source_probability * log_source, axis=1, keepdims=True)
    output = np.empty((source_probability.shape[0], target_probability.shape[0]), dtype=float)
    for start in range(0, source_probability.shape[0], max(1, block_size)):
        stop = min(start + max(1, block_size), source_probability.shape[0])
        output[start:stop] = source_entropy[start:stop] - source_probability[start:stop] @ log_target.T
    return np.maximum(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def robust_normalize_cost(cost: np.ndarray) -> np.ndarray:
    values = np.asarray(cost, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values[...] = 0.0
        return values
    low, high = np.percentile(finite, [5.0, 95.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        values[...] = 0.0
        return values
    values = (values - low) / (high - low)
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def block_kl_cost(
    n_source: np.ndarray,
    n_target: np.ndarray,
    g_source: np.ndarray,
    g_target: np.ndarray,
    *,
    beta: float,
    weight_n: float,
    weight_g: float,
) -> np.ndarray:
    n_cost = pairwise_feature_kl(n_source, n_target, beta=beta)
    if weight_g == 0:
        return n_cost
    g_cost = pairwise_feature_kl(g_source, g_target, beta=beta)
    if n_cost.shape != g_cost.shape:
        raise ValueError(f"N and GRN KL shapes differ: {n_cost.shape}, {g_cost.shape}")
    if weight_n == 0:
        return g_cost
    return weight_n * robust_normalize_cost(n_cost) + weight_g * robust_normalize_cost(g_cost)


def pij_from_cost(cost: np.ndarray, temperature: float) -> np.ndarray:
    kernel = np.exp(-np.nan_to_num(cost, nan=np.inf, posinf=np.inf, neginf=0.0) / float(temperature))
    return safe_row_normalize(kernel)


def effective_information(pij: np.ndarray, eps: float = 1.0e-12) -> float:
    probability = safe_row_normalize(np.asarray(pij, dtype=float).copy())
    if probability.size == 0:
        return 0.0
    marginal = probability.mean(axis=0)
    output_entropy = -np.sum(marginal * np.log2(marginal + eps))
    conditional_entropy = -np.mean(np.sum(probability * np.log2(probability + eps), axis=1))
    return float(output_entropy - conditional_entropy)


def calculate_pair(
    source: StageNetwork,
    target: StageNetwork,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    seed = cfg.nmf_seed + int(round(float(source.stage) * 10)) * 1009 + int(round(float(target.stage) * 10))
    gene_source_dense = source.gene_adjacency.toarray().astype(float, copy=False)
    gene_target_dense = target.gene_adjacency.toarray().astype(float, copy=False)
    if gene_source_dense.shape[1] != gene_target_dense.shape[1]:
        raise ValueError(
            "Gene GRN node counts differ between stages. The project compare_N pairwise joint NMF "
            f"requires equal counts, got {gene_source_dense.shape} and {gene_target_dense.shape}."
        )
    gene_n_source, gene_n_target = pairwise_joint_nmf(
        gene_source_dense,
        gene_target_dense,
        components=cfg.nmf_components,
        max_iter=cfg.nmf_max_iter,
        seed=seed,
    )
    gene_n_source, gene_n_target = pairwise_zscore(gene_n_source, gene_n_target)
    gene_cost = pairwise_feature_kl(gene_n_source, gene_n_target, beta=cfg.pij_entropy_epsilon)
    gene_pij = pij_from_cost(gene_cost, cfg.pij_temperature)

    spot_n_source, spot_n_target = shared_core_directed_nmf(
        source.spot_cci_adjacency.toarray().astype(float, copy=False),
        target.spot_cci_adjacency.toarray().astype(float, copy=False),
        components=cfg.nmf_components,
        max_iter=cfg.nmf_max_iter,
        seed=seed,
    )
    spot_n_source, spot_n_target = pairwise_zscore(spot_n_source, spot_n_target)
    spot_g_source, spot_g_target = pairwise_zscore(source.spot_grn_state, target.spot_grn_state)
    spot_cost = block_kl_cost(
        spot_n_source,
        spot_n_target,
        spot_g_source,
        spot_g_target,
        beta=cfg.pij_entropy_epsilon,
        weight_n=cfg.kl_block_weight_n,
        weight_g=cfg.kl_block_weight_g,
    )
    spot_pij = pij_from_cost(spot_cost, cfg.pij_temperature)
    ei_gene = effective_information(gene_pij)
    ei_spot = effective_information(spot_pij)
    row = {
        "network_method": "light_cci_grn",
        "pij_method": "compare_N_kl",
        "organ": "heart",
        "lower_layer": "gene",
        "upper_layer": "spot",
        "time_pair": f"{source.stage}->{target.stage}",
        "EI_gene": ei_gene,
        "EI_spot": ei_spot,
        "deltaEI": ei_spot - ei_gene,
        "EI_lower": ei_gene,
        "EI_upper": ei_spot,
        "EI_gain": ei_spot - ei_gene,
        "gene_pij_shape": list(gene_pij.shape),
        "spot_pij_shape": list(spot_pij.shape),
        "gene_pij_row_sum_max_error": float(np.max(np.abs(gene_pij.sum(axis=1) - 1.0))),
        "spot_pij_row_sum_max_error": float(np.max(np.abs(spot_pij.sum(axis=1) - 1.0))),
    }
    return gene_pij, spot_pij, row


def save_cci_stage(stage_dir: Path, units: Sequence[str], matrix: sp.spmatrix, ligrec: pd.DataFrame, summary: dict[str, object]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_sparse_new(stage_dir / "CCI_total.npz", matrix)
    write_dataframe_new(stage_dir / "units.csv", pd.DataFrame({"spot_id": list(units)}))
    write_dataframe_new(stage_dir / "ligand_receptor_pairs.csv", ligrec)
    write_json_new(stage_dir / "summary.json", summary)


def load_cci_stage(stage_dir: Path) -> tuple[list[str], sp.csr_matrix]:
    units = pd.read_csv(stage_dir / "units.csv")["spot_id"].astype(str).tolist()
    matrix = load_npz(stage_dir / "CCI_total.npz").tocsr()
    return units, matrix


def cci_stage_complete(stage_dir: Path) -> bool:
    return all((stage_dir / name).exists() for name in ("CCI_total.npz", "units.csv", "ligand_receptor_pairs.csv", "summary.json"))


def save_grn_stage(stage_dir: Path, vim: np.ndarray, edges: pd.DataFrame, summary: dict[str, object]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_array_new(stage_dir / "grn_vim.npy", vim)
    write_dataframe_new(stage_dir / "grn_edges.csv", edges)
    write_json_new(stage_dir / "summary.json", summary)


def grn_stage_complete(stage_dir: Path) -> bool:
    return all((stage_dir / name).exists() for name in ("grn_vim.npy", "grn_edges.csv", "summary.json"))


def save_network_stage(stage_dir: Path, network: StageNetwork) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_sparse_new(stage_dir / "gene_adjacency.npz", network.gene_adjacency)
    write_dataframe_new(stage_dir / "gene_units.csv", pd.DataFrame({"gene": network.gene_units}))
    save_sparse_new(stage_dir / "spot_cci_adjacency.npz", network.spot_cci_adjacency)
    write_dataframe_new(stage_dir / "spot_units.csv", pd.DataFrame({"spot_id": network.spot_units}))
    save_array_new(stage_dir / "spot_grn_state.npy", network.spot_grn_state)
    write_json_new(
        stage_dir / "summary.json",
        {
            "stage": network.stage,
            "gene_adjacency_shape": list(network.gene_adjacency.shape),
            "gene_adjacency_nnz": int(network.gene_adjacency.nnz),
            "spot_cci_shape": list(network.spot_cci_adjacency.shape),
            "spot_cci_nnz": int(network.spot_cci_adjacency.nnz),
            "spot_grn_state_shape": list(network.spot_grn_state.shape),
        },
    )


def network_stage_complete(stage_dir: Path) -> bool:
    return all(
        (stage_dir / name).exists()
        for name in (
            "gene_adjacency.npz",
            "gene_units.csv",
            "spot_cci_adjacency.npz",
            "spot_units.csv",
            "spot_grn_state.npy",
            "summary.json",
        )
    )


def load_network_stage(stage: str, stage_dir: Path) -> StageNetwork:
    return StageNetwork(
        stage=stage,
        gene_units=pd.read_csv(stage_dir / "gene_units.csv")["gene"].astype(str).tolist(),
        gene_adjacency=load_npz(stage_dir / "gene_adjacency.npz").tocsr(),
        spot_units=pd.read_csv(stage_dir / "spot_units.csv")["spot_id"].astype(str).tolist(),
        spot_cci_adjacency=load_npz(stage_dir / "spot_cci_adjacency.npz").tocsr(),
        spot_grn_state=np.load(stage_dir / "spot_grn_state.npy"),
    )


def make_run_directory(input_dir: Path, output_root: Path | None, run_dir: Path | None) -> tuple[Path, bool]:
    if run_dir is not None:
        resolved = run_dir.resolve()
        if resolved.exists():
            if not resolved.is_dir():
                raise NotADirectoryError(resolved)
            return resolved, True
        resolved.mkdir(parents=True, exist_ok=False)
        return resolved, False
    root = (output_root or input_dir / "light_cci_grn_compare_N_kl").resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = f"run_{timestamp}" if suffix == 0 else f"run_{timestamp}_{suffix:03d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate, False
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a new run directory under {root}.")


def save_pair_outputs(
    pair_dir: Path,
    source: StageNetwork,
    target: StageNetwork,
    gene_pij: np.ndarray,
    spot_pij: np.ndarray,
    summary: dict[str, object],
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=False)
    save_sparse_new(pair_dir / "pij_gene.npz", sp.csr_matrix(gene_pij))
    save_sparse_new(pair_dir / "pij_spot.npz", sp.csr_matrix(spot_pij))
    write_dataframe_new(pair_dir / "gene_units_source.csv", pd.DataFrame({"gene": source.gene_units}))
    write_dataframe_new(pair_dir / "gene_units_target.csv", pd.DataFrame({"gene": target.gene_units}))
    write_dataframe_new(pair_dir / "spot_units_source.csv", pd.DataFrame({"spot_id": source.spot_units}))
    write_dataframe_new(pair_dir / "spot_units_target.csv", pd.DataFrame({"spot_id": target.spot_units}))
    write_json_new(pair_dir / "summary.json", summary)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        stages=tuple(args.stages),
        csv_chunk_rows=args.csv_chunk_rows,
        cci_workers=args.cci_workers,
        cci_lr_chunk_size=args.cci_lr_chunk_size,
        cci_distance_threshold=args.cci_distance_threshold,
        grn_workers=args.grn_workers,
        grn_n_trees=args.grn_n_trees,
        grn_top_hvg=args.grn_top_hvg,
        grn_top_edge_count=args.grn_top_edge_count,
        grn_topk_targets=args.grn_topk_targets,
        grn_state_dim=args.grn_state_dim,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        kl_block_weight_n=args.kl_block_weight_n,
        kl_block_weight_g=args.kl_block_weight_g,
        pij_entropy_epsilon=args.pij_entropy_epsilon,
        pij_temperature=args.pij_temperature,
    )
    config.validate()
    return config


def run_dry_run(args: argparse.Namespace, config: PipelineConfig, csvs: dict[str, Path]) -> int:
    log("Quickly inspecting CSV headers and sample rows.")
    input_rows = [inspect_expression_csv(csvs[stage]) | {"stage": stage} for stage in config.stages]
    print(pd.DataFrame(input_rows).to_string(index=False))
    log("Checking full-run dependencies.")
    dependency_rows, ready = dependency_report(args.commot_path)
    print(pd.DataFrame(dependency_rows).to_string(index=False))
    payload = {
        "script_version": SCRIPT_VERSION,
        "input_ready": True,
        "dependencies_ready": ready,
        "planned_time_pairs": [f"{config.stages[left]}->{config.stages[right]}" for left, right in all_time_pairs(config.stages)],
        "note": "Dry-run does not create output files or start COMMOT/GRN/NMF.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ready else 2


def run_pipeline(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    input_dir = args.input_dir.resolve()
    csvs = discover_expression_csvs(input_dir, config.stages)
    if args.dry_run:
        return run_dry_run(args, config, csvs)

    dependency_rows, ready = dependency_report(args.commot_path)
    if not ready:
        raise RuntimeError("Full-run dependencies are not ready:\n" + pd.DataFrame(dependency_rows).to_string(index=False))
    ct = load_commot(args.commot_path)
    run_dir, resumed = make_run_directory(input_dir, args.output_root, args.run_dir)
    log(f"{'Resuming' if resumed else 'Created'} run directory: {run_dir}")
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        write_json_new(
            config_path,
            {
                "script_version": SCRIPT_VERSION,
                "command": " ".join(sys.argv),
                "input_dir": str(input_dir),
                "run_dir": str(run_dir),
                "config": asdict(config),
                "deltaEI_definition": "EI_spot(light_cci_grn) - EI_gene(GRN)",
            },
        )

    networks: list[StageNetwork] = []
    for stage in config.stages:
        log(f"========== Stage {stage} ==========")
        network_dir = run_dir / "network" / stage
        if network_stage_complete(network_dir):
            log(f"Stage {stage}: loading completed network checkpoint.")
            networks.append(load_network_stage(stage, network_dir))
            continue
        expression = read_expression_csv(csvs[stage], stage, chunk_rows=config.csv_chunk_rows)

        cci_dir = run_dir / "cci" / stage
        if cci_stage_complete(cci_dir):
            cci_units, cci_matrix = load_cci_stage(cci_dir)
            if cci_units != expression.spot_ids:
                raise ValueError(f"Stage {stage}: CCI checkpoint unit order differs from expression CSV.")
        else:
            if cci_dir.exists() and any(cci_dir.iterdir()):
                raise RuntimeError(f"Stage {stage}: incomplete CCI checkpoint exists; refusing to overwrite: {cci_dir}")
            cci_matrix, ligrec, cci_summary = infer_cci(expression, config, ct)
            save_cci_stage(cci_dir, expression.spot_ids, cci_matrix, ligrec, cci_summary)

        grn_dir = run_dir / "grn" / stage
        if grn_stage_complete(grn_dir):
            edges = pd.read_csv(grn_dir / "grn_edges.csv")
        else:
            if grn_dir.exists() and any(grn_dir.iterdir()):
                raise RuntimeError(f"Stage {stage}: incomplete GRN checkpoint exists; refusing to overwrite: {grn_dir}")
            vim, edges, grn_summary = infer_grn(expression, config)
            save_grn_stage(grn_dir, vim, edges, grn_summary)

        if network_dir.exists() and any(network_dir.iterdir()):
            raise RuntimeError(f"Stage {stage}: incomplete network checkpoint exists; refusing to overwrite: {network_dir}")
        network = build_stage_network(expression, cci_matrix, edges, config)
        save_network_stage(network_dir, network)
        networks.append(network)

    rows: list[dict[str, object]] = []
    for source_index, target_index in all_time_pairs(config.stages):
        source, target = networks[source_index], networks[target_index]
        pair_name = f"{source.stage}_to_{target.stage}"
        pair_dir = run_dir / "pij" / pair_name
        if pair_dir.exists():
            summary_path = pair_dir / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Incomplete Pij checkpoint exists; refusing to overwrite: {pair_dir}")
            with summary_path.open("r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
            continue
        log(f"Calculating compare_N_kl Pij: {source.stage}->{target.stage}")
        gene_pij, spot_pij, summary = calculate_pair(source, target, config)
        save_pair_outputs(pair_dir, source, target, gene_pij, spot_pij, summary)
        rows.append(summary)

    metrics = pd.DataFrame(rows)
    metrics_path = run_dir / "deltaEI.csv"
    if not metrics_path.exists():
        write_dataframe_new(metrics_path, metrics)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        write_json_new(
            summary_path,
            {
                "status": "complete",
                "script_version": SCRIPT_VERSION,
                "run_dir": str(run_dir),
                "stages": list(config.stages),
                "time_pairs": metrics["time_pair"].tolist(),
                "deltaEI_file": str(metrics_path),
            },
        )
    log(f"Complete. deltaEI table: {metrics_path}")
    print(metrics.loc[:, ["time_pair", "EI_gene", "EI_spot", "deltaEI"]].to_string(index=False))
    return 0


def _synthetic_edges(genes: Sequence[str], shift: float) -> pd.DataFrame:
    rows = []
    for index, regulator in enumerate(genes):
        for offset in (1, 2):
            rows.append(
                {
                    "regulator": regulator,
                    "target": genes[(index + offset) % len(genes)],
                    "weight": 1.0 + shift + index * 0.1 + offset * 0.05,
                }
            )
    return pd.DataFrame(rows)


def self_test_command(_: argparse.Namespace) -> int:
    config = PipelineConfig(
        stages=("11.5", "12.5"),
        cci_workers=1,
        grn_workers=1,
        grn_n_trees=3,
        grn_top_hvg=5,
        grn_top_edge_count=100,
        grn_topk_targets=2,
        grn_state_dim=4,
        nmf_components=2,
        nmf_max_iter=8,
        kl_block_weight_n=0.5,
        kl_block_weight_g=0.5,
        pij_entropy_epsilon=0.5,
    )
    genes = ["A", "B", "C", "D", "E"]
    counts_a = sp.csr_matrix(
        np.array(
            [
                [5, 0, 2, 1, 0],
                [1, 4, 0, 2, 1],
                [0, 1, 5, 0, 2],
                [2, 0, 1, 4, 1],
            ],
            dtype=float,
        )
    )
    counts_b = sp.csr_matrix(
        np.array(
            [
                [4, 1, 3, 0, 1],
                [0, 5, 1, 2, 0],
                [1, 0, 4, 1, 3],
            ],
            dtype=float,
        )
    )
    expr_a = ExpressionMatrix("11.5", Path("synthetic_a.csv"), ["a1", "a2", "a3", "a4"], genes, np.zeros((4, 2)), counts_a)
    expr_b = ExpressionMatrix("12.5", Path("synthetic_b.csv"), ["b1", "b2", "b3"], genes, np.zeros((3, 2)), counts_b)
    cci_a = sp.csr_matrix(np.array([[1, 2, 0, 1], [0.5, 1, 3, 0], [1, 0, 1, 2], [0, 2, 0.5, 1]], dtype=float))
    cci_b = sp.csr_matrix(np.array([[1, 1.5, 0], [0.3, 1, 2], [1.2, 0.2, 1]], dtype=float))
    network_a = build_stage_network(expr_a, cci_a, _synthetic_edges(genes, 0.0), config)
    network_b = build_stage_network(expr_b, cci_b, _synthetic_edges(genes, 0.2), config)
    gene_pij, spot_pij, summary = calculate_pair(network_a, network_b, config)
    assertions = {
        "gene_pij_finite": bool(np.all(np.isfinite(gene_pij))),
        "spot_pij_finite": bool(np.all(np.isfinite(spot_pij))),
        "gene_pij_nonnegative": bool(np.all(gene_pij >= 0)),
        "spot_pij_nonnegative": bool(np.all(spot_pij >= 0)),
        "gene_rows_stochastic": bool(np.allclose(gene_pij.sum(axis=1), 1.0)),
        "spot_rows_stochastic": bool(np.allclose(spot_pij.sum(axis=1), 1.0)),
        "deltaEI_finite": bool(np.isfinite(summary["deltaEI"])),
    }
    if not all(assertions.values()):
        raise AssertionError(f"Self-test failed: {assertions}")
    print(json.dumps({"status": "passed", "assertions": assertions, "metrics": summary}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Export count+spatial expression CSV files from heart h5ad files.")
    extract_parser.add_argument("--h5ad-dir", type=Path, required=True)
    extract_parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    extract_parser.add_argument("--chunk-rows", type=int, default=64)
    extract_parser.set_defaults(handler=extract_command)

    run_parser = subparsers.add_parser("run", help="Run or dry-run the complete standalone pipeline.")
    run_parser.add_argument("--input-dir", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, default=None)
    run_parser.add_argument("--run-dir", type=Path, default=None, help="Explicit new or resumable run directory.")
    run_parser.add_argument("--commot-path", type=Path, default=None, help="Optional official COMMOT source checkout.")
    run_parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--csv-chunk-rows", type=int, default=128)
    run_parser.add_argument("--cci-workers", type=int, default=min(8, os.cpu_count() or 1))
    run_parser.add_argument("--cci-lr-chunk-size", type=int, default=1)
    run_parser.add_argument("--cci-distance-threshold", type=float, default=200.0)
    run_parser.add_argument("--grn-workers", type=int, default=min(32, os.cpu_count() or 1))
    run_parser.add_argument("--grn-n-trees", type=int, default=500)
    run_parser.add_argument("--grn-top-hvg", type=int, default=2_000)
    run_parser.add_argument("--grn-top-edge-count", type=int, default=500_000)
    run_parser.add_argument("--grn-topk-targets", type=int, default=50)
    run_parser.add_argument("--grn-state-dim", type=int, default=64)
    run_parser.add_argument("--nmf-components", type=int, default=5)
    run_parser.add_argument("--nmf-max-iter", type=int, default=300)
    run_parser.add_argument("--kl-block-weight-n", type=float, default=0.5)
    run_parser.add_argument("--kl-block-weight-g", type=float, default=0.5)
    run_parser.add_argument("--pij-entropy-epsilon", type=float, default=0.05)
    run_parser.add_argument("--pij-temperature", type=float, default=1.0)
    run_parser.set_defaults(handler=run_pipeline)

    self_test_parser = subparsers.add_parser("self-test", help="Run a fast synthetic end-to-end math smoke test.")
    self_test_parser.set_defaults(handler=self_test_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        log("Interrupted by user; completed files were left untouched.")
        return 130
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
