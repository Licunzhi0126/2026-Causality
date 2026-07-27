#!/usr/bin/env python3
"""Official COMMOT-based CCI workflow shared by local/global drivers.

This module intentionally keeps the legacy CellPhoneDB-inspired implementation
untouched in ``CCI_part.py`` and ``CCI_global.py``. New ``*_COMMOT.py``
scripts should use this file to run the official COMMOT workflow and to export
results in a layout that remains readable for downstream MIGNet integration.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, read_h5ad
from scipy import sparse
from scipy.sparse import save_npz


LOGGER = logging.getLogger("CCI_COMMOT")
FLOAT_DTYPE = np.float32
_SCRIPT_PATH = Path(__file__).resolve()
_PARALLEL_WORK: Optional[AnnData] = None
_PARALLEL_CFG: Optional["CommotRunConfig"] = None
_PARALLEL_CT = None


def _default_commot_reference_dir() -> Path:
    """Pick a sensible official COMMOT checkout location for both local and server layouts."""
    candidates = [
        _SCRIPT_PATH.parent / "COMMOT" / "reference" / "COMMOT",
        _SCRIPT_PATH.parent.parent / "reference" / "COMMOT",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_COMMOT_REFERENCE_DIR = _default_commot_reference_dir()


@dataclass(frozen=True)
class CommotRunConfig:
    dataset_dir: Path
    output_dir: Path
    unit_kind: str
    index_column_name: str
    database_name: str = "cellphonedb_v4_mouse"
    lr_database: str = "CellPhoneDB_v4.0"
    species: str = "mouse"
    signaling_type: Optional[str] = None
    heteromeric: bool = True
    heteromeric_rule: str = "min"
    heteromeric_delimiter: str = "_"
    filter_criteria: str = "min_cell_pct"
    min_cell: int = 100
    min_cell_pct: float = 0.05
    normalize_target_sum: float = 1e4
    apply_log1p: bool = True
    pathway_sum: bool = True
    dis_thr: float = 200.0
    cost_type: str = "euc"
    cot_eps_p: float = 1e-1
    cot_eps_mu: Optional[float] = None
    cot_eps_nu: Optional[float] = None
    cot_rho: float = 1e1
    cot_nitermax: int = 10000
    cot_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    smooth: bool = False
    smth_eta: Optional[float] = None
    smth_nu: Optional[float] = None
    smth_kernel: str = "exp"
    auxiliary_suffixes: Tuple[str, ...] = tuple()
    commot_reference_dir: Path = DEFAULT_COMMOT_REFERENCE_DIR


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_environment() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def load_commot(reference_dir: Path):
    """Import official COMMOT, preferring an installed package and falling back to the local reference checkout."""
    try:
        import commot as ct  # type: ignore

        return ct
    except Exception as pkg_exc:
        if reference_dir.exists():
            ref = str(reference_dir)
            if ref not in sys.path:
                sys.path.insert(0, ref)
            try:
                import commot as ct  # type: ignore

                return ct
            except Exception as ref_exc:
                raise ImportError(_format_commot_import_error(reference_dir, ref_exc)) from ref_exc
        raise ImportError(_format_commot_import_error(reference_dir, pkg_exc)) from pkg_exc


def _format_commot_import_error(reference_dir: Path, exc: Exception) -> str:
    return (
        "Unable to import official COMMOT. "
        f"Tried installed package first and local reference checkout at {reference_dir}. "
        f"The current failure is: {type(exc).__name__}: {exc}. "
        "For the provided `Causality` environment, install COMMOT dependencies first, "
        "especially the POT package (`ot`) required by official COMMOT."
    )


def list_h5ad_files(dataset_dir: Path, auxiliary_suffixes: Sequence[str]) -> List[Path]:
    suffixes = tuple(auxiliary_suffixes)
    files = []
    for file_path in sorted(dataset_dir.rglob("*.h5ad")):
        if not file_path.is_file():
            continue
        if suffixes and any(file_path.name.endswith(suf) for suf in suffixes):
            continue
        files.append(file_path)
    if not files:
        LOGGER.warning("No .h5ad files found under %s", dataset_dir)
    return files


def process_dataset_dir(cfg: CommotRunConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ct = load_commot(cfg.commot_reference_dir)
    files = list_h5ad_files(cfg.dataset_dir, cfg.auxiliary_suffixes)
    if not files:
        LOGGER.error("No input files found. Nothing to process.")
        return
    LOGGER.info("COMMOT CCI planned for %d samples under %s", len(files), cfg.dataset_dir)
    for file_path in files:
        process_sample(file_path, cfg, ct)


def process_sample(file_path: Path, cfg: CommotRunConfig, ct) -> None:
    LOGGER.info("Start COMMOT sample: %s", file_path.name)
    t0 = time.perf_counter()

    sample_name, work, df_ligrec = prepare_commot_work(file_path, cfg, ct)

    run_spatial_communication(work, cfg, ct, df_ligrec, pathway_sum=cfg.pathway_sum)
    export_commot_outputs(work, sample_name=sample_name, cfg=cfg)

    dt = time.perf_counter() - t0
    LOGGER.info("COMMOT sample %s finished in %.2fs -> %s", sample_name, dt, cfg.output_dir)


def prepare_commot_work(file_path: Path, cfg: CommotRunConfig, ct) -> Tuple[str, AnnData, pd.DataFrame]:
    adata = read_h5ad(file_path)
    sample_name = file_path.stem
    _validate_adata_inputs(adata, file_path)

    unit_ids = _extract_unit_ids(adata, cfg.unit_kind)
    work = adata.copy()
    work.obs_names = pd.Index(unit_ids)
    work.obsm["spatial"] = _coerce_spatial_matrix(work.obsm["spatial"], expected_rows=work.n_obs)
    work.var_names_make_unique()
    _ensure_nonnegative_expression(work)
    _apply_official_preprocessing(work, cfg)

    df_ligrec = ct.pp.ligand_receptor_database(
        database=cfg.lr_database,
        species=cfg.species,
        heteromeric_delimiter=cfg.heteromeric_delimiter,
        signaling_type=cfg.signaling_type,
    )
    df_ligrec = ct.pp.filter_lr_database(
        df_ligrec,
        work,
        heteromeric=cfg.heteromeric,
        heteromeric_delimiter=cfg.heteromeric_delimiter,
        heteromeric_rule=cfg.heteromeric_rule,
        filter_criteria=cfg.filter_criteria,
        min_cell=cfg.min_cell,
        min_cell_pct=cfg.min_cell_pct,
    )
    if df_ligrec.empty:
        raise ValueError(f"No LR pairs remain after COMMOT filtering for sample {sample_name}.")
    df_ligrec = _normalize_ligrec_table(df_ligrec)
    return sample_name, work, df_ligrec


def _normalize_ligrec_table(df_ligrec: pd.DataFrame) -> pd.DataFrame:
    """Keep the three columns expected by COMMOT spatial_communication.

    Some COMMOT database loaders return an extra annotation/signaling-type
    column. Official COMMOT reads ligand, receptor, and pathway from the first
    three columns, so this helper mirrors that behavior before our parallel
    chunk runner splits the table.
    """
    df = pd.DataFrame(df_ligrec).copy()
    if df.empty:
        return pd.DataFrame(columns=["ligand", "receptor", "pathway"])
    if {"ligand", "receptor", "pathway"}.issubset(set(map(str, df.columns))):
        df = df.loc[:, ["ligand", "receptor", "pathway"]].copy()
    elif df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ["ligand", "receptor", "pathway"]
    else:
        raise ValueError(
            "COMMOT ligand-receptor table must contain at least ligand, receptor, and pathway columns; "
            f"got shape={df.shape}."
        )
    for column in ("ligand", "receptor", "pathway"):
        df[column] = df[column].astype(str)
    return df.drop_duplicates().reset_index(drop=True)


def run_spatial_communication(
    work: AnnData,
    cfg: CommotRunConfig,
    ct,
    df_ligrec: pd.DataFrame,
    pathway_sum: bool,
) -> None:
    ct.tl.spatial_communication(
        work,
        database_name=cfg.database_name,
        df_ligrec=df_ligrec,
        pathway_sum=pathway_sum,
        heteromeric=cfg.heteromeric,
        heteromeric_rule=cfg.heteromeric_rule,
        heteromeric_delimiter=cfg.heteromeric_delimiter,
        dis_thr=cfg.dis_thr,
        cost_type=cfg.cost_type,
        cot_eps_p=cfg.cot_eps_p,
        cot_eps_mu=cfg.cot_eps_mu,
        cot_eps_nu=cfg.cot_eps_nu,
        cot_rho=cfg.cot_rho,
        cot_nitermax=cfg.cot_nitermax,
        cot_weights=cfg.cot_weights,
        smooth=cfg.smooth,
        smth_eta=cfg.smth_eta,
        smth_nu=cfg.smth_nu,
        smth_kernel=cfg.smth_kernel,
        copy=False,
    )


def _progress(iterable: Iterable, total: int, desc: str, unit: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


def _split_ligrec(df_ligrec: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, df_ligrec.shape[0], chunk_size):
        chunks.append(df_ligrec.iloc[start:start + chunk_size, :].copy())
    return chunks


def _parallel_lr_chunk_worker(task: Tuple[int, List[List[str]], str, str]) -> List[Dict[str, object]]:
    global _PARALLEL_WORK, _PARALLEL_CFG, _PARALLEL_CT
    if _PARALLEL_WORK is None or _PARALLEL_CFG is None:
        raise RuntimeError("Parallel COMMOT worker was not initialized with AnnData/config.")

    chunk_id, records, lr_dir_text, sample_name = task
    cfg = _PARALLEL_CFG
    ct = _PARALLEL_CT
    if ct is None:
        ct = load_commot(cfg.commot_reference_dir)

    df_chunk = pd.DataFrame(records, columns=["ligand", "receptor", "pathway"])
    work = _PARALLEL_WORK.copy()
    run_spatial_communication(work, cfg, ct, df_chunk, pathway_sum=False)

    lr_dir = Path(lr_dir_text)
    out_records: List[Dict[str, object]] = []
    for row in df_chunk.itertuples(index=False):
        lr_key = f"{row.ligand}-{row.receptor}"
        obsp_key = f"commot-{cfg.database_name}-{lr_key}"
        if obsp_key not in work.obsp:
            continue
        filename = f"{sample_name}_COMMOT_LR_{_safe_filename(lr_key)}.npz"
        mat = sparse.csr_matrix(work.obsp[obsp_key])
        save_npz(lr_dir / filename, mat)
        out_records.append(
            {
                "chunk_id": int(chunk_id),
                "lr_key": lr_key,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "pathway": row.pathway,
                "obsp_key": obsp_key,
                "filename": filename,
                "nnz": int(mat.nnz),
                "shape_0": int(mat.shape[0]),
                "shape_1": int(mat.shape[1]),
            }
        )
    return out_records


def process_sample_parallel_lr(
    file_path: Path,
    cfg: CommotRunConfig,
    ct,
    inner_workers: int = 64,
    lr_chunk_size: int = 1,
) -> None:
    """Run one COMMOT sample by splitting LR pairs across worker processes.

    This keeps the official COMMOT call for each LR chunk, but changes execution
    from one full-database COMMOT call into many LR-chunk COMMOT calls so a
    single sample can show LR-level progress and use multiple processes.
    """
    global _PARALLEL_WORK, _PARALLEL_CFG, _PARALLEL_CT

    LOGGER.info("Start parallel COMMOT sample: %s", file_path.name)
    t0 = time.perf_counter()
    sample_name, work, df_ligrec = prepare_commot_work(file_path, cfg, ct)

    if inner_workers <= 1 or df_ligrec.shape[0] <= 1:
        run_spatial_communication(work, cfg, ct, df_ligrec, pathway_sum=cfg.pathway_sum)
        export_commot_outputs(work, sample_name=sample_name, cfg=cfg)
        dt = time.perf_counter() - t0
        LOGGER.info("COMMOT sample %s finished in %.2fs -> %s", sample_name, dt, cfg.output_dir)
        return

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_dir = output_dir / f"{sample_name}_COMMOT_by_LR"
    pathway_dir = output_dir / f"{sample_name}_COMMOT_by_pathway"
    shutil.rmtree(lr_dir, ignore_errors=True)
    shutil.rmtree(pathway_dir, ignore_errors=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    pathway_dir.mkdir(parents=True, exist_ok=True)

    chunks = _split_ligrec(df_ligrec, chunk_size=lr_chunk_size)
    max_workers = max(1, min(int(inner_workers), len(chunks)))
    LOGGER.info(
        "Parallel COMMOT sample %s: %d LR pairs, %d chunks, %d workers",
        sample_name,
        df_ligrec.shape[0],
        len(chunks),
        max_workers,
    )

    _PARALLEL_WORK = work
    _PARALLEL_CFG = cfg
    _PARALLEL_CT = ct
    tasks = [
        (idx, chunk.loc[:, ["ligand", "receptor", "pathway"]].astype(str).values.tolist(), str(lr_dir), sample_name)
        for idx, chunk in enumerate(chunks)
    ]

    lr_records: List[Dict[str, object]] = []
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None
    executor_kwargs = {"max_workers": max_workers}
    if ctx is not None:
        executor_kwargs["mp_context"] = ctx
    with ProcessPoolExecutor(**executor_kwargs) as pool:
        futures = [pool.submit(_parallel_lr_chunk_worker, task) for task in tasks]
        for future in _progress(
            as_completed(futures),
            total=len(futures),
            desc=f"{sample_name} COMMOT LR chunks",
            unit="chunk",
        ):
            lr_records.extend(future.result())

    _PARALLEL_WORK = None
    _PARALLEL_CFG = None
    _PARALLEL_CT = None

    export_commot_outputs_from_lr_files(
        work=work,
        sample_name=sample_name,
        cfg=cfg,
        df_ligrec=df_ligrec,
        lr_records=lr_records,
    )

    dt = time.perf_counter() - t0
    LOGGER.info("Parallel COMMOT sample %s finished in %.2fs -> %s", sample_name, dt, cfg.output_dir)


def export_commot_outputs_from_lr_files(
    work: AnnData,
    sample_name: str,
    cfg: CommotRunConfig,
    df_ligrec: pd.DataFrame,
    lr_records: Sequence[Dict[str, object]],
) -> None:
    db = cfg.database_name
    output_dir = cfg.output_dir
    lr_dir = output_dir / f"{sample_name}_COMMOT_by_LR"
    pathway_dir = output_dir / f"{sample_name}_COMMOT_by_pathway"
    output_dir.mkdir(parents=True, exist_ok=True)
    pathway_dir.mkdir(parents=True, exist_ok=True)

    n_units = int(work.n_obs)
    total_mat = sparse.csr_matrix((n_units, n_units), dtype=FLOAT_DTYPE)
    pathway_mats: Dict[str, sparse.csr_matrix] = {}
    sender_cols: List[np.ndarray] = []
    receiver_cols: List[np.ndarray] = []
    sender_names: List[str] = []
    receiver_names: List[str] = []
    normalized_records: List[Dict[str, object]] = []

    for record in sorted(lr_records, key=lambda item: str(item["lr_key"])):
        mat = sparse.load_npz(lr_dir / str(record["filename"])).tocsr().astype(FLOAT_DTYPE)
        total_mat = total_mat + mat
        pathway = str(record["pathway"])
        pathway_mats[pathway] = pathway_mats.get(pathway, sparse.csr_matrix((n_units, n_units), dtype=FLOAT_DTYPE)) + mat
        sender_cols.append(np.asarray(mat.sum(axis=1)).ravel())
        receiver_cols.append(np.asarray(mat.sum(axis=0)).ravel())
        sender_names.append(f"s-{record['lr_key']}")
        receiver_names.append(f"r-{record['lr_key']}")
        normalized_records.append(dict(record))

    save_npz(output_dir / f"{sample_name}_CCI_total.npz", total_mat)
    _write_index_file(output_dir / f"{sample_name}_index.tsv", work.obs_names.tolist(), cfg.index_column_name)

    pathway_records = []
    pathway_sender_cols: List[np.ndarray] = []
    pathway_receiver_cols: List[np.ndarray] = []
    pathway_names = sorted(pathway_mats)
    for pathway in pathway_names:
        mat = pathway_mats[pathway].tocsr()
        filename = f"{sample_name}_COMMOT_pathway_{_safe_filename(pathway)}.npz"
        save_npz(pathway_dir / filename, mat)
        pathway_records.append(
            {
                "pathway": pathway,
                "obsp_key": f"commot-{db}-{pathway}",
                "filename": filename,
                "nnz": int(mat.nnz),
                "shape_0": int(mat.shape[0]),
                "shape_1": int(mat.shape[1]),
            }
        )
        pathway_sender_cols.append(np.asarray(mat.sum(axis=1)).ravel())
        pathway_receiver_cols.append(np.asarray(mat.sum(axis=0)).ravel())
        work.obsp[f"commot-{db}-{pathway}"] = mat

    total_sender = np.asarray(total_mat.sum(axis=1)).ravel()
    total_receiver = np.asarray(total_mat.sum(axis=0)).ravel()
    all_sender_cols = sender_cols + [total_sender] + pathway_sender_cols
    all_receiver_cols = receiver_cols + [total_receiver] + pathway_receiver_cols
    all_sender_names = sender_names + ["s-total-total"] + [f"s-{name}" for name in pathway_names]
    all_receiver_names = receiver_names + ["r-total-total"] + [f"r-{name}" for name in pathway_names]

    sender_df = pd.DataFrame(
        np.column_stack(all_sender_cols) if all_sender_cols else np.empty((n_units, 0)),
        columns=all_sender_names,
        index=work.obs_names.astype(str),
    )
    receiver_df = pd.DataFrame(
        np.column_stack(all_receiver_cols) if all_receiver_cols else np.empty((n_units, 0)),
        columns=all_receiver_names,
        index=work.obs_names.astype(str),
    )
    sender_df.to_csv(output_dir / f"{sample_name}_COMMOT_sender_summary.tsv", sep="\t")
    receiver_df.to_csv(output_dir / f"{sample_name}_COMMOT_receiver_summary.tsv", sep="\t")

    df_ligrec = _normalize_ligrec_table(df_ligrec)
    df_ligrec.to_csv(output_dir / f"{sample_name}_COMMOT_ligrec.tsv", sep="\t", index=False)
    lr_manifest = pd.DataFrame(normalized_records)
    lr_manifest.to_csv(output_dir / f"{sample_name}_COMMOT_lr_pairs.tsv", sep="\t", index=False)
    pathway_manifest = pd.DataFrame(pathway_records)
    pathway_manifest.to_csv(output_dir / f"{sample_name}_COMMOT_pathways.tsv", sep="\t", index=False)

    work.uns[f"commot-{db}-info"] = {
        "df_ligrec": df_ligrec,
        "distance_threshold": cfg.dis_thr,
        "parallel_lr_chunks": True,
    }
    work.obsp[f"commot-{db}-total-total"] = total_mat
    work.obsm[f"commot-{db}-sum-sender"] = sender_df
    work.obsm[f"commot-{db}-sum-receiver"] = receiver_df

    info_payload = {
        "method": "official_commot_parallel_lr_chunks",
        "sample_name": sample_name,
        "unit_kind": cfg.unit_kind,
        "database_name": cfg.database_name,
        "lr_database": cfg.lr_database,
        "species": cfg.species,
        "signaling_type": cfg.signaling_type,
        "distance_threshold": cfg.dis_thr,
        "n_units": int(work.n_obs),
        "n_genes": int(work.n_vars),
        "n_lr_pairs_exported": int(len(lr_manifest)),
        "n_pathways_exported": int(len(pathway_manifest)),
        "sender_summary_file": f"{sample_name}_COMMOT_sender_summary.tsv",
        "receiver_summary_file": f"{sample_name}_COMMOT_receiver_summary.tsv",
        "compat_total_matrix_file": f"{sample_name}_CCI_total.npz",
        "compat_index_file": f"{sample_name}_index.tsv",
        "note": "LR matrices are stored as external npz files; the h5ad stores total and pathway matrices.",
        "config": _json_ready(asdict(cfg)),
    }
    with (output_dir / f"{sample_name}_COMMOT_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info_payload, fh, ensure_ascii=False, indent=2)

    work.write_h5ad(output_dir / f"{sample_name}_COMMOT.h5ad")


def export_commot_outputs(adata: AnnData, sample_name: str, cfg: CommotRunConfig) -> None:
    db = cfg.database_name
    info_key = f"commot-{db}-info"
    sender_key = f"commot-{db}-sum-sender"
    receiver_key = f"commot-{db}-sum-receiver"
    total_key = f"commot-{db}-total-total"

    if info_key not in adata.uns:
        raise KeyError(f"Missing {info_key} in adata.uns after COMMOT run.")
    if total_key not in adata.obsp:
        raise KeyError(f"Missing {total_key} in adata.obsp after COMMOT run.")
    if sender_key not in adata.obsm or receiver_key not in adata.obsm:
        raise KeyError("COMMOT sender/receiver summaries are missing.")

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_dir = output_dir / f"{sample_name}_COMMOT_by_LR"
    pathway_dir = output_dir / f"{sample_name}_COMMOT_by_pathway"
    lr_dir.mkdir(parents=True, exist_ok=True)
    pathway_dir.mkdir(parents=True, exist_ok=True)

    total_mat = sparse.csr_matrix(adata.obsp[total_key])
    save_npz(output_dir / f"{sample_name}_CCI_total.npz", total_mat)
    _write_index_file(output_dir / f"{sample_name}_index.tsv", adata.obs_names.tolist(), cfg.index_column_name)

    sender_df = pd.DataFrame(adata.obsm[sender_key]).copy()
    sender_df.index = adata.obs_names.astype(str)
    sender_df.to_csv(output_dir / f"{sample_name}_COMMOT_sender_summary.tsv", sep="\t")

    receiver_df = pd.DataFrame(adata.obsm[receiver_key]).copy()
    receiver_df.index = adata.obs_names.astype(str)
    receiver_df.to_csv(output_dir / f"{sample_name}_COMMOT_receiver_summary.tsv", sep="\t")

    df_ligrec = _normalize_ligrec_table(pd.DataFrame(adata.uns[info_key]["df_ligrec"]).copy())
    df_ligrec.to_csv(output_dir / f"{sample_name}_COMMOT_ligrec.tsv", sep="\t", index=False)

    lr_manifest = _export_lr_matrices(
        adata=adata,
        sample_name=sample_name,
        output_dir=lr_dir,
        database_name=db,
        df_ligrec=df_ligrec,
    )
    lr_manifest.to_csv(output_dir / f"{sample_name}_COMMOT_lr_pairs.tsv", sep="\t", index=False)

    pathway_manifest = _export_pathway_matrices(
        adata=adata,
        sample_name=sample_name,
        output_dir=pathway_dir,
        database_name=db,
        df_ligrec=df_ligrec,
    )
    pathway_manifest.to_csv(output_dir / f"{sample_name}_COMMOT_pathways.tsv", sep="\t", index=False)

    info_payload = {
        "method": "official_commot",
        "sample_name": sample_name,
        "unit_kind": cfg.unit_kind,
        "database_name": cfg.database_name,
        "lr_database": cfg.lr_database,
        "species": cfg.species,
        "signaling_type": cfg.signaling_type,
        "distance_threshold": cfg.dis_thr,
        "n_units": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_lr_pairs_exported": int(len(lr_manifest)),
        "n_pathways_exported": int(len(pathway_manifest)),
        "sender_summary_file": f"{sample_name}_COMMOT_sender_summary.tsv",
        "receiver_summary_file": f"{sample_name}_COMMOT_receiver_summary.tsv",
        "compat_total_matrix_file": f"{sample_name}_CCI_total.npz",
        "compat_index_file": f"{sample_name}_index.tsv",
        "config": _json_ready(asdict(cfg)),
    }
    with (output_dir / f"{sample_name}_COMMOT_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info_payload, fh, ensure_ascii=False, indent=2)

    adata.write_h5ad(output_dir / f"{sample_name}_COMMOT.h5ad")


def _export_lr_matrices(
    adata: AnnData,
    sample_name: str,
    output_dir: Path,
    database_name: str,
    df_ligrec: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    unique_pairs = df_ligrec.loc[:, ["ligand", "receptor", "pathway"]].drop_duplicates().reset_index(drop=True)
    for row in unique_pairs.itertuples(index=False):
        lr_key = f"{row.ligand}-{row.receptor}"
        obsp_key = f"commot-{database_name}-{lr_key}"
        if obsp_key not in adata.obsp:
            continue
        filename = f"{sample_name}_COMMOT_LR_{_safe_filename(lr_key)}.npz"
        mat = sparse.csr_matrix(adata.obsp[obsp_key])
        save_npz(output_dir / filename, mat)
        records.append(
            {
                "lr_key": lr_key,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "pathway": row.pathway,
                "obsp_key": obsp_key,
                "filename": filename,
                "nnz": int(mat.nnz),
                "shape_0": int(mat.shape[0]),
                "shape_1": int(mat.shape[1]),
            }
        )
    return pd.DataFrame(records)


def _export_pathway_matrices(
    adata: AnnData,
    sample_name: str,
    output_dir: Path,
    database_name: str,
    df_ligrec: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    pathways = sorted({str(x) for x in df_ligrec["pathway"].dropna().astype(str) if str(x).strip()})
    for pathway in pathways:
        obsp_key = f"commot-{database_name}-{pathway}"
        if obsp_key not in adata.obsp:
            continue
        filename = f"{sample_name}_COMMOT_pathway_{_safe_filename(pathway)}.npz"
        mat = sparse.csr_matrix(adata.obsp[obsp_key])
        save_npz(output_dir / filename, mat)
        records.append(
            {
                "pathway": pathway,
                "obsp_key": obsp_key,
                "filename": filename,
                "nnz": int(mat.nnz),
                "shape_0": int(mat.shape[0]),
                "shape_1": int(mat.shape[1]),
            }
        )
    return pd.DataFrame(records)


def _validate_adata_inputs(adata: AnnData, file_path: Path) -> None:
    if "spatial" not in adata.obsm:
        raise KeyError(f"{file_path} is missing obsm['spatial'], which is required by official COMMOT.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{file_path} has empty expression dimensions: {adata.shape}")


def _extract_unit_ids(adata: AnnData, unit_kind: str) -> List[str]:
    if unit_kind == "spot":
        return list(map(str, adata.obs_names))
    if unit_kind != "domain":
        raise ValueError(f"Unsupported unit_kind: {unit_kind}")

    obs = adata.obs.copy()
    if "domain_id" in obs.columns:
        raw_domain_ids = obs["domain_id"].astype(str).tolist()
        if len(raw_domain_ids) == len(set(raw_domain_ids)):
            return raw_domain_ids

    domain_col = "domain_label" if "domain_label" in obs.columns else None
    raw = obs.index.astype(str).tolist() if domain_col is None else obs[domain_col].astype(str).tolist()
    if len(raw) == len(set(raw)):
        return raw
    return _canonicalize_domain_ids(raw, n_domains=adata.n_obs)


def _canonicalize_domain_ids(values: Sequence[str], n_domains: int) -> List[str]:
    parsed: List[int] = []
    has_alpha = False
    for value in values:
        text = str(value).strip()
        if any(ch.isalpha() for ch in text):
            has_alpha = True
        match = re.search(r"(\d+)", text)
        if not match:
            raise ValueError(f"Cannot parse domain number from value: {value!r}")
        parsed.append(int(match.group(1)))

    base = 1
    if not has_alpha and 0 in parsed:
        base = 0
    canonical = []
    for number in parsed:
        one_based = number + 1 if base == 0 else number
        if not (1 <= one_based <= n_domains):
            raise ValueError(f"Domain id {number} normalized to {one_based}, out of range 1..{n_domains}.")
        canonical.append(f"domain_{one_based:03d}")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Canonicalized domain ids are not unique.")
    return canonical


def _coerce_spatial_matrix(spatial_obj, expected_rows: int) -> np.ndarray:
    if isinstance(spatial_obj, pd.DataFrame):
        arr = spatial_obj.to_numpy(dtype=FLOAT_DTYPE, copy=False)
    elif isinstance(spatial_obj, np.ndarray):
        arr = spatial_obj.astype(FLOAT_DTYPE, copy=False)
    else:
        arr = np.asarray(spatial_obj, dtype=FLOAT_DTYPE)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise ValueError(f"Unexpected spatial matrix shape: {arr.shape}, expected ({expected_rows}, n_dims)")
    return arr


def _ensure_nonnegative_expression(adata: AnnData) -> None:
    x = adata.X
    min_value = float(x.min()) if sparse.issparse(x) else float(np.min(x))
    if min_value < 0:
        raise ValueError("Official COMMOT preprocessing expects nonnegative expression values.")


def _apply_official_preprocessing(adata: AnnData, cfg: CommotRunConfig) -> None:
    sc.pp.normalize_total(adata, target_sum=cfg.normalize_target_sum, inplace=True)
    if cfg.apply_log1p:
        sc.pp.log1p(adata)


def _write_index_file(path: Path, unit_ids: Sequence[str], column_name: str) -> None:
    pd.DataFrame({column_name: list(map(str, unit_ids))}).to_csv(path, sep="\t", index=False)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    safe = safe.strip("._")
    return safe or "unnamed"


def _json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    return obj
