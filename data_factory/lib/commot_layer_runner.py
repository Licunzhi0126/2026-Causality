from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad

from factory_common import AUXILIARY_H5AD_SUFFIXES, COMMOT_REFERENCE_DIR, append_csv, ensure_dir, iter_h5ad_files, write_csv


DEFAULT_COMMOT_WORKERS = 64
DEFAULT_LR_CHUNK_SIZE = 1
DEFAULT_HEARTBEAT_SECONDS = 300


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _start_heartbeat(label: str, interval_seconds: int) -> Callable[[], None]:
    if interval_seconds <= 0:
        return lambda: None
    stop = threading.Event()
    t0 = time.perf_counter()

    def loop() -> None:
        while not stop.wait(interval_seconds):
            elapsed = _format_elapsed(time.perf_counter() - t0)
            print(f"[COMMOT heartbeat] {label} still running, elapsed={elapsed}", flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop.set


def _progress(iterable: Iterable, total: int, desc: str, unit: str = "sample"):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


def build_config(
    CommotRunConfig,
    input_root: Path,
    output_root: Path,
    unit_kind: str,
    commot_reference_dir: Path,
    dis_thr: float,
) -> object:
    return CommotRunConfig(
        dataset_dir=input_root,
        output_dir=output_root,
        unit_kind=unit_kind,
        index_column_name="spot_id" if unit_kind == "spot" else "domain_id",
        database_name="cellphonedb_v4_mouse",
        lr_database="CellPhoneDB_v4.0",
        species="mouse",
        signaling_type=None,
        heteromeric=True,
        heteromeric_rule="min",
        heteromeric_delimiter="_",
        filter_criteria="min_cell_pct",
        min_cell_pct=0.05,
        dis_thr=dis_thr,
        pathway_sum=True,
        cost_type="euc",
        cot_eps_p=1e-1,
        cot_rho=1e1,
        cot_nitermax=10000,
        cot_weights=(0.25, 0.25, 0.25, 0.25),
        smooth=False,
        auxiliary_suffixes=AUXILIARY_H5AD_SUFFIXES,
        commot_reference_dir=commot_reference_dir,
    )


def count_units(path: Path) -> int:
    adata = read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def run_commot_layer(
    input_root: Path,
    output_root: Path,
    unit_kind: str,
    manifest_name: str,
    commot_reference_dir: Path = COMMOT_REFERENCE_DIR,
    min_units: int = 2,
    dis_thr: float = 200.0,
    sample_names: Sequence[str] = (),
    workers: int = DEFAULT_COMMOT_WORKERS,
    lr_chunk_size: int = DEFAULT_LR_CHUNK_SIZE,
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
    manifest_root: Path | None = None,
) -> None:
    ensure_dir(output_root)

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
    runnable: List[Tuple[Path, Dict[str, object]]] = []

    for path in files:
        row: Dict[str, object] = {
            "input_file": str(path),
            "sample_name": path.stem,
            "output_dir": str(output_root),
            "unit_kind": unit_kind,
            "status": "planned",
        }
        try:
            n_units = count_units(path)
            row["n_units"] = n_units
            if n_units < min_units:
                row["status"] = "too_few_units_skipped"
                row["reason"] = f"n_units={n_units} < min_units={min_units}"
                rows.append(row)
                skipped.append(row)
                continue
            runnable.append((path, row))
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            skipped.append(row)

    if runnable:
        from CCI_IO_COMMOT import CommotRunConfig, configure_logging, ensure_environment, load_commot, process_sample_parallel_lr

        cfg = build_config(CommotRunConfig, input_root, output_root, unit_kind, commot_reference_dir, dis_thr=dis_thr)
        configure_logging()
        ensure_environment()
        ct = load_commot(commot_reference_dir)
        print(
            f"[COMMOT] Running {len(runnable)} samples sequentially; "
            f"each sample uses up to {int(workers)} LR-chunk worker processes"
        )
        for path, row in _progress(runnable, total=len(runnable), desc="COMMOT samples", unit="sample"):
            stop_heartbeat = _start_heartbeat(path.name, heartbeat_seconds)
            try:
                process_sample_parallel_lr(
                    path,
                    cfg,
                    ct,
                    inner_workers=int(workers),
                    lr_chunk_size=int(lr_chunk_size),
                )
                row["status"] = "written"
            except Exception as exc:
                row["status"] = "error"
                row["reason"] = f"{type(exc).__name__}: {exc}"
                skipped.append(row)
            finally:
                stop_heartbeat()
            rows.append(row)

    manifest_dir = manifest_root if manifest_root is not None else output_root.parents[1] / "manifests"
    manifest = manifest_dir / manifest_name
    write_csv(manifest, rows)
    append_csv(manifest_dir / "skipped_jobs.csv", skipped)
    print(f"[COMMOT] Wrote manifest: {manifest}")


def build_argparser(description: str, default_input: Path, default_output: Path, unit_kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-root", "--dataset-root", type=Path, default=default_input)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--commot-reference-dir", type=Path, default=COMMOT_REFERENCE_DIR)
    parser.add_argument("--unit-kind", choices=("spot", "domain"), default=unit_kind)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--dis-thr", type=float, default=200.0)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_COMMOT_WORKERS,
        help="Number of worker processes used inside one sample across LR chunks. Default: 64.",
    )
    parser.add_argument(
        "--lr-chunk-size",
        type=int,
        default=DEFAULT_LR_CHUNK_SIZE,
        help="Number of ligand-receptor pairs per internal COMMOT chunk. Default: 1 for the most detailed progress bar.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Print a still-running message for each active COMMOT sample every N seconds. Use 0 to disable.",
    )
    return parser
