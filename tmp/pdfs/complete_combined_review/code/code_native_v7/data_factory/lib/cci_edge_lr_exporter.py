from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from factory_common import ORGANS, STAGES, parse_sample_stem


SUPPORTED_LAYERS: Tuple[str, ...] = ("seurat_k150", "seurat_k40", "spot")
LAYER_SAMPLE_PREFIX: Dict[str, str] = {
    "seurat_k150": "seurat150",
    "seurat_k40": "seurat",
    "spot": "spot",
}
LR_MANIFEST_SUFFIX = "_COMMOT_lr_pairs.tsv"
REQUIRED_LR_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "lr_key",
    "ligand",
    "receptor",
    "pathway",
    "filename",
    "nnz",
    "shape_0",
    "shape_1",
)
OUTPUT_COLUMNS: Tuple[str, ...] = (
    "layer",
    "sample",
    "organ",
    "stage",
    "sender",
    "receiver",
    "lr_key",
    "ligand",
    "receptor",
    "pathway",
    "weight",
)
DEFAULT_WORKERS = 64
DEFAULT_CHUNK_ROWS = 250_000
DEFAULT_COMPRESSION = "zstd"


@dataclass(frozen=True)
class ExportJob:
    layer: str
    sample: str
    organ: str
    stage: str
    lr_manifest_path: Path
    index_path: Path
    lr_dir: Path
    output_path: Path
    n_units: int
    n_lr_pairs: int
    expected_rows: int


@dataclass(frozen=True)
class ExportOptions:
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    compression: str = DEFAULT_COMPRESSION
    overwrite: bool = False


@dataclass
class ExportResult:
    layer: str
    sample: str
    organ: str
    stage: str
    status: str
    output_file: str
    expected_rows: int
    exported_rows: int = 0
    n_units: int = 0
    n_lr_pairs: int = 0
    output_bytes: int = 0
    expected_weight_sum: Optional[float] = None
    exported_weight_sum: Optional[float] = None
    elapsed_seconds: float = 0.0
    partial_file: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


ProgressCallback = Callable[[Dict[str, object]], None]


def _validate_requested_layers(layers: Sequence[str]) -> Tuple[str, ...]:
    requested = tuple(dict.fromkeys(map(str, layers)))
    if not requested:
        raise ValueError("At least one layer is required.")
    unsupported = sorted(set(requested) - set(SUPPORTED_LAYERS))
    if unsupported:
        raise ValueError(
            f"Unsupported CCI layers: {unsupported}. Expected only {list(SUPPORTED_LAYERS)}."
        )
    return requested


def expected_sample_names(layer: str) -> Tuple[str, ...]:
    if layer not in LAYER_SAMPLE_PREFIX:
        raise ValueError(f"Unsupported layer: {layer!r}")
    prefix = LAYER_SAMPLE_PREFIX[layer]
    return tuple(f"{prefix}_{organ}_{stage}" for organ in ORGANS for stage in STAGES)


def _read_index(path: Path) -> Tuple[str, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COMMOT index file: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one column in {path}, found {frame.shape[1]}.")
    column = str(frame.columns[0])
    unit_ids = frame.iloc[:, 0].astype(str).tolist()
    if not unit_ids:
        raise ValueError(f"COMMOT index is empty: {path}")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError(f"COMMOT index contains duplicate unit IDs: {path}")
    return column, unit_ids


def _read_lr_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COMMOT LR manifest: {path}")
    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in REQUIRED_LR_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"LR manifest {path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"LR manifest is empty: {path}")
    for column in ("lr_key", "ligand", "receptor", "filename"):
        if frame[column].isna().any():
            raise ValueError(f"LR manifest {path} contains missing values in {column!r}.")
        frame[column] = frame[column].astype(str)
    for column in ("nnz", "shape_0", "shape_1"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if (numeric < 0).any():
            raise ValueError(f"LR manifest {path} contains negative {column!r} values.")
        frame[column] = numeric.astype(np.int64)
    if frame["lr_key"].duplicated().any():
        duplicates = frame.loc[frame["lr_key"].duplicated(), "lr_key"].tolist()
        raise ValueError(f"LR manifest {path} contains duplicate lr_key values: {duplicates[:5]}")
    if frame["filename"].duplicated().any():
        duplicates = frame.loc[frame["filename"].duplicated(), "filename"].tolist()
        raise ValueError(f"LR manifest {path} contains duplicate filenames: {duplicates[:5]}")
    return frame


def discover_export_jobs(
    cci_root: Path,
    output_root: Path,
    layers: Sequence[str] = SUPPORTED_LAYERS,
    sample_names: Sequence[str] = (),
    strict_grid: bool = False,
) -> List[ExportJob]:
    cci_root = Path(cci_root)
    output_root = Path(output_root)
    requested_layers = _validate_requested_layers(layers)
    allowed_samples = set(map(str, sample_names))
    jobs: List[ExportJob] = []

    for layer in requested_layers:
        layer_root = cci_root / layer
        if not layer_root.is_dir():
            raise FileNotFoundError(f"Requested CCI layer directory does not exist: {layer_root}")
        manifests = sorted(layer_root.glob(f"*{LR_MANIFEST_SUFFIX}"))
        discovered_samples = {
            path.name[: -len(LR_MANIFEST_SUFFIX)]
            for path in manifests
        }
        if strict_grid:
            missing = sorted(set(expected_sample_names(layer)) - discovered_samples)
            if missing:
                raise FileNotFoundError(
                    f"Layer {layer!r} is missing {len(missing)} expected organ-stage samples: {missing}"
                )
        if allowed_samples:
            manifests = [
                path
                for path in manifests
                if path.name[: -len(LR_MANIFEST_SUFFIX)] in allowed_samples
            ]
        if not manifests:
            qualifier = f" matching {sorted(allowed_samples)}" if allowed_samples else ""
            raise FileNotFoundError(f"No COMMOT LR manifests found in {layer_root}{qualifier}.")

        for manifest_path in manifests:
            sample = manifest_path.name[: -len(LR_MANIFEST_SUFFIX)]
            organ, stage = parse_sample_stem(sample)
            _, unit_ids = _read_index(layer_root / f"{sample}_index.tsv")
            manifest = _read_lr_manifest(manifest_path)
            shapes = manifest.loc[:, ["shape_0", "shape_1"]].drop_duplicates()
            if len(shapes) != 1:
                raise ValueError(f"LR manifest contains inconsistent matrix shapes: {manifest_path}")
            shape = tuple(map(int, shapes.iloc[0].tolist()))
            expected_shape = (len(unit_ids), len(unit_ids))
            if shape != expected_shape:
                raise ValueError(
                    f"Manifest shape {shape} does not match index-derived shape {expected_shape}: {manifest_path}"
                )
            lr_dir = layer_root / f"{sample}_COMMOT_by_LR"
            if not lr_dir.is_dir():
                raise FileNotFoundError(f"Missing COMMOT LR matrix directory: {lr_dir}")
            missing_files = [name for name in manifest["filename"] if not (lr_dir / name).is_file()]
            if missing_files:
                raise FileNotFoundError(
                    f"Sample {sample!r} is missing {len(missing_files)} LR matrix files; "
                    f"first files: {missing_files[:5]}"
                )
            jobs.append(
                ExportJob(
                    layer=layer,
                    sample=sample,
                    organ=organ,
                    stage=stage,
                    lr_manifest_path=manifest_path,
                    index_path=layer_root / f"{sample}_index.tsv",
                    lr_dir=lr_dir,
                    output_path=output_root / layer / f"{sample}_edge_lr_long.parquet",
                    n_units=len(unit_ids),
                    n_lr_pairs=int(len(manifest)),
                    expected_rows=int(manifest["nnz"].sum()),
                )
            )

    if allowed_samples:
        found = {job.sample for job in jobs}
        missing_requested = sorted(allowed_samples - found)
        if missing_requested:
            raise FileNotFoundError(f"Requested samples were not found in the selected layers: {missing_requested}")
    return sorted(jobs, key=lambda job: (job.layer, job.organ, float(job.stage), job.sample))


def _dictionary_type() -> pa.DictionaryType:
    return pa.dictionary(pa.int32(), pa.string())


def _output_schema(job: ExportJob) -> pa.Schema:
    dictionary = _dictionary_type()
    metadata = {
        b"format": b"commot-edge-lr-long-v1",
        b"layer": job.layer.encode("utf-8"),
        b"sample": job.sample.encode("utf-8"),
        b"organ": job.organ.encode("utf-8"),
        b"stage": job.stage.encode("utf-8"),
        b"direction": b"matrix-row-is-sender;matrix-column-is-receiver",
        b"source_lr_manifest": str(job.lr_manifest_path).encode("utf-8"),
        b"source_index": str(job.index_path).encode("utf-8"),
    }
    return pa.schema(
        [
            pa.field("layer", dictionary, nullable=False),
            pa.field("sample", dictionary, nullable=False),
            pa.field("organ", dictionary, nullable=False),
            pa.field("stage", dictionary, nullable=False),
            pa.field("sender", dictionary, nullable=False),
            pa.field("receiver", dictionary, nullable=False),
            pa.field("lr_key", dictionary, nullable=False),
            pa.field("ligand", dictionary, nullable=False),
            pa.field("receptor", dictionary, nullable=False),
            pa.field("pathway", dictionary, nullable=True),
            pa.field("weight", pa.float64(), nullable=False),
        ],
        metadata=metadata,
    )


def _constant_dictionary_array(
    value: Optional[str],
    length: int,
    zero_indices: pa.Int32Array,
) -> pa.DictionaryArray:
    if value is None:
        return pa.DictionaryArray.from_arrays(
            pa.nulls(length, type=pa.int32()),
            pa.array([], type=pa.string()),
        )
    return pa.DictionaryArray.from_arrays(
        zero_indices,
        pa.array([str(value)], type=pa.string()),
    )


def _nullable_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return None if text.strip().lower() in {"", "nan", "none", "null"} else text


def iter_csr_chunks(
    matrix: sp.csr_matrix,
    chunk_rows: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}.")
    n_matrix_rows = int(matrix.shape[0])
    start_row = 0
    while start_row < n_matrix_rows:
        data_start = int(matrix.indptr[start_row])
        target_end = data_start + chunk_rows
        end_row = int(np.searchsorted(matrix.indptr, target_end, side="right") - 1)
        end_row = min(n_matrix_rows, max(start_row + 1, end_row))
        data_end = int(matrix.indptr[end_row])
        row_counts = np.diff(matrix.indptr[start_row : end_row + 1])
        if data_end > data_start:
            row_indices = np.repeat(
                np.arange(start_row, end_row, dtype=np.int32),
                row_counts,
            )
            column_indices = matrix.indices[data_start:data_end].astype(np.int32, copy=False)
            values = matrix.data[data_start:data_end].astype(np.float64, copy=False)
            yield row_indices, column_indices, values
        start_row = end_row


def _record_batch(
    job: ExportJob,
    schema: pa.Schema,
    unit_dictionary: pa.StringArray,
    lr_record: pd.Series,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    values: np.ndarray,
) -> pa.RecordBatch:
    length = int(values.size)
    zero_indices = pa.array(np.zeros(length, dtype=np.int32), type=pa.int32())
    sender = pa.DictionaryArray.from_arrays(
        pa.array(row_indices, type=pa.int32()),
        unit_dictionary,
    )
    receiver = pa.DictionaryArray.from_arrays(
        pa.array(column_indices, type=pa.int32()),
        unit_dictionary,
    )
    arrays = [
        _constant_dictionary_array(job.layer, length, zero_indices),
        _constant_dictionary_array(job.sample, length, zero_indices),
        _constant_dictionary_array(job.organ, length, zero_indices),
        _constant_dictionary_array(job.stage, length, zero_indices),
        sender,
        receiver,
        _constant_dictionary_array(str(lr_record["lr_key"]), length, zero_indices),
        _constant_dictionary_array(str(lr_record["ligand"]), length, zero_indices),
        _constant_dictionary_array(str(lr_record["receptor"]), length, zero_indices),
        _constant_dictionary_array(_nullable_text(lr_record["pathway"]), length, zero_indices),
        pa.array(values, type=pa.float64()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _emit_progress(progress_queue: Any, event: Dict[str, object]) -> None:
    if progress_queue is not None:
        progress_queue.put(event)


def export_one_job(
    job: ExportJob,
    options: ExportOptions = ExportOptions(),
    progress_queue: Any = None,
) -> ExportResult:
    started = time.perf_counter()
    result = ExportResult(
        layer=job.layer,
        sample=job.sample,
        organ=job.organ,
        stage=job.stage,
        status="running",
        output_file=str(job.output_path),
        expected_rows=job.expected_rows,
        n_units=job.n_units,
        n_lr_pairs=job.n_lr_pairs,
    )
    _emit_progress(
        progress_queue,
        {
            "type": "started",
            "layer": job.layer,
            "sample": job.sample,
            "expected_rows": job.expected_rows,
        },
    )
    writer: Optional[pq.ParquetWriter] = None
    partial_path: Optional[Path] = None
    try:
        if options.chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {options.chunk_rows}.")
        if not pa.Codec.is_available(options.compression):
            raise ValueError(f"PyArrow compression codec is unavailable: {options.compression!r}")
        if job.output_path.exists() and not options.overwrite:
            metadata = pq.read_metadata(job.output_path)
            result.status = "skipped_existing"
            result.exported_rows = int(metadata.num_rows)
            result.output_bytes = int(job.output_path.stat().st_size)
            return result

        _, unit_ids = _read_index(job.index_path)
        manifest = _read_lr_manifest(job.lr_manifest_path)
        if len(unit_ids) != job.n_units or len(manifest) != job.n_lr_pairs:
            raise ValueError(
                f"Inputs changed after discovery for {job.sample}: "
                f"units {len(unit_ids)} != {job.n_units} or LR pairs {len(manifest)} != {job.n_lr_pairs}."
            )
        current_expected_rows = int(manifest["nnz"].sum())
        if current_expected_rows != job.expected_rows:
            raise ValueError(
                f"LR manifest changed after discovery for {job.sample}: "
                f"rows {current_expected_rows} != {job.expected_rows}."
            )

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = job.output_path.with_name(
            f"{job.output_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
        )
        result.partial_file = str(partial_path)
        schema = _output_schema(job)
        unit_dictionary = pa.array(unit_ids, type=pa.string())
        writer = pq.ParquetWriter(
            partial_path,
            schema=schema,
            compression=options.compression,
            use_dictionary=True,
            write_statistics=True,
        )

        exported_rows = 0
        exported_weight_sum = 0.0
        for _, lr_record in manifest.iterrows():
            matrix_path = job.lr_dir / str(lr_record["filename"])
            if not matrix_path.is_file():
                raise FileNotFoundError(f"Missing LR matrix during export: {matrix_path}")
            matrix = sp.load_npz(matrix_path).tocsr(copy=False)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            expected_shape = (job.n_units, job.n_units)
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"LR matrix {matrix_path} has shape {matrix.shape}; expected {expected_shape}."
                )
            declared_nnz = int(lr_record["nnz"])
            if matrix.nnz != declared_nnz:
                raise ValueError(
                    f"LR matrix {matrix_path} has nnz={matrix.nnz}; manifest declares {declared_nnz}."
                )
            if matrix.data.size and not np.isfinite(matrix.data).all():
                raise ValueError(f"LR matrix contains NaN or infinity: {matrix_path}")
            if matrix.data.size and (matrix.data < 0).any():
                raise ValueError(f"LR matrix contains negative COMMOT weights: {matrix_path}")

            for row_indices, column_indices, values in iter_csr_chunks(matrix, options.chunk_rows):
                batch = _record_batch(
                    job,
                    schema,
                    unit_dictionary,
                    lr_record,
                    row_indices,
                    column_indices,
                    values,
                )
                writer.write_batch(batch, row_group_size=len(batch))
                delta = int(values.size)
                exported_rows += delta
                exported_weight_sum += float(values.sum(dtype=np.float64))
                result.exported_rows = exported_rows
                result.exported_weight_sum = exported_weight_sum
                _emit_progress(
                    progress_queue,
                    {
                        "type": "rows",
                        "layer": job.layer,
                        "sample": job.sample,
                        "delta": delta,
                        "exported_rows": exported_rows,
                        "expected_rows": job.expected_rows,
                    },
                )

        writer.close()
        writer = None
        if exported_rows != job.expected_rows:
            raise ValueError(
                f"Exported row count {exported_rows} does not match expected nnz sum "
                f"{job.expected_rows} for {job.sample}."
            )

        total_path = job.lr_manifest_path.parent / f"{job.sample}_CCI_total.npz"
        expected_weight_sum: Optional[float] = None
        if total_path.is_file():
            total_matrix = sp.load_npz(total_path)
            if total_matrix.shape != (job.n_units, job.n_units):
                raise ValueError(
                    f"CCI total matrix {total_path} has shape {total_matrix.shape}; "
                    f"expected {(job.n_units, job.n_units)}."
                )
            expected_weight_sum = float(total_matrix.data.sum(dtype=np.float64))
            if not np.isclose(exported_weight_sum, expected_weight_sum, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    f"Exported LR weight sum {exported_weight_sum} does not match CCI total "
                    f"weight sum {expected_weight_sum} for {job.sample}."
                )

        partial_path.replace(job.output_path)
        result.partial_file = ""
        result.status = "completed"
        result.exported_rows = exported_rows
        result.exported_weight_sum = exported_weight_sum
        result.expected_weight_sum = expected_weight_sum
        result.output_bytes = int(job.output_path.stat().st_size)
        return result
    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return result
    finally:
        result.elapsed_seconds = float(time.perf_counter() - started)
        _emit_progress(
            progress_queue,
            {
                "type": "finished",
                "layer": job.layer,
                "sample": job.sample,
                "status": result.status,
                "exported_rows": result.exported_rows,
                "expected_rows": job.expected_rows,
            },
        )


def _run_serial(
    jobs: Sequence[ExportJob],
    options: ExportOptions,
    progress_callback: Optional[ProgressCallback],
) -> List[ExportResult]:
    class CallbackQueue:
        def put(self, event: Dict[str, object]) -> None:
            if progress_callback is not None:
                progress_callback(event)

    queue = CallbackQueue() if progress_callback is not None else None
    return [export_one_job(job, options, queue) for job in jobs]


def run_export_jobs(
    jobs: Sequence[ExportJob],
    options: ExportOptions = ExportOptions(),
    workers: int = DEFAULT_WORKERS,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[ExportResult]:
    jobs = list(jobs)
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}.")
    if not jobs:
        return []
    if workers == 1:
        return _run_serial(jobs, options, progress_callback)

    results: List[ExportResult] = []
    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        max_workers = min(int(workers), len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending = {
                executor.submit(export_one_job, job, options, progress_queue): job
                for job in jobs
            }
            while pending:
                try:
                    event = progress_queue.get(timeout=0.2)
                    if progress_callback is not None:
                        progress_callback(event)
                except Empty:
                    pass

                completed = [future for future in pending if future.done()]
                for future in completed:
                    job = pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(
                            ExportResult(
                                layer=job.layer,
                                sample=job.sample,
                                organ=job.organ,
                                stage=job.stage,
                                status="error",
                                output_file=str(job.output_path),
                                expected_rows=job.expected_rows,
                                n_units=job.n_units,
                                n_lr_pairs=job.n_lr_pairs,
                                error=f"WorkerProcessError: {type(exc).__name__}: {exc}",
                            )
                        )

            while True:
                try:
                    event = progress_queue.get_nowait()
                except Empty:
                    break
                if progress_callback is not None:
                    progress_callback(event)

    return sorted(results, key=lambda result: (result.layer, result.organ, float(result.stage), result.sample))


def jobs_as_rows(jobs: Iterable[ExportJob]) -> List[Dict[str, object]]:
    return [
        {
            "layer": job.layer,
            "sample": job.sample,
            "organ": job.organ,
            "stage": job.stage,
            "n_units": job.n_units,
            "n_lr_pairs": job.n_lr_pairs,
            "expected_rows": job.expected_rows,
            "output_file": str(job.output_path),
        }
        for job in jobs
    ]
