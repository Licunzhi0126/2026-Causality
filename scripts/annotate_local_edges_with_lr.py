#!/usr/bin/env python3
"""Annotate local directed-edge CSV files with heart spot COMMOT LR pairs.

The script is independent of all project-local Python modules.  It reads one
CSV or a directory of CSV files containing at least::

    timepoint, source_local_id, target_local_id

``source_local_id`` and ``target_local_id`` are interpreted as zero-based row
and column indices into the corresponding COMMOT LR matrices.  Time labels
such as ``12p5`` are normalized to ``12.5``.  The original rows and columns
are preserved, and seven LR annotation columns are appended.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPT_VERSION = "1.0.0"
REQUIRED_COLUMNS = ("timepoint", "source_local_id", "target_local_id")
ANNOTATION_COLUMNS = (
    "ligand",
    "receptor",
    "lr_pair",
    "lr_weight",
    "lr_count",
    "lr_weight_sum",
    "lr_match_status",
)
UINT32_LIMIT = 1 << 32


@dataclass(frozen=True)
class CCIAssets:
    stage: str
    sample_stem: str
    manifest_path: Path
    lr_dir: Path
    total_path: Path
    manifest: pd.DataFrame
    matrix_size: int


@dataclass
class StageAnnotations:
    stage: str
    edge_keys: np.ndarray
    weights: sp.csr_matrix
    ligands: np.ndarray
    receptors: np.ndarray
    lr_pairs: np.ndarray
    matrix_size: int


@dataclass(frozen=True)
class InputPlan:
    input_path: Path
    output_path: Path
    rows: int
    stages: tuple[str, ...]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_stage(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError(f"Invalid empty timepoint: {value!r}")
    if text[:1].lower() == "e":
        text = text[1:]
    match = re.fullmatch(r"(\d+)[pP](\d+)", text)
    if match:
        text = f"{match.group(1)}.{match.group(2)}"
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Cannot parse timepoint {value!r}; expected values such as 12p5 or 12.5.") from exc
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def pack_edge_ids(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_u64 = np.asarray(source, dtype=np.uint64)
    target_u64 = np.asarray(target, dtype=np.uint64)
    return (source_u64 << np.uint64(32)) | target_u64


def unpack_edge_ids(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(keys, dtype=np.uint64)
    source = (values >> np.uint64(32)).astype(np.int64)
    target = (values & np.uint64(UINT32_LIMIT - 1)).astype(np.int64)
    return source, target


def coerce_local_ids(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    source_numeric = pd.to_numeric(frame["source_local_id"], errors="coerce").to_numpy(dtype=float)
    target_numeric = pd.to_numeric(frame["target_local_id"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(source_numeric)) or not np.all(np.isfinite(target_numeric)):
        raise ValueError(f"{path} contains missing or nonnumeric local IDs.")
    if not np.all(source_numeric == np.floor(source_numeric)) or not np.all(target_numeric == np.floor(target_numeric)):
        raise ValueError(f"{path} contains noninteger local IDs.")
    if source_numeric.size and (source_numeric.min() < 0 or target_numeric.min() < 0):
        raise ValueError(f"{path} contains negative local IDs.")
    if source_numeric.size and (source_numeric.max() >= UINT32_LIMIT or target_numeric.max() >= UINT32_LIMIT):
        raise ValueError(f"{path} contains local IDs too large for the packed key format.")
    return source_numeric.astype(np.int64), target_numeric.astype(np.int64)


def iter_required_chunks(path: Path, chunk_rows: int, max_rows: int | None) -> Iterator[pd.DataFrame]:
    remaining = max_rows
    reader = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), chunksize=max(1, int(chunk_rows)))
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
        if not chunk.empty:
            yield chunk


def discover_inputs(input_csv: Path | None, input_dir: Path | None, recursive: bool) -> list[Path]:
    if input_csv is not None:
        path = input_csv.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    if input_dir is None:
        raise ValueError("One of --input-csv or --input-dir is required.")
    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(path for path in (root.rglob("*.csv") if recursive else root.glob("*.csv")) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return paths


def resolve_output_paths(
    input_paths: Sequence[Path],
    output_dir: Path,
    output_name: str | None,
) -> dict[Path, Path]:
    if output_name is not None and len(input_paths) != 1:
        raise ValueError("--output-name can only be used with one input CSV.")
    root = output_dir.resolve()
    outputs: dict[Path, Path] = {}
    for input_path in input_paths:
        name = output_name if output_name is not None else input_path.name
        output_path = root / name
        if output_path in outputs.values():
            raise ValueError(f"Multiple inputs resolve to the same output: {output_path}")
        outputs[input_path] = output_path
    return outputs


def validate_columns(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    conflicts = [column for column in ANNOTATION_COLUMNS if column in columns]
    if conflicts:
        raise ValueError(f"{path} already contains annotation columns and will not be overwritten: {conflicts}")
    return columns


def collect_edge_queries(
    input_paths: Sequence[Path],
    output_paths: dict[Path, Path],
    *,
    chunk_rows: int,
    max_rows: int | None,
) -> tuple[list[InputPlan], dict[str, np.ndarray]]:
    stage_key_parts: dict[str, list[np.ndarray]] = {}
    plans: list[InputPlan] = []
    for path in input_paths:
        validate_columns(path)
        output_path = output_paths[path]
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        stages_seen: set[str] = set()
        row_count = 0
        for chunk in iter_required_chunks(path, chunk_rows, max_rows):
            source, target = coerce_local_ids(chunk, path)
            stages = chunk["timepoint"].map(normalize_stage).to_numpy(dtype=object)
            row_count += len(chunk)
            for stage in sorted(set(map(str, stages))):
                mask = stages == stage
                keys = pack_edge_ids(source[mask], target[mask])
                stage_key_parts.setdefault(stage, []).append(keys)
                stages_seen.add(stage)
        if row_count == 0:
            raise ValueError(f"{path} has no rows to process.")
        plans.append(
            InputPlan(
                input_path=path,
                output_path=output_path,
                rows=row_count,
                stages=tuple(sorted(stages_seen, key=lambda value: Decimal(value))),
            )
        )
    unique_by_stage: dict[str, np.ndarray] = {}
    for stage, parts in stage_key_parts.items():
        unique_by_stage[stage] = np.unique(np.concatenate(parts).astype(np.uint64, copy=False))
    return plans, unique_by_stage


def resolve_cci_assets(cci_root: Path, layer: str, organ: str, stage: str) -> CCIAssets:
    layer_root = cci_root.resolve() / layer
    sample_stem = f"{layer}_{organ}_{stage}"
    manifest_path = layer_root / f"{sample_stem}_COMMOT_lr_pairs.tsv"
    lr_dir = layer_root / f"{sample_stem}_COMMOT_by_LR"
    total_path = layer_root / f"{sample_stem}_CCI_total.npz"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing COMMOT LR manifest: {manifest_path}")
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"Missing COMMOT LR directory: {lr_dir}")
    if not total_path.is_file():
        raise FileNotFoundError(f"Missing CCI total matrix used for validation: {total_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    required = {"filename", "ligand", "receptor"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
    if "lr_key" not in manifest.columns:
        manifest["lr_key"] = manifest["ligand"].astype(str) + "-" + manifest["receptor"].astype(str)
    manifest = manifest.copy()
    for column in ("filename", "ligand", "receptor", "lr_key"):
        manifest[column] = manifest[column].astype(str)
    manifest = manifest.sort_values(["lr_key", "filename"], kind="mergesort").reset_index(drop=True)
    missing_files = [filename for filename in manifest["filename"] if not (lr_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} LR files listed in {manifest_path} are missing; first: {missing_files[0]}")
    if {"shape_0", "shape_1"}.issubset(manifest.columns):
        shape_0 = pd.to_numeric(manifest["shape_0"], errors="raise").astype(int)
        shape_1 = pd.to_numeric(manifest["shape_1"], errors="raise").astype(int)
        if shape_0.nunique() != 1 or shape_1.nunique() != 1 or int(shape_0.iloc[0]) != int(shape_1.iloc[0]):
            raise ValueError(f"Inconsistent or nonsquare LR matrix shapes in {manifest_path}")
        matrix_size = int(shape_0.iloc[0])
    else:
        first = sp.load_npz(lr_dir / manifest.iloc[0]["filename"])
        if first.shape[0] != first.shape[1]:
            raise ValueError(f"LR matrices must be square, got {first.shape}")
        matrix_size = int(first.shape[0])
    return CCIAssets(
        stage=stage,
        sample_stem=sample_stem,
        manifest_path=manifest_path,
        lr_dir=lr_dir,
        total_path=total_path,
        manifest=manifest,
        matrix_size=matrix_size,
    )


def validate_edge_bounds(stage: str, edge_keys: np.ndarray, matrix_size: int) -> None:
    source, target = unpack_edge_ids(edge_keys)
    invalid = (source < 0) | (target < 0) | (source >= matrix_size) | (target >= matrix_size)
    if np.any(invalid):
        index = int(np.flatnonzero(invalid)[0])
        raise IndexError(
            f"Stage {stage} local edge ({source[index]}, {target[index]}) is outside "
            f"the zero-based CCI range 0..{matrix_size - 1}."
        )


def query_sparse_values(
    matrix: sp.spmatrix,
    source: np.ndarray,
    target: np.ndarray,
    chunk_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    csr = matrix.tocsr()
    for start in range(0, len(source), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(source))
        values = np.asarray(csr[source[start:stop], target[start:stop]]).reshape(-1)
        yield start, values


def build_stage_annotations(
    assets: CCIAssets,
    edge_keys: np.ndarray,
    *,
    query_chunk_size: int,
    validation_tolerance: float,
) -> StageAnnotations:
    validate_edge_bounds(assets.stage, edge_keys, assets.matrix_size)
    source, target = unpack_edge_ids(edge_keys)
    hit_rows: list[np.ndarray] = []
    hit_lr_columns: list[np.ndarray] = []
    hit_weights: list[np.ndarray] = []
    total_lr_files = len(assets.manifest)
    log(
        f"Stage {assets.stage}: querying {len(edge_keys)} unique directed edges "
        f"against {total_lr_files} LR matrices."
    )
    for lr_index, record in enumerate(assets.manifest.itertuples(index=False), start=0):
        matrix_path = assets.lr_dir / str(record.filename)
        matrix = sp.load_npz(matrix_path).tocsr()
        if matrix.shape != (assets.matrix_size, assets.matrix_size):
            raise ValueError(f"Unexpected LR matrix shape {matrix.shape}: {matrix_path}")
        for start, values in query_sparse_values(matrix, source, target, query_chunk_size):
            hits = np.flatnonzero(values > 0)
            if hits.size:
                hit_rows.append((hits + start).astype(np.int64, copy=False))
                hit_lr_columns.append(np.full(hits.size, lr_index, dtype=np.int32))
                hit_weights.append(values[hits].astype(np.float64, copy=False))
        if (lr_index + 1) % 25 == 0 or lr_index + 1 == total_lr_files:
            log(f"  LR matrices {lr_index + 1}/{total_lr_files}")
    if hit_rows:
        rows = np.concatenate(hit_rows)
        columns = np.concatenate(hit_lr_columns)
        values = np.concatenate(hit_weights)
        weights = sp.coo_matrix(
            (values, (rows, columns)),
            shape=(len(edge_keys), total_lr_files),
            dtype=np.float64,
        ).tocsr()
        weights.sum_duplicates()
        weights.sort_indices()
    else:
        weights = sp.csr_matrix((len(edge_keys), total_lr_files), dtype=np.float64)

    total = sp.load_npz(assets.total_path).tocsr()
    if total.shape != (assets.matrix_size, assets.matrix_size):
        raise ValueError(f"Unexpected CCI total shape {total.shape}: {assets.total_path}")
    annotation_sums = np.asarray(weights.sum(axis=1)).ravel()
    total_values = np.empty(len(edge_keys), dtype=float)
    for start, values in query_sparse_values(total, source, target, query_chunk_size):
        total_values[start : start + len(values)] = values
    max_error = float(np.max(np.abs(annotation_sums - total_values))) if len(edge_keys) else 0.0
    if max_error > validation_tolerance:
        raise ValueError(
            f"Stage {assets.stage} LR sums do not reproduce CCI_total: "
            f"max_abs_error={max_error:.12g} > tolerance={validation_tolerance:.12g}"
        )
    log(
        f"Stage {assets.stage}: LR annotations ready, matched_edges={int(np.count_nonzero(np.diff(weights.indptr)))}, "
        f"LR_hits={weights.nnz}, max_total_error={max_error:.3g}."
    )
    return StageAnnotations(
        stage=assets.stage,
        edge_keys=edge_keys,
        weights=weights,
        ligands=assets.manifest["ligand"].to_numpy(dtype=object),
        receptors=assets.manifest["receptor"].to_numpy(dtype=object),
        lr_pairs=assets.manifest["lr_key"].to_numpy(dtype=object),
        matrix_size=assets.matrix_size,
    )


def locate_annotation_rows(annotation: StageAnnotations, keys: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(annotation.edge_keys, keys)
    valid = positions < len(annotation.edge_keys)
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = annotation.edge_keys[positions[valid_indices]] == keys[valid_indices]
    if not np.all(valid):
        missing_index = int(np.flatnonzero(~valid)[0])
        source, target = unpack_edge_ids(keys[missing_index : missing_index + 1])
        raise KeyError(f"Annotation cache is missing edge ({source[0]}, {target[0]}) for stage {annotation.stage}.")
    return positions.astype(np.int64, copy=False)


def format_annotation_row(annotation: StageAnnotations, row_index: int) -> tuple[str, str, str, str, int, float]:
    start = int(annotation.weights.indptr[row_index])
    stop = int(annotation.weights.indptr[row_index + 1])
    lr_indices = annotation.weights.indices[start:stop]
    values = annotation.weights.data[start:stop]
    if len(lr_indices) == 0:
        return "", "", "", "", 0, 0.0
    ligand = ";".join(map(str, annotation.ligands[lr_indices]))
    receptor = ";".join(map(str, annotation.receptors[lr_indices]))
    lr_pair = ";".join(map(str, annotation.lr_pairs[lr_indices]))
    lr_weight = ";".join(format(float(value), ".12g") for value in values)
    return ligand, receptor, lr_pair, lr_weight, int(len(lr_indices)), float(values.sum())


def annotate_chunk(
    chunk: pd.DataFrame,
    path: Path,
    annotations_by_stage: dict[str, StageAnnotations],
    weight_tolerance: float,
) -> pd.DataFrame:
    source, target = coerce_local_ids(chunk, path)
    stages = chunk["timepoint"].map(normalize_stage).to_numpy(dtype=object)
    output_ligand = np.empty(len(chunk), dtype=object)
    output_receptor = np.empty(len(chunk), dtype=object)
    output_lr_pair = np.empty(len(chunk), dtype=object)
    output_lr_weight = np.empty(len(chunk), dtype=object)
    output_lr_count = np.zeros(len(chunk), dtype=np.int32)
    output_lr_weight_sum = np.zeros(len(chunk), dtype=np.float64)
    output_status = np.empty(len(chunk), dtype=object)
    for stage in sorted(set(map(str, stages)), key=lambda value: Decimal(value)):
        if stage not in annotations_by_stage:
            raise KeyError(f"No LR annotation data was built for stage {stage}.")
        mask = stages == stage
        output_indices = np.flatnonzero(mask)
        keys = pack_edge_ids(source[mask], target[mask])
        annotation = annotations_by_stage[stage]
        annotation_rows = locate_annotation_rows(annotation, keys)
        for output_index, annotation_row in zip(output_indices, annotation_rows):
            ligand, receptor, lr_pair, lr_weight, lr_count, lr_weight_sum = format_annotation_row(
                annotation,
                int(annotation_row),
            )
            output_ligand[output_index] = ligand
            output_receptor[output_index] = receptor
            output_lr_pair[output_index] = lr_pair
            output_lr_weight[output_index] = lr_weight
            output_lr_count[output_index] = lr_count
            output_lr_weight_sum[output_index] = lr_weight_sum
            output_status[output_index] = "matched" if lr_count else "no_lr"

    if "local_edge_weight" in chunk.columns:
        expected = pd.to_numeric(chunk["local_edge_weight"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(expected)
        mismatched = finite & (np.abs(expected - output_lr_weight_sum) > weight_tolerance)
        matched_mask = output_lr_count > 0
        output_status[mismatched & matched_mask] = "matched_weight_mismatch"

    output = chunk.copy()
    output["ligand"] = output_ligand
    output["receptor"] = output_receptor
    output["lr_pair"] = output_lr_pair
    output["lr_weight"] = output_lr_weight
    output["lr_count"] = output_lr_count
    output["lr_weight_sum"] = output_lr_weight_sum
    output["lr_match_status"] = output_status
    return output


def write_annotated_csv(
    plan: InputPlan,
    annotations_by_stage: dict[str, StageAnnotations],
    *,
    chunk_rows: int,
    max_rows: int | None,
    weight_tolerance: float,
) -> dict[str, object]:
    if plan.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {plan.output_path}")
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = plan.output_path.with_name(f"{plan.output_path.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse existing partial output: {partial}")
    remaining = max_rows
    rows_written = 0
    status_counts: dict[str, int] = {}
    max_weight_error = 0.0
    first_chunk = True
    log(f"Writing annotated CSV: {plan.input_path.name} -> {plan.output_path}")
    try:
        with partial.open("x", newline="", encoding="utf-8") as handle:
            reader = pd.read_csv(plan.input_path, chunksize=max(1, int(chunk_rows)))
            for chunk in reader:
                if remaining is not None:
                    if remaining <= 0:
                        break
                    chunk = chunk.iloc[:remaining].copy()
                    remaining -= len(chunk)
                if chunk.empty:
                    continue
                annotated = annotate_chunk(chunk, plan.input_path, annotations_by_stage, weight_tolerance)
                if "local_edge_weight" in annotated.columns:
                    expected = pd.to_numeric(annotated["local_edge_weight"], errors="coerce").to_numpy(dtype=float)
                    observed = annotated["lr_weight_sum"].to_numpy(dtype=float)
                    finite = np.isfinite(expected)
                    if np.any(finite):
                        max_weight_error = max(max_weight_error, float(np.max(np.abs(expected[finite] - observed[finite]))))
                for status, count in annotated["lr_match_status"].value_counts().items():
                    status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)
                annotated.to_csv(handle, index=False, header=first_chunk)
                first_chunk = False
                rows_written += len(annotated)
                log(f"  rows written: {rows_written}/{plan.rows}")
        partial.rename(plan.output_path)
    except Exception:
        log(f"Write failed; partial file was retained for diagnosis: {partial}")
        raise
    return {
        "input": str(plan.input_path),
        "output": str(plan.output_path),
        "rows": rows_written,
        "stages": ";".join(plan.stages),
        "status_counts": status_counts,
        "max_local_edge_weight_error": max_weight_error,
    }


def dry_run_report(
    plans: Sequence[InputPlan],
    unique_by_stage: dict[str, np.ndarray],
    assets_by_stage: dict[str, CCIAssets],
) -> int:
    rows = []
    for plan in plans:
        rows.append(
            {
                "input": str(plan.input_path),
                "output": str(plan.output_path),
                "rows": plan.rows,
                "stages": ";".join(plan.stages),
                "output_exists": plan.output_path.exists(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    stage_rows = []
    for stage, edge_keys in sorted(unique_by_stage.items(), key=lambda item: Decimal(item[0])):
        assets = assets_by_stage[stage]
        source, target = unpack_edge_ids(edge_keys)
        stage_rows.append(
            {
                "stage": stage,
                "sample": assets.sample_stem,
                "unique_edges": len(edge_keys),
                "source_min": int(source.min()),
                "source_max": int(source.max()),
                "target_min": int(target.min()),
                "target_max": int(target.max()),
                "cci_size": assets.matrix_size,
                "lr_files": len(assets.manifest),
                "manifest": str(assets.manifest_path),
            }
        )
    print(pd.DataFrame(stage_rows).to_string(index=False))
    print("Dry-run passed: no output directory or file was created.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv", type=Path)
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--cci-root", type=Path, required=True)
    parser.add_argument("--organ", type=str, default="heart")
    parser.add_argument("--layer", choices=("spot",), default="spot")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--csv-chunk-rows", type=int, default=100_000)
    parser.add_argument("--query-chunk-size", type=int, default=250_000)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--weight-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive.")
    if args.csv_chunk_rows <= 0 or args.query_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive.")
    if args.validation_tolerance < 0 or args.weight_tolerance < 0:
        raise ValueError("Tolerances must be nonnegative.")
    input_paths = discover_inputs(args.input_csv, args.input_dir, args.recursive)
    output_paths = resolve_output_paths(input_paths, args.output_dir, args.output_name)
    log(f"Planning {len(input_paths)} input CSV file(s).")
    plans, unique_by_stage = collect_edge_queries(
        input_paths,
        output_paths,
        chunk_rows=args.csv_chunk_rows,
        max_rows=args.max_rows,
    )
    assets_by_stage: dict[str, CCIAssets] = {}
    for stage, edge_keys in sorted(unique_by_stage.items(), key=lambda item: Decimal(item[0])):
        assets = resolve_cci_assets(args.cci_root, args.layer, args.organ, stage)
        validate_edge_bounds(stage, edge_keys, assets.matrix_size)
        assets_by_stage[stage] = assets
    if args.dry_run:
        return dry_run_report(plans, unique_by_stage, assets_by_stage)

    annotations_by_stage = {
        stage: build_stage_annotations(
            assets_by_stage[stage],
            edge_keys,
            query_chunk_size=args.query_chunk_size,
            validation_tolerance=args.validation_tolerance,
        )
        for stage, edge_keys in sorted(unique_by_stage.items(), key=lambda item: Decimal(item[0]))
    }
    summaries = [
        write_annotated_csv(
            plan,
            annotations_by_stage,
            chunk_rows=args.csv_chunk_rows,
            max_rows=args.max_rows,
            weight_tolerance=args.weight_tolerance,
        )
        for plan in plans
    ]
    log("All requested CSV files were annotated.")
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        log("Interrupted by user; completed outputs were left untouched.")
        return 130
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
