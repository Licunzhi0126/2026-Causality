#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cci_edge_lr_exporter import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_COMPRESSION,
    DEFAULT_WORKERS,
    SUPPORTED_LAYERS,
    ExportOptions,
    ExportResult,
    discover_export_jobs,
    jobs_as_rows,
    run_export_jobs,
)
from factory_common import FACTORY_OUTPUT_ROOT, ensure_dir, print_table, write_csv


def build_argparser() -> argparse.ArgumentParser:
    default_cci_root = FACTORY_OUTPUT_ROOT / "cci"
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate external COMMOT LR sparse matrices into one directed edge-LR "
            "long-table Parquet file per layer/organ/stage sample."
        )
    )
    parser.add_argument("--cci-root", type=Path, default=default_cci_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <cci-root>/edge_lr_long",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=SUPPORTED_LAYERS,
        default=list(SUPPORTED_LAYERS),
    )
    parser.add_argument(
        "--sample-names",
        nargs="+",
        default=[],
        help="Optional exact COMMOT sample stems to export.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Sample-level worker process count (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help=f"Maximum long-table rows per in-memory Arrow batch (default: {DEFAULT_CHUNK_ROWS}).",
    )
    parser.add_argument(
        "--compression",
        default=DEFAULT_COMPRESSION,
        help=f"PyArrow Parquet compression codec (default: {DEFAULT_COMPRESSION}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: <output-root>/cci_edge_lr_export_manifest.csv",
    )
    parser.add_argument(
        "--strict-grid",
        action="store_true",
        help="Require all configured heart/brain/lung x 11.5/12.5/13.5/14.5 samples per layer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing completed Parquet outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list planned exports without writing files.",
    )
    return parser


class ProgressDisplay:
    def __init__(self, jobs) -> None:
        self.jobs = list(jobs)
        self.overall = tqdm(
            total=len(self.jobs),
            desc="CCI edge-LR samples",
            unit="sample",
            position=0,
            dynamic_ncols=True,
        )
        self.positions = {
            (job.layer, job.sample): position
            for position, job in enumerate(self.jobs, start=1)
        }
        self.bars: Dict[tuple, tqdm] = {}
        self.finished = set()

    def _bar(self, layer: str, sample: str, expected_rows: int) -> tqdm:
        key = (layer, sample)
        if key not in self.bars:
            self.bars[key] = tqdm(
                total=int(expected_rows),
                desc=f"{layer}/{sample}",
                unit="row",
                unit_scale=True,
                position=self.positions[key],
                leave=True,
                dynamic_ncols=True,
            )
        return self.bars[key]

    def handle(self, event: Dict[str, object]) -> None:
        layer = str(event["layer"])
        sample = str(event["sample"])
        expected_rows = int(event.get("expected_rows", 0))
        key = (layer, sample)
        event_type = str(event.get("type", ""))
        bar = self._bar(layer, sample, expected_rows)
        if event_type == "rows":
            absolute = int(event.get("exported_rows", bar.n))
            if absolute > bar.n:
                bar.update(absolute - bar.n)
        elif event_type == "finished" and key not in self.finished:
            absolute = int(event.get("exported_rows", bar.n))
            if absolute > bar.n:
                bar.update(absolute - bar.n)
            bar.set_postfix_str(str(event.get("status", "finished")))
            bar.refresh()
            self.finished.add(key)
            self.overall.update(1)

    def close(self, results: List[ExportResult]) -> None:
        by_key = {(result.layer, result.sample): result for result in results}
        for job in self.jobs:
            key = (job.layer, job.sample)
            result = by_key.get(key)
            if key not in self.finished:
                bar = self._bar(job.layer, job.sample, job.expected_rows)
                if result is not None and result.exported_rows > bar.n:
                    bar.update(result.exported_rows - bar.n)
                bar.set_postfix_str(result.status if result is not None else "unknown")
                self.finished.add(key)
                self.overall.update(1)
        for bar in self.bars.values():
            bar.close()
        self.overall.close()


def _print_dry_run(rows: List[dict]) -> None:
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "layer": row["layer"],
                "sample": row["sample"],
                "units": row["n_units"],
                "LR": row["n_lr_pairs"],
                "expected_rows": f"{int(row['expected_rows']):,}",
                "output": row["output_file"],
            }
        )
    print_table(display_rows, columns=("layer", "sample", "units", "LR", "expected_rows", "output"))
    print(f"\nPlanned samples: {len(rows)}")
    print(f"Planned rows: {sum(int(row['expected_rows']) for row in rows):,}")


def main() -> None:
    args = build_argparser().parse_args()
    output_root = args.output_root or args.cci_root / "edge_lr_long"
    manifest_path = args.manifest or output_root / "cci_edge_lr_export_manifest.csv"
    jobs = discover_export_jobs(
        cci_root=args.cci_root,
        output_root=output_root,
        layers=args.layers,
        sample_names=args.sample_names,
        strict_grid=args.strict_grid,
    )
    planned_rows = jobs_as_rows(jobs)
    if args.dry_run:
        _print_dry_run(planned_rows)
        return

    print(
        f"Exporting {len(jobs)} samples with up to {args.workers} worker processes; "
        f"expected rows={sum(job.expected_rows for job in jobs):,}."
    )
    display = ProgressDisplay(jobs)
    try:
        results = run_export_jobs(
            jobs,
            options=ExportOptions(
                chunk_rows=args.chunk_rows,
                compression=args.compression,
                overwrite=args.overwrite,
            ),
            workers=args.workers,
            progress_callback=display.handle,
        )
    finally:
        if "results" not in locals():
            results = []
        display.close(results)

    ensure_dir(manifest_path.parent)
    write_csv(manifest_path, [result.to_dict() for result in results])
    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped_existing" for result in results)
    failed = [result for result in results if result.status == "error"]
    print(
        f"CCI edge-LR export finished: completed={completed}, skipped={skipped}, "
        f"failed={len(failed)}. Manifest: {manifest_path}"
    )
    if failed:
        failure_table = pd.DataFrame(
            [
                {
                    "layer": result.layer,
                    "sample": result.sample,
                    "partial_file": result.partial_file,
                    "error": result.error.splitlines()[0] if result.error else "unknown error",
                }
                for result in failed
            ]
        )
        print(failure_table.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
