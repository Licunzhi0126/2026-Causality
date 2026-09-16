#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.io.multiscale_coarse_inputs import (  # noqa: E402
    DEFAULT_SCALES,
    DEFAULT_TIME_POINTS,
    InputRoots,
    PreparedStageInputs,
    prepare_multiscale_inputs,
)


METHOD = "complete_combined_coarse_maturity_cci_grn"
DEFAULT_K_BY_SCALE = {"spot": [40], "seurat_k150": [40], "seurat_k40": [10]}
SUMMARY_REQUIRED_KEYS = {
    "EI_micro_fixed",
    "EI_macro_best_checkpoint",
    "delta_EI_best_checkpoint",
    "best_epoch",
    "hardK_t",
    "hardK_tp",
    "Keff_t",
    "Keff_tp",
}


@dataclass(frozen=True)
class RunSpec:
    scale: str
    source: PreparedStageInputs
    target: PreparedStageInputs
    k: int
    seed: int
    run_dir: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run maturity+CCI+GRN coarse graining across input scales.")
    parser.add_argument("--spot-root", type=Path, required=True)
    parser.add_argument("--seurat-k40-root", type=Path, required=True)
    parser.add_argument("--seurat-k150-root", type=Path, required=True)
    parser.add_argument("--cci-root", type=Path, required=True)
    parser.add_argument("--grn-root", type=Path, required=True)
    parser.add_argument("--developmental-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--time-points", nargs="+", default=list(DEFAULT_TIME_POINTS))
    parser.add_argument("--pairs", nargs="+", default=None, help="Optional source:target pairs; defaults to adjacent time points.")
    parser.add_argument("--scales", nargs="+", choices=list(DEFAULT_SCALES), default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--k-by-scale",
        nargs="+",
        default=["spot=40", "seurat_k150=40", "seurat_k40=10"],
        help="Assignments such as spot=40 seurat_k150=40 seurat_k40=5,10,20.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--overwrite-prepared", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--runner-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments after this flag are appended unchanged to the existing WYT runner command.",
    )
    return parser


def parse_k_by_scale(values: Sequence[str], requested_scales: Sequence[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --k-by-scale value {value!r}; expected scale=K or scale=K1,K2.")
        scale, raw_values = value.split("=", 1)
        scale = scale.strip()
        if scale not in DEFAULT_SCALES:
            raise ValueError(f"Unknown scale in --k-by-scale: {scale!r}.")
        try:
            ks = [int(item) for item in raw_values.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid K list for {scale}: {raw_values!r}.") from exc
        if not ks or any(k <= 0 for k in ks):
            raise ValueError(f"K values for {scale} must be positive.")
        parsed.setdefault(scale, []).extend(ks)
    missing = [scale for scale in requested_scales if scale not in parsed]
    if missing:
        raise ValueError(f"Missing --k-by-scale assignments for {missing}.")
    return {scale: list(dict.fromkeys(parsed[scale])) for scale in requested_scales}


def parse_pairs(time_points: Sequence[str], values: Sequence[str] | None) -> list[tuple[str, str]]:
    points = list(map(str, time_points))
    if values is None:
        return list(zip(points[:-1], points[1:]))
    pairs: list[tuple[str, str]] = []
    for value in values:
        delimiter = ":" if ":" in value else "->" if "->" in value else None
        if delimiter is None:
            raise ValueError(f"Invalid pair {value!r}; expected source:target.")
        source, target = (part.strip() for part in value.split(delimiter, 1))
        if source not in points or target not in points:
            raise ValueError(f"Pair {value!r} contains a time not listed in --time-points.")
        pairs.append((source, target))
    return pairs


def build_run_specs(
    prepared: Sequence[PreparedStageInputs],
    *,
    out_root: Path,
    scales: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    k_by_scale: dict[str, list[int]],
    seeds: Sequence[int],
) -> list[RunSpec]:
    lookup = {(item.resolved.scale, item.resolved.stage): item for item in prepared}
    specs: list[RunSpec] = []
    for scale in scales:
        for source_time, target_time in pairs:
            source = lookup[(scale, source_time)]
            target = lookup[(scale, target_time)]
            for k in k_by_scale[scale]:
                if k >= min(source.n_units, target.n_units):
                    raise ValueError(
                        f"Genuine coarse graining requires K < min(n_t, n_tp); "
                        f"got scale={scale}, pair={source_time}->{target_time}, K={k}, "
                        f"counts=({source.n_units}, {target.n_units})."
                    )
                for seed in seeds:
                    run_dir = (
                        Path(out_root)
                        / METHOD
                        / scale
                        / f"K{k}"
                        / f"{source_time}_to_{target_time}"
                        / f"seed_{seed}"
                    )
                    specs.append(RunSpec(scale, source, target, int(k), int(seed), run_dir))
    return specs


def runner_command(spec: RunSpec, runner_extra_args: Sequence[str]) -> list[str]:
    source = spec.source
    target = spec.target
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_wyt_deltaei_coarse_grain.py"),
        "--method",
        METHOD,
        "--h5ad-t",
        str(source.resolved.h5ad),
        "--h5ad-tp",
        str(target.resolved.h5ad),
        "--cci-t",
        str(source.resolved.cci_total),
        "--cci-tp",
        str(target.resolved.cci_total),
        "--cci-index-t",
        str(source.cci_index),
        "--cci-index-tp",
        str(target.cci_index),
        "--grn-t",
        str(source.resolved.grn_edges),
        "--grn-tp",
        str(target.resolved.grn_edges),
        "--maturity-t",
        str(source.resolved.maturity_features),
        "--maturity-tp",
        str(target.resolved.maturity_features),
        "--maturity-id-column",
        "unit_id",
        "--maturity-column",
        "pseudotime",
        "--k",
        str(spec.k),
        "--seed",
        str(spec.seed),
        "--out-dir",
        str(spec.run_dir),
    ]
    return [*command, *map(str, runner_extra_args)]


def run_experiments(
    specs: Sequence[RunSpec],
    *,
    out_root: Path,
    runner_extra_args: Sequence[str],
    resume: bool,
    force: bool,
    continue_on_error: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        summary_path = spec.run_dir / "summary.json"
        command = runner_command(spec, runner_extra_args)
        if resume and _valid_summary(summary_path):
            rows.append(aggregate_run(spec, status="resumed"))
            continue
        if spec.run_dir.exists():
            if force:
                _remove_run_dir(spec.run_dir, out_root=out_root)
            else:
                raise FileExistsError(
                    f"Run directory already exists: {spec.run_dir}. Use --resume for a valid completed run or --force to rerun."
                )
        spec.run_dir.mkdir(parents=True, exist_ok=False)
        context = {
            "input_scale": spec.scale,
            "organ": spec.source.resolved.organ,
            "source_time": spec.source.resolved.stage,
            "target_time": spec.target.resolved.stage,
            "k_slots": spec.k,
            "seed": spec.seed,
            "resolved_inputs": {
                "source": spec.source.manifest(),
                "target": spec.target.manifest(),
            },
            "maturity_aggregation": "mean_over_member_spots" if spec.scale != "spot" else "spot_native",
            "cci_index_provenance": {
                "source": spec.source.index_source,
                "target": spec.target.index_source,
            },
            "command": command,
        }
        _write_json(spec.run_dir / "experiment_context.json", context)
        try:
            subprocess.run(command, check=True, cwd=REPO_ROOT)
            if not _valid_summary(summary_path):
                raise RuntimeError(f"Runner completed without a valid summary: {summary_path}")
            rows.append(aggregate_run(spec, status="success"))
        except Exception as exc:
            rows.append(aggregate_run(spec, status="failure", error=str(exc)))
            if not continue_on_error:
                write_aggregate_outputs(rows, out_root=out_root)
                raise
    return rows


def aggregate_run(spec: RunSpec, *, status: str, error: str = "") -> dict[str, object]:
    row: dict[str, object] = {
        "input_scale": spec.scale,
        "time_pair": f"{spec.source.resolved.stage}_to_{spec.target.resolved.stage}",
        "source_time": spec.source.resolved.stage,
        "target_time": spec.target.resolved.stage,
        "K": spec.k,
        "seed": spec.seed,
        "n_micro_t": spec.source.n_units,
        "n_micro_tp": spec.target.n_units,
        "run_directory": str(spec.run_dir),
        "status": status,
        "error": error,
    }
    summary_path = spec.run_dir / "summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        input_manifest_path = spec.run_dir / "input_manifest.json"
        if input_manifest_path.exists():
            input_manifest = _read_json(input_manifest_path)
            row["n_micro_t"] = input_manifest.get("unit_count_t", row["n_micro_t"])
            row["n_micro_tp"] = input_manifest.get("unit_count_tp", row["n_micro_tp"])
        fixed_keys = [
            "EI_micro_fixed",
            "EI_macro_best_checkpoint",
            "delta_EI_best_checkpoint",
            "best_epoch",
            "hardK_t",
            "hardK_tp",
            "Keff_t",
            "Keff_tp",
            "L_dev_best_checkpoint",
        ]
        for key in fixed_keys:
            row[key] = summary.get(key)
        for key, value in summary.items():
            if key.startswith("EI_strict") or key.startswith("deltaEI_strict"):
                row[key] = value
    return row


def write_aggregate_outputs(rows: Sequence[dict[str, object]], *, out_root: Path) -> tuple[Path, Path]:
    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "multiscale_summary.csv"
    manifest_path = output_root / "multiscale_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "method": METHOD,
            "run_count": len(rows),
            "status_counts": pd.Series([row.get("status") for row in rows]).value_counts().to_dict(),
            "preparation_manifest": str(output_root / "_prepared_inputs" / "preparation_manifest.json"),
            "summary_csv": str(csv_path),
            "runs": list(rows),
        },
    )
    return csv_path, manifest_path


def _valid_summary(path: Path) -> bool:
    try:
        payload = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return SUMMARY_REQUIRED_KEYS.issubset(payload)


def _remove_run_dir(run_dir: Path, *, out_root: Path) -> None:
    resolved = run_dir.resolve()
    allowed_root = (Path(out_root).resolve() / METHOD)
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove run directory outside {allowed_root}: {resolved}") from exc
    if len(relative.parts) != 4 or resolved == allowed_root:
        raise ValueError(f"Refusing to remove unexpected run-directory shape: {resolved}")
    shutil.rmtree(resolved)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def roots_from_args(args: argparse.Namespace) -> InputRoots:
    return InputRoots(
        spot_root=args.spot_root,
        seurat_k40_root=args.seurat_k40_root,
        seurat_k150_root=args.seurat_k150_root,
        cci_root=args.cci_root,
        grn_root=args.grn_root,
        developmental_root=args.developmental_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    k_by_scale = parse_k_by_scale(args.k_by_scale, args.scales)
    pairs = parse_pairs(args.time_points, args.pairs)
    required_times = list(dict.fromkeys(time for pair in pairs for time in pair))
    prepared = prepare_multiscale_inputs(
        roots_from_args(args),
        out_root=args.out_root,
        organ=args.organ,
        time_points=required_times,
        scales=args.scales,
        overwrite=args.overwrite_prepared,
    )
    specs = build_run_specs(
        prepared,
        out_root=args.out_root,
        scales=args.scales,
        pairs=pairs,
        k_by_scale=k_by_scale,
        seeds=args.seeds,
    )
    rows = run_experiments(
        specs,
        out_root=args.out_root,
        runner_extra_args=args.runner_extra_args,
        resume=args.resume,
        force=args.force,
        continue_on_error=args.continue_on_error,
    )
    csv_path, _ = write_aggregate_outputs(rows, out_root=args.out_root)
    print(f"Completed {len(rows)} multiscale runs; aggregate summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
