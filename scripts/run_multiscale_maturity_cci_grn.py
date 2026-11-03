#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
from mignet_ce.coarse_frontends.method_specs import (  # noqa: E402
    get_coarse_method_spec,
)


DEFAULT_METHOD = "complete_combined_coarse_maturity_cci_grn"
SUPPORTED_METHODS = (
    DEFAULT_METHOD,
    "maturity_cci_grn_two_stage",
    "shr_method_8",
    "shr_method_10",
)
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
    method: str
    training_mode: str
    frontend: str
    scale: str
    source: PreparedStageInputs
    target: PreparedStageInputs
    k: int
    seed: int
    run_dir: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run maturity+CCI+GRN coarse graining across input scales.")
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=DEFAULT_METHOD,
        help="Registered maturity+CCI+GRN method to run; defaults to the legacy single-stage method.",
    )
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
    parser.add_argument(
        "--seeds",
        nargs=1,
        type=int,
        default=[42],
        help="Exactly one seed per invocation; the seed is recorded in metadata, not the path.",
    )
    parser.add_argument("--overwrite-prepared", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--force", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="Requires --resume; back up and rerun only incomplete or protocol-mismatched run directories")
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
    method: str,
    out_root: Path,
    scales: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    k_by_scale: dict[str, list[int]],
    seeds: Sequence[int],
) -> list[RunSpec]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported multiscale method {method!r}; expected one of {SUPPORTED_METHODS}.")
    if len(seeds) != 1:
        raise ValueError(
            "The simplified method-scoped output layout supports exactly one seed per invocation."
        )
    method_spec = get_coarse_method_spec(method)
    frontend = method_spec.frontend or method_spec.name
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
                        / method
                        / scale
                        / f"K{k}"
                        / f"{source_time}_to_{target_time}"
                    )
                    specs.append(
                        RunSpec(
                            method=method,
                            training_mode=method_spec.training_mode,
                            frontend=frontend,
                            scale=scale,
                            source=source,
                            target=target,
                            k=int(k),
                            seed=int(seed),
                            run_dir=run_dir,
                        )
                    )
    return specs


def runner_command(spec: RunSpec, runner_extra_args: Sequence[str]) -> list[str]:
    _validate_runner_extra_args(runner_extra_args)
    source = spec.source
    target = spec.target
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_wyt_deltaei_coarse_grain.py"),
        "--method",
        spec.method,
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
    extra = list(map(str, runner_extra_args))
    if spec.method == "shr_method_10" and _runner_float_option(extra, "--keff-min") is None:
        # Materialize the native 0.30*K floor in config.json for each K.
        # This is 12 for K=40 and 3 for K=10; it does not change the default objective.
        extra.extend(["--keff-min", f"{0.30 * spec.k:g}"])
    return [*command, *extra]


def run_experiments(
    specs: Sequence[RunSpec],
    *,
    out_root: Path,
    runner_extra_args: Sequence[str],
    resume: bool,
    force: bool,
    resume_incomplete: bool,
    continue_on_error: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_checkpoint_floor = _runner_float_option(runner_extra_args, "--checkpoint-keff-min")
    for spec in specs:
        summary_path = spec.run_dir / "summary.json"
        command = runner_command(spec, runner_extra_args)
        method_checkpoint_floor = expected_checkpoint_floor if spec.method == "shr_method_10" else None
        method_training_floor = (
            _runner_float_option(runner_extra_args, "--keff-min")
            if spec.method == "shr_method_10" else None
        )
        if spec.method == "shr_method_10" and method_training_floor is None:
            method_training_floor = 0.30 * spec.k
        if resume and _valid_summary(
            summary_path, expected_method=spec.method,
            expected_checkpoint_keff_min=method_checkpoint_floor,
            expected_keff_min=method_training_floor,
        ):
            rows.append(aggregate_run(spec, status="resumed"))
            continue
        if spec.run_dir.exists():
            if resume_incomplete:
                resolved = spec.run_dir.resolve()
                allowed_root = (Path(out_root).resolve() / spec.method)
                try:
                    relative = resolved.relative_to(allowed_root)
                except ValueError as exc:
                    raise ValueError(f"Refusing to back up run outside {allowed_root}: {resolved}") from exc
                if len(relative.parts) != 3:
                    raise ValueError(f"Unexpected incomplete run-directory shape: {resolved}")
                # Keep backups OUTSIDE optimal_runs so recursive summary discovery
                # cannot mistake an old partial summary for a second scientific run.
                backup = (Path(out_root).resolve().parent / "_incomplete_run_backups" / spec.method / relative)
                if backup.exists():
                    raise FileExistsError(f"Refusing to overwrite existing incomplete-run backup: {backup}")
                backup.parent.mkdir(parents=True, exist_ok=True)
                print(f"[resume-incomplete] preserving failed artifacts at {backup}", flush=True)
                spec.run_dir.rename(backup)
            elif force:
                _remove_run_dir(spec.run_dir, out_root=out_root, method=spec.method)
            elif not any(spec.run_dir.iterdir()):
                pass
            else:
                raise FileExistsError(
                    f"Run directory already exists: {spec.run_dir}. Use --resume for a valid completed run or --force to rerun."
                )
        context = {
            "method": spec.method,
            "training_mode": spec.training_mode,
            "frontend": spec.frontend,
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
            "status": "pending",
        }
        context_path = _context_path(spec, out_root=out_root)
        _write_json(context_path, context)
        try:
            subprocess.run(command, check=True, cwd=REPO_ROOT)
            if not _valid_summary(
                summary_path, expected_method=spec.method,
                expected_checkpoint_keff_min=method_checkpoint_floor,
                expected_keff_min=method_training_floor,
            ):
                raise RuntimeError(f"Runner completed without a valid summary/protocol match: {summary_path}")
            context["status"] = "success"
            _write_json(context_path, context)
            _write_json(spec.run_dir / "experiment_context.json", context)
            rows.append(aggregate_run(spec, status="success"))
        except Exception as exc:
            context["status"] = "failure"
            context["error"] = str(exc)
            _write_json(context_path, context)
            rows.append(aggregate_run(spec, status="failure", error=str(exc)))
            if not continue_on_error:
                write_aggregate_outputs(rows, out_root=out_root, method=spec.method)
                raise
    return rows


def aggregate_run(spec: RunSpec, *, status: str, error: str = "") -> dict[str, object]:
    row: dict[str, object] = {
        "method": spec.method,
        "training_mode": spec.training_mode,
        "frontend": spec.frontend,
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
            "closure_quality",
            "closure_leakage",
            "I_available",
            "I_retained",
            "normalized_retained",
            "L_closure",
            "best_joint_fallback_used",
            "best_joint_selection",
        ]
        for key in fixed_keys:
            row[key] = summary.get(key)
        for key, value in summary.items():
            if key == "trained_delta_EI":
                row[key] = value
    return row


def write_aggregate_outputs(
    rows: Sequence[dict[str, object]],
    *,
    out_root: Path,
    method: str,
) -> tuple[Path, Path]:
    output_root = Path(out_root)
    method_root = output_root / method
    method_root.mkdir(parents=True, exist_ok=True)
    csv_path = method_root / "multiscale_summary.csv"
    manifest_path = method_root / "multiscale_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    method_spec = get_coarse_method_spec(method)
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "method": method,
            "training_mode": method_spec.training_mode,
            "frontend": method_spec.frontend or method_spec.name,
            "run_count": len(rows),
            "status_counts": pd.Series([row.get("status") for row in rows]).value_counts().to_dict(),
            "preparation_manifest": str(output_root / "_prepared_inputs" / "preparation_manifest.json"),
            "summary_csv": str(csv_path),
            "runs": list(rows),
        },
    )
    if method == DEFAULT_METHOD:
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, output_root / "multiscale_summary.csv")
        shutil.copy2(manifest_path, output_root / "multiscale_manifest.json")
    return csv_path, manifest_path


def _valid_summary(
    path: Path,
    *,
    expected_method: str | None = None,
    expected_checkpoint_keff_min: float | None = None,
    expected_keff_min: float | None = None,
) -> bool:
    try:
        payload = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not SUMMARY_REQUIRED_KEYS.issubset(payload):
        return False
    if expected_method is not None and payload.get("method") != expected_method:
        return False
    if (expected_checkpoint_keff_min is not None or expected_keff_min is not None) and expected_method == "shr_method_10":
        try:
            config = _read_json(Path(path).parent / "config.json")
            if expected_checkpoint_keff_min is not None:
                actual_checkpoint = config.get("checkpoint_keff_min")
                if actual_checkpoint is None or not math.isclose(
                    float(actual_checkpoint), float(expected_checkpoint_keff_min), rel_tol=0.0, abs_tol=1e-12
                ):
                    return False
            if expected_keff_min is not None:
                actual_keff = config.get("keff_min")
                if actual_keff is None:
                    actual_keff = 0.30 * int(config.get("k", payload.get("K", 0)))
                if not math.isclose(
                    float(actual_keff), float(expected_keff_min), rel_tol=0.0, abs_tol=1e-12
                ):
                    return False
            # A summary is reusable only when the exact selected epoch is
            # independently represented as an eligible positive-Delta-EI row.
            history_path = Path(path).parent / "training_history.csv"
            history = pd.read_csv(history_path)
            required_history = {"epoch", "delta_EI", "Keff_t", "Keff_tp", "checkpoint_eligible"}
            if not required_history <= set(history.columns):
                return False
            best_epoch = int(payload.get("best_epoch", -1))
            selected = history.loc[pd.to_numeric(history["epoch"], errors="coerce").eq(best_epoch)]
            if len(selected) != 1:
                return False
            selected_row = selected.iloc[0]
            selected_delta = float(selected_row["delta_EI"])
            summary_delta = float(payload.get("trained_delta_EI", payload["delta_EI_best_checkpoint"]))
            if (not bool(selected_row["checkpoint_eligible"]) or selected_delta <= 0.0
                    or not math.isclose(selected_delta, summary_delta, rel_tol=0.0, abs_tol=2e-4)):
                return False
            if expected_checkpoint_keff_min is not None and (
                float(selected_row["Keff_t"]) < float(expected_checkpoint_keff_min) - 1e-8
                or float(selected_row["Keff_tp"]) < float(expected_checkpoint_keff_min) - 1e-8
            ):
                return False
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError, KeyError):
            return False
    return True


def _runner_float_option(values: Sequence[str], option: str) -> float | None:
    values = list(map(str, values))
    for index, value in enumerate(values):
        if value == option:
            if index + 1 >= len(values):
                raise ValueError(f"Missing value after {option}")
            return float(values[index + 1])
        if value.startswith(option + "="):
            return float(value.split("=", 1)[1])
    return None


def _remove_run_dir(run_dir: Path, *, out_root: Path, method: str) -> None:
    resolved = run_dir.resolve()
    allowed_root = Path(out_root).resolve() / method
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove run directory outside {allowed_root}: {resolved}") from exc
    if len(relative.parts) != 3 or resolved == allowed_root:
        raise ValueError(f"Refusing to remove unexpected run-directory shape: {resolved}")
    shutil.rmtree(resolved)


def _context_path(spec: RunSpec, *, out_root: Path) -> Path:
    return (
        Path(out_root)
        / "_run_contexts"
        / spec.method
        / spec.scale
        / f"K{spec.k}"
        / f"{spec.source.resolved.stage}_to_{spec.target.resolved.stage}"
        / "experiment_context.json"
    )


def _validate_runner_extra_args(values: Sequence[str]) -> None:
    reserved = {
        "--method", "--out-dir", "--k", "--seed",
        "--h5ad-t", "--h5ad-tp", "--cci-t", "--cci-tp",
        "--cci-index-t", "--cci-index-tp", "--grn-t", "--grn-tp",
        "--maturity-t", "--maturity-tp",
    }
    blocked = sorted(
        token
        for token in map(str, values)
        if token in reserved or any(token.startswith(f"{name}=") for name in reserved)
    )
    if blocked:
        raise ValueError(
            "--runner-extra-args cannot override run identity or resolved inputs: "
            f"{blocked}."
        )


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
    if args.resume_incomplete and not args.resume:
        raise SystemExit("--resume-incomplete requires --resume")
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
        method=args.method,
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
        resume_incomplete=args.resume_incomplete,
        continue_on_error=args.continue_on_error,
    )
    csv_path, _ = write_aggregate_outputs(rows, out_root=args.out_root, method=args.method)
    print(f"Completed {len(rows)} multiscale runs; aggregate summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
