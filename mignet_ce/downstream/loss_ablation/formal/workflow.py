"""Shared pair preparation, reusable Stage 1 jobs and isolated subprocess runs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock

from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL

from .variants import TIME_PAIRS, VARIANTS, report_name, required_stage1_variants, select_variants


REPO = Path(__file__).resolve().parents[4]
RUNNER = REPO / "scripts" / "run_wyt_deltaei_coarse_grain.py"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_code_digest() -> str:
    """Presentation and round selection can evolve independently of the learner."""
    code = hashlib.sha256()
    mignet = REPO / "mignet_ce"
    sources = [RUNNER, *sorted(path for path in mignet.rglob("*.py")
                              if path.relative_to(mignet).parts[0] != "downstream"),
               *sorted((REPO / "wyt_deltaei_coarse_grain").rglob("*.py"))]
    for path in sources:
        code.update(str(path.relative_to(REPO)).replace("\\", "/").encode())
        code.update(path.read_bytes())
    return code.hexdigest()


def command(args, pair: tuple[str, str], variant, out: Path) -> list[str]:
    source, target = pair
    result = [sys.executable, str(RUNNER), "--method", "formal_two_stage", "--k", "40",
              "--seed", str(args.seed), "--epochs", str(args.epochs), "--device", args.device,
              "--out-dir", str(out), "--grn-feature-method", "legacy_nmf",
              "--cci-feature-method", "laplacian_hks", "--alpha-cci", "0.20",
              "--maturity-id-column", "unit_id", "--maturity-column", "pseudotime"]
    for suffix, time in (("t", source), ("tp", target)):
        sample = f"spot_heart_{time}"
        paths = {
            "h5ad": args.data_root / "spot" / "heart" / f"{sample}.h5ad",
            "cci": args.cci_root / "spot" / f"{sample}_CCI_total.npz",
            "cci-index": args.cci_root / "spot" / f"{sample}_index.tsv",
            "grn": args.grn_root / "spot" / sample / "grn_edges.csv",
            "maturity": args.developmental_root / "spot" / f"heart_{time}_features.csv",
        }
        for name, path in paths.items():
            result.extend((f"--{name}-{suffix}", str(path.resolve())))
    return [*result, *variant.flags]


def _execute(job: dict) -> Path:
    parent = job["parent"]
    parent.mkdir(parents=True, exist_ok=True)
    identity = {"command": job["command"], "cache_sha256": job["cache_sha256"],
                "source_sha256": job["source_sha256"]}
    attempts = sorted(parent.glob("attempt_*.json"))
    for context_file in reversed(attempts):
        try:
            context = json.loads(context_file.read_text(encoding="utf-8"))
            run_dir = context_file.with_suffix("")
            if context.get("identity") == identity and context.get("status") == "complete" and all(
                file_digest(run_dir / name) == checksum for name, checksum in context["artifacts"].items()
            ):
                print(f"[reuse] {parent.relative_to(job['root'])}", flush=True)
                return run_dir
        except (OSError, ValueError, KeyError):
            continue
    number = max((int(path.stem.split("_")[1]) for path in attempts), default=0) + 1
    run_dir = parent / f"attempt_{number:04d}"
    context_file = run_dir.with_suffix(".json")
    log_file = run_dir.with_suffix(".log")
    actual_command = list(job["command"])
    actual_command[actual_command.index("--out-dir") + 1] = str(run_dir)
    context = {"identity": identity, "command": actual_command, "status": "running"}
    context_file.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    print(f"[start] {parent.relative_to(job['root'])}", flush=True)
    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(actual_command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    context["status"] = "complete" if result.returncode == 0 else "failed"
    if result.returncode == 0:
        names = ("stage1_summary.json", "best_ei.pt", "metrics.csv", "config.json") if job["stage1"] else (
            "summary.json", "best_joint.pt", "best_ei.pt", "metrics.csv", "config.json",
        )
        context["artifacts"] = {name: file_digest(run_dir / name) for name in names}
    context_file.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
        raise RuntimeError(f"Training failed with exit {result.returncode}. Inspect {log_file}\n{tail}")
    print(f"[done] {parent.relative_to(job['root'])}", flush=True)
    return run_dir


def _execute_pair(job: dict) -> dict[str, Path]:
    """One worker owns a time pair through both stages, with no batch barrier."""
    source_job = job["stage1_job"]
    ready = job["stage1_ready"]
    # Stage-2-only branches consume the exact same completed state, optimizer
    # and RNG as their baseline. Any task may produce that group's Stage 1.
    with job["stage1_lock"]:
        if not ready.done():
            try:
                ready.set_result(_execute(source_job))
            except Exception as error:
                ready.set_exception(error)
    source_run = ready.result()
    source = source_run / "best_ei.pt"
    stage2 = dict(job["stage2_job"])
    stage2["command"] = [*stage2["command"], "--stage1-source", str(source)]
    stage2["source_sha256"] = file_digest(source)
    # This happens as soon as this pair is ready, even while other pairs or
    # methods are still in Stage 1. _execute returns only after its child exits.
    return {"stage1_run": source_run, "stage2_run": _execute(stage2)}


def _parallel(jobs: list[dict], workers: int) -> dict:
    completed = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_pair if "stage1_job" in job else _execute, job): job["key"]
                   for job in jobs}
        for future in as_completed(futures):
            try:
                completed[futures[future]] = future.result()
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError("\n".join(errors))
    return completed


def _verified_baseline_runs(args, plan: dict, caches: dict) -> tuple[dict, dict]:
    """Audit old 1B artifacts without rewriting or relabelling their files."""
    import math
    import torch
    from joblib.externals import cloudpickle
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    from wyt_deltaei_coarse_grain.two_stage_trainer import formal_stage1_source_matches
    from ..tables import checkpoint_rows

    previous_root = args.reuse_baseline_from.resolve()
    previous = json.loads((previous_root / "run_config.json").read_text(encoding="utf-8"))
    if previous.get("protocol") != "formal_two_stage_v1" or any(
        previous.get(key) != plan[key] for key in ("input_sha256", "epochs", "seed", "device")
    ):
        raise ValueError("Previous 1B inputs or training settings do not match the new baseline.")
    results, audit = {}, {"source_root": str(previous_root), "source_protocol": previous["protocol"], "runs": []}
    for pair in TIME_PAIRS:
        parent = previous_root / "stage2" / "branch1b" / "_to_".join(pair)
        candidates = []
        for context_file in sorted(parent.glob("attempt_*.json"), reverse=True):
            context = json.loads(context_file.read_text(encoding="utf-8"))
            if context.get("status") == "complete":
                candidates.append((context_file.with_suffix(""), context))
        if not candidates:
            raise ValueError(f"No completed previous Branch 1B run for {pair}: {parent}")
        run_dir, context = candidates[0]
        artifact_names = ("summary.json", "best_ei.pt", "best_joint.pt", "metrics.csv", "config.json")
        for name in artifact_names:
            if file_digest(run_dir / name) != context.get("artifacts", {}).get(name):
                raise ValueError(f"Previous 1B artifact checksum mismatch: {run_dir / name}")
        parsed = build_argparser().parse_args(command(args, pair, VARIANTS[0], Path("unused"))[2:])
        expected = json.loads(json.dumps(asdict(build_training_config(parsed)), default=str))
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        execution_fields = {"out_dir", "stage1_source", "stage1_only"}
        if {key: value for key, value in saved.items() if key not in execution_fields} != {
            key: value for key, value in expected.items() if key not in execution_fields
        }:
            raise ValueError(f"Previous 1B loss/model configuration does not match: {run_dir}")
        with caches[pair][0].open("rb") as stream:
            prepared = cloudpickle.load(stream)
        checkpoint = torch.load(run_dir / "best_ei.pt", map_location="cpu", weights_only=False)
        config = build_training_config(parsed)
        if not formal_stage1_source_matches(prepared, config, checkpoint.get("stage1_signature")):
            raise ValueError(f"Previous 1B Stage 1 input/state fingerprint mismatch: {run_dir}")
        for key in ("encoder", "macro_net", "optimizer", "torch_rng_state", "cuda_rng_states"):
            if key not in checkpoint:
                raise ValueError(f"Previous 1B Stage 1 state is incomplete ({key}): {run_dir}")
        with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        stage1_epochs = min(config.epochs - 1, max(1, round(config.epochs * config.stage1_ratio)))
        if len(history) != config.epochs or len(checkpoint.get("stage1_history", [])) != stage1_epochs or any(
            int(row["epoch"]) != epoch or row["stage"] != ("stage1" if epoch <= stage1_epochs else "stage2")
            for epoch, row in enumerate(history, 1)
        ):
            raise ValueError(f"Previous 1B training history is incomplete: {run_dir}")
        for row, state_row in zip(history[:stage1_epochs], checkpoint["stage1_history"]):
            for key in ("delta_EI", "I_available", "I_retained", "closure_quality"):
                if not math.isclose(float(row[key]), float(state_row[key]), rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError(f"Previous 1B history/checkpoint mismatch: {run_dir}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        joint = torch.load(run_dir / "best_joint.pt", map_location="cpu", weights_only=False)
        if summary["stage1_checkpoint_sha256"] != file_digest(run_dir / "best_ei.pt") or (
            int(summary["stage1_best_epoch"]) != int(checkpoint["epoch"])
            or int(summary["best_epoch"]) != int(joint["epoch"])
        ):
            raise ValueError(f"Previous 1B selected checkpoint identity mismatch: {run_dir}")
        for checkpoint_key, summary_key in (
            ("delta_EI", "delta_EI_best_checkpoint"), ("I_available", "I_available"),
            ("I_retained", "I_retained"), ("closure_quality", "closure_quality"),
        ):
            if not math.isclose(float(joint[checkpoint_key]), float(summary[summary_key]), rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError(f"Previous 1B Stage 2 state/summary mismatch: {run_dir}")
        checkpoint_rows(run_dir, method=VARIANTS[0].label, scale="spot", pair=pair, k=40)
        results["baseline", pair] = {"stage1_run": run_dir, "stage2_run": run_dir}
        audit["runs"].append({"time_pair": list(pair), "run_dir": str(run_dir),
                              "artifacts": {name: file_digest(run_dir / name) for name in artifact_names}})
    return results, audit


def run(args) -> None:
    from filelock import FileLock
    from joblib.externals import cloudpickle
    from mignet_ce.coarse_frontends import CoarseFrontendRequest, prepare_coarse_input
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from ..plots import draw_tables
    from ..tables import checkpoint_rows, table_grid, write_tables

    root = args.output_root.resolve()
    selected = select_variants(getattr(args, "branches", None))
    methods = tuple(variant.label for variant in selected)
    inputs = {}
    for pair in TIME_PAIRS:
        parsed = build_argparser().parse_args(command(args, pair, VARIANTS[0], Path("unused"))[2:])
        for name in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "cci_index_t", "cci_index_tp",
                     "grn_t", "grn_tp", "maturity_t", "maturity_tp"):
            path = getattr(parsed, name)
            inputs[str(path)] = file_digest(path)
    plan = {"protocol": FORMAL_TWO_STAGE_PROTOCOL, "manifest_version": 3, "input_sha256": inputs,
            "training_code_sha256": _training_code_digest(),
            "epochs": args.epochs, "seed": args.seed, "device": args.device}
    reuse_root = getattr(args, "reuse_baseline_from", None)
    if reuse_root is not None:
        plan["baseline_reuse_root"] = str(reuse_root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    lock_file = root / ".running.lock"
    lock = FileLock(lock_file, timeout=0)
    lock.acquire()
    round_file = None
    try:
        config_file = root / "run_config.json"
        if args.resume:
            saved = json.loads(config_file.read_text(encoding="utf-8"))
            if saved.get("manifest_version") != 3:
                raise ValueError("This folder uses a previous loss protocol. Start the 1B baseline in a fresh output folder.")
            if saved != plan:
                old_settings = {key: value for key, value in saved.items() if key != "training_code_sha256"}
                new_settings = {key: value for key, value in plan.items() if key != "training_code_sha256"}
                finished = any(root.glob("stage1/*/*/*/stage1_summary.json")) or any(root.glob("stage2/*/*/*/summary.json"))
                if old_settings != new_settings or finished:
                    raise ValueError("Saved inputs, code or training settings differ from this run.")
                previous = sorted(root.glob("run_config.previous_*.json"))
                revision = max((int(path.stem.rsplit("_", 1)[1]) for path in previous), default=0) + 1
                backup = root / f"run_config.previous_{revision:04d}.json"
                shutil.copyfile(config_file, backup)
                config_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                print(f"[resume] Updated code fingerprint for the failed run; previous configuration at {backup}", flush=True)
        else:
            if any(path != lock_file for path in root.iterdir()):
                raise FileExistsError(f"Use a fresh output folder or --resume: {root}")
            config_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        report_dir = root / "reports" / report_name(selected)
        report_dir.mkdir(parents=True, exist_ok=True)
        round_manifest = {
            "status": "running", "shared_config": str(config_file),
            "branches": {variant.name: list(variant.flags) for variant in selected},
            "workers": min(args.workers, len(selected) * len(TIME_PAIRS)), "requested_workers": args.workers,
            "schedule": "method_time_pair_stage1_then_stage2",
        }
        round_file = report_dir / "run_manifest.json"
        round_file.write_text(json.dumps(round_manifest, indent=2) + "\n", encoding="utf-8")

        caches = {}
        for pair in TIME_PAIRS:
            key = "_to_".join(pair)
            cache_dir = root / "_prepared"
            cache_dir.mkdir(exist_ok=True)
            cache = cache_dir / f"{key}.pkl"
            checksum_file = cache.with_suffix(".sha256")
            if cache.is_file() and checksum_file.is_file():
                if file_digest(cache) != checksum_file.read_text().strip():
                    raise ValueError(f"Prepared cache checksum mismatch: {cache}")
            else:
                if cache.exists():
                    raise ValueError(f"Incomplete prepared cache, use a fresh output folder: {cache}")
                parsed = build_argparser().parse_args(command(args, pair, VARIANTS[0], Path("unused"))[2:])
                request = CoarseFrontendRequest(**{field.name: getattr(parsed, field.name)
                                                  for field in fields(CoarseFrontendRequest) if hasattr(parsed, field.name)})
                print(f"[prepare] Spot {key}", flush=True)
                prepared = prepare_coarse_input("formal_two_stage", request)
                with cache.open("wb") as stream:
                    cloudpickle.dump(prepared, stream)
                checksum_file.write_text(file_digest(cache) + "\n")
                del prepared
            caches[pair] = (cache, file_digest(cache))

        results = {}
        if reuse_root is not None:
            results, audit = _verified_baseline_runs(args, plan, caches)
            audit_file = root / "baseline_reuse.json"
            if audit_file.exists() and json.loads(audit_file.read_text(encoding="utf-8")) != audit:
                raise ValueError("Previously imported baseline artifacts have changed.")
            audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            round_manifest["baseline_reuse_audit"] = str(audit_file)
            print("[reuse] Verified previous Branch 1B as the new baseline", flush=True)
        stage1_variants = {variant.name: variant for variant in required_stage1_variants(selected)}
        shared_stage1 = {}
        for group in stage1_variants:
            for pair in TIME_PAIRS:
                ready = Future()
                if (group, pair) in results:
                    ready.set_result(results[group, pair]["stage1_run"])
                shared_stage1[group, pair] = (Lock(), ready)
        jobs = []
        for variant in selected:
            for pair in TIME_PAIRS:
                if (variant.name, pair) in results:
                    continue
                cache, checksum = caches[pair]
                source_variant = stage1_variants[variant.stage1_group]
                source_command = command(args, pair, source_variant, Path("unused"))
                stage1_job = {"root": root, "stage1": True,
                        "parent": root / "stage1" / source_variant.name / "_to_".join(pair),
                        "command": [*source_command, "--stage1-only", "--prepared-cache", str(cache)],
                        "cache_sha256": checksum, "source_sha256": None}
                stage2_job = {"root": root, "stage1": False,
                    "parent": root / "stage2" / variant.name / "_to_".join(pair),
                    "command": [*command(args, pair, variant, Path("unused")), "--prepared-cache", str(cache)],
                    "cache_sha256": checksum, "source_sha256": None}
                stage1_lock, ready = shared_stage1[variant.stage1_group, pair]
                jobs.append({"key": (variant.name, pair), "stage1_job": stage1_job, "stage2_job": stage2_job,
                             "stage1_lock": stage1_lock, "stage1_ready": ready})
        workers = min(args.workers, max(1, len(jobs)))
        round_manifest.update(workers=workers, training_tasks=len(jobs), reused_tasks=len(results))
        round_file.write_text(json.dumps(round_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[schedule] {len(jobs)} method/time-pair tasks; up to {workers} concurrent training processes; "
              "each pair proceeds directly from Stage 1 to Stage 2", flush=True)
        results.update(_parallel(jobs, workers))
        rows = []
        for variant in selected:
            for pair in TIME_PAIRS:
                pair_result = results[variant.name, pair]
                run_dir = pair_result["stage2_run"]
                rows.extend(checkpoint_rows(run_dir, method=variant.label, scale="spot", pair=pair, k=40))
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                source = pair_result["stage1_run"] / "best_ei.pt"
                if summary["stage1_checkpoint_sha256"] != file_digest(source):
                    raise ValueError(f"Restored Stage 1 checkpoint hash mismatch: {run_dir}")
        table_grid(rows, ("spot",), methods)
        write_tables(rows, report_dir, baseline_method=VARIANTS[0].label)
        draw_tables(rows, report_dir, ("spot",), methods=methods, baseline_method=VARIANTS[0].label)
        round_manifest.update(status="complete", results=[
            {"variant": variant.name, "time_pair": list(pair),
             "stage1_run": str(results[variant.name, pair]["stage1_run"]),
             "stage2_run": str(results[variant.name, pair]["stage2_run"])}
            for variant in selected for pair in TIME_PAIRS
        ])
        round_file.write_text(json.dumps(round_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Completed formal loss ablation round. Tables saved to {report_dir}", flush=True)
    except Exception as error:
        if round_file is not None:
            round_manifest.update(status="failed", error=str(error))
            round_file.write_text(json.dumps(round_manifest, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        lock.release()
