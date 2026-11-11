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

from mignet_ce.coarse_frontends.method_specs import (
    FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL,
    FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOLS,
    FORMAL_TWO_STAGE_MANIFEST_VERSIONS,
)

from .variants import (
    TIME_PAIRS, VARIANTS, comparison_predecessors, report_name, required_stage1_variants,
    select_variants, variants_for_protocol,
)


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
    """Audit a mathematically equivalent prior method as this protocol's baseline."""
    import math
    import torch
    from joblib.externals import cloudpickle
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    from wyt_deltaei_coarse_grain.two_stage_trainer import formal_stage1_source_matches
    from ..tables import checkpoint_rows

    previous_root = args.reuse_baseline_from.resolve()
    previous = json.loads((previous_root / "run_config.json").read_text(encoding="utf-8"))
    protocol = plan.get("protocol", FORMAL_TWO_STAGE_PROTOCOL)
    baseline = variants_for_protocol(protocol)[0]
    migrations = {
        (FORMAL_TWO_STAGE_PROTOCOL, FORMAL_TWO_STAGE_GAP_PROTOCOL): "branch5",
        (FORMAL_TWO_STAGE_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL): "baseline",
        (FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL): "branch3",
        (FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_GAP_PROTOCOL): "baseline",
        (FORMAL_TWO_STAGE_1B_PROTOCOL, FORMAL_TWO_STAGE_ORIGINAL_PROTOCOL): "branch1b",
        (FORMAL_TWO_STAGE_1B_PROTOCOL, FORMAL_TWO_STAGE_1B_PROTOCOL): "baseline",
    }
    source_variant = migrations.get((protocol, previous.get("protocol")))
    if source_variant is None or any(
        previous.get(key) != plan[key] for key in ("input_sha256", "epochs", "seed", "device")
    ):
        raise ValueError("Previous baseline inputs or training settings/protocol do not match the new baseline.")
    results, audit = {}, {"source_root": str(previous_root), "source_protocol": previous["protocol"],
                          "source_variant": source_variant, "target_protocol": protocol, "runs": []}
    for pair in TIME_PAIRS:
        parent = previous_root / "stage2" / source_variant / "_to_".join(pair)
        candidates = []
        for context_file in sorted(parent.glob("attempt_*.json"), reverse=True):
            context = json.loads(context_file.read_text(encoding="utf-8"))
            if context.get("status") == "complete":
                candidates.append((context_file.with_suffix(""), context))
        if not candidates:
            raise ValueError(f"No completed previous {source_variant} run for {pair}: {parent}")
        run_dir, context = candidates[0]
        artifact_names = ("summary.json", "best_ei.pt", "best_joint.pt", "metrics.csv", "config.json")
        for name in artifact_names:
            if file_digest(run_dir / name) != context.get("artifacts", {}).get(name):
                raise ValueError(f"Previous baseline artifact checksum mismatch: {run_dir / name}")
        parsed = build_argparser().parse_args(command(args, pair, baseline, Path("unused"))[2:])
        expected = json.loads(json.dumps(asdict(build_training_config(parsed)), default=str))
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        execution_fields = {"out_dir", "stage1_source", "stage1_only"}
        if {key: value for key, value in saved.items() if key not in execution_fields} != {
            key: value for key, value in expected.items() if key not in execution_fields
        }:
            raise ValueError(f"Previous baseline loss/model configuration does not match: {run_dir}")
        with caches[pair][0].open("rb") as stream:
            prepared = cloudpickle.load(stream)
        checkpoint = torch.load(run_dir / "best_ei.pt", map_location="cpu", weights_only=False)
        config = build_training_config(parsed)
        if not formal_stage1_source_matches(prepared, config, checkpoint.get("stage1_signature")):
            raise ValueError(f"Previous baseline Stage 1 input/state fingerprint mismatch: {run_dir}")
        for key in ("encoder", "macro_net", "optimizer", "torch_rng_state", "cuda_rng_states"):
            if key not in checkpoint:
                raise ValueError(f"Previous baseline Stage 1 state is incomplete ({key}): {run_dir}")
        with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        stage1_epochs = min(config.epochs - 1, max(1, round(config.epochs * config.stage1_ratio)))
        if len(history) != config.epochs or len(checkpoint.get("stage1_history", [])) != stage1_epochs or any(
            int(row["epoch"]) != epoch or row["stage"] != ("stage1" if epoch <= stage1_epochs else "stage2")
            for epoch, row in enumerate(history, 1)
        ):
            raise ValueError(f"Previous baseline training history is incomplete: {run_dir}")
        for row, state_row in zip(history[:stage1_epochs], checkpoint["stage1_history"]):
            for key in ("delta_EI", "I_available", "I_retained", "closure_quality"):
                if not math.isclose(float(row[key]), float(state_row[key]), rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError(f"Previous baseline history/checkpoint mismatch: {run_dir}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        joint = torch.load(run_dir / "best_joint.pt", map_location="cpu", weights_only=False)
        if summary["stage1_checkpoint_sha256"] != file_digest(run_dir / "best_ei.pt") or (
            int(summary["stage1_best_epoch"]) != int(checkpoint["epoch"])
            or int(summary["best_epoch"]) != int(joint["epoch"])
        ):
            raise ValueError(f"Previous baseline selected checkpoint identity mismatch: {run_dir}")
        for checkpoint_key, summary_key in (
            ("delta_EI", "delta_EI_best_checkpoint"), ("I_available", "I_available"),
            ("I_retained", "I_retained"), ("closure_quality", "closure_quality"),
        ):
            if not math.isclose(float(joint[checkpoint_key]), float(summary[summary_key]), rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError(f"Previous baseline Stage 2 state/summary mismatch: {run_dir}")
        checkpoint_rows(run_dir, method=baseline.label, scale="spot", pair=pair, k=40)
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
    plan = {"protocol": FORMAL_TWO_STAGE_PROTOCOL, "manifest_version": 5, "input_sha256": inputs,
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
            if saved.get("manifest_version") != 5 or saved.get("protocol") != FORMAL_TWO_STAGE_PROTOCOL:
                raise ValueError("This folder uses a previous loss protocol. Start the retain/CQ staircase in a fresh output folder.")
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
            "stage1_groups": {variant.name: variant.stage1_group for variant in selected},
            "comparison_predecessors": comparison_predecessors(selected),
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
            print(f"[reuse] Verified previous {audit['source_variant']} as the new baseline", flush=True)
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
        write_tables(rows, report_dir, baseline_method=VARIANTS[0].label,
                     predecessors=comparison_predecessors(selected))
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


def finalize(args) -> Path:
    """Recover an existing round using its tensors, history and selected weights."""
    from dataclasses import replace
    from types import SimpleNamespace
    from filelock import FileLock
    from joblib.externals import cloudpickle
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    from wyt_deltaei_coarse_grain.two_stage_trainer import audit_formal_run, finalize_two_stage_run
    from ..plots import draw_tables
    from ..tables import checkpoint_rows, table_grid, write_tables

    root = args.output_root.resolve()
    config_file = root / "run_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"No existing formal round to finalize: {root}")
    with FileLock(root / ".running.lock", timeout=0):
        plan = json.loads(config_file.read_text(encoding="utf-8"))
        protocol = plan.get("protocol")
        if protocol not in FORMAL_TWO_STAGE_PROTOCOLS or plan.get("manifest_version") not in FORMAL_TWO_STAGE_MANIFEST_VERSIONS[protocol]:
            raise ValueError("Finalization requires a supported recorded formal loss protocol.")
        selected = select_variants(getattr(args, "branches", None), protocol=protocol)
        methods = tuple(variant.label for variant in selected)
        if getattr(args, "device", None) not in (None, plan["device"]):
            raise ValueError("Finalization uses the recorded training device for checkpoint comparison.")
        for name, checksum in plan["input_sha256"].items():
            if file_digest(Path(name)) != checksum:
                raise ValueError(f"Original training input checksum mismatch: {name}")

        # A different evaluator is allowed only in this no-training mode. The
        # original training manifest and completed attempt records stay intact.
        code = hashlib.sha256(_training_code_digest().encode())
        for path in (Path(__file__), Path(__file__).with_name("__main__.py")):
            code.update(path.read_bytes())
        parent = root / "finalized" / report_name(selected)
        parent.mkdir(parents=True, exist_ok=True)
        attempts = list(parent.glob("attempt_*"))
        number = max((int(path.name.rsplit("_", 1)[1]) for path in attempts if path.is_dir()), default=0) + 1
        out = parent / f"attempt_{number:04d}"
        out.mkdir()
        manifest = {"status": "running", "mode": "finalize_only", "optimizer_steps": 0,
                    "loss_protocol": protocol, "source_run_config": str(config_file),
                    "source_run_config_sha256": file_digest(config_file),
                    "training_code_sha256": plan["training_code_sha256"],
                    "finalization_code_sha256": code.hexdigest(), "results": []}
        manifest_file = out / "run_manifest.json"

        def save_manifest():
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        save_manifest()
        try:
            imported = {}
            import_file = root / "baseline_reuse.json"
            if "baseline_reuse_root" in plan:
                imported_audit = json.loads(import_file.read_text(encoding="utf-8"))
                if imported_audit["source_root"] != plan["baseline_reuse_root"]:
                    raise ValueError("Imported baseline source differs from the training manifest.")
                imported = {tuple(item["time_pair"]): item for item in imported_audit["runs"]}
            rows = []
            caches = {}
            expected_args = SimpleNamespace(seed=plan["seed"], epochs=plan["epochs"], device=plan["device"],
                                            data_root=Path("."), cci_root=Path("."), grn_root=Path("."),
                                            developmental_root=Path("."))
            for variant in selected:
                for pair in TIME_PAIRS:
                    key = "_to_".join(pair)
                    if pair not in caches:
                        cache = root / "_prepared" / f"{key}.pkl"
                        checksum = file_digest(cache)
                        if checksum != cache.with_suffix(".sha256").read_text().strip():
                            raise ValueError(f"Prepared cache checksum mismatch: {cache}")
                        caches[pair] = (cache, checksum)
                    cache, checksum = caches[pair]
                    imported_run = imported.get(pair) if variant.name == "baseline" else None
                    if imported_run is not None:
                        source = Path(imported_run["run_dir"])
                        context = {"status": "complete", "artifacts": imported_run["artifacts"]}
                    else:
                        source_parent = root / "stage2" / variant.name / key
                        contexts = sorted(source_parent.glob("attempt_*.json"), reverse=True)
                        if not contexts:
                            raise FileNotFoundError(f"No trained checkpoint for finalization: {source_parent}")
                        context_file = contexts[0]
                        context = json.loads(context_file.read_text(encoding="utf-8"))
                        source = context_file.with_suffix("")
                        if context.get("status") not in {"complete", "failed"}:
                            raise ValueError(f"Training attempt is not finalized: {context_file}")
                        if context["identity"]["cache_sha256"] != checksum:
                            raise ValueError(f"Training attempt prepared-cache identity mismatch: {source}")
                        source_hash = context["identity"]["source_sha256"]
                        if source_hash is None or file_digest(source / "best_ei.pt") != source_hash:
                            raise ValueError(f"Training attempt Stage 1 checkpoint identity mismatch: {source}")
                    names = ("config.json", "metrics.csv", "best_ei.pt", "best_joint.pt")
                    source_hashes = {name: file_digest(source / name) for name in names}
                    if context["status"] == "complete":
                        for name in (*names, "summary.json"):
                            actual = file_digest(source / name)
                            if actual != context.get("artifacts", {}).get(name):
                                raise ValueError(f"Completed training artifact checksum mismatch: {source / name}")
                            source_hashes[name] = actual
                    parsed = build_argparser().parse_args(command(expected_args, pair, variant, out / "unused")[2:])
                    expected = build_training_config(parsed)
                    with cache.open("rb") as stream:
                        prepared = cloudpickle.load(stream)
                    # Check all completed and failed sources before selecting a
                    # report result. A failed attempt must have all epochs and
                    # matching saved checkpoint metrics, not just a .pt file.
                    audited = audit_formal_run(prepared, expected, source)
                    del audited
                    if context["status"] == "complete":
                        run_dir = source
                        action = "reused_complete"
                        print(f"[reuse] completed {variant.name}/{key}", flush=True)
                    else:
                        run_dir = out / "runs" / variant.name / key
                        print(f"[finalize] {variant.name}/{key}; zero optimizer steps", flush=True)
                        finalize_two_stage_run(prepared, replace(expected, out_dir=run_dir), source)
                        action = "recomputed_checkpoint"
                    del prepared
                    if any(file_digest(source / name) != value for name, value in source_hashes.items()):
                        raise ValueError(f"Training source changed during finalization: {source}")
                    rows.extend(checkpoint_rows(run_dir, method=variant.label, scale="spot", pair=pair, k=40))
                    manifest["results"].append({"variant": variant.name, "time_pair": list(pair),
                        "action": action, "training_source": str(source), "run_dir": str(run_dir),
                        "source_artifacts": source_hashes})
                    save_manifest()
            table_grid(rows, ("spot",), methods)
            write_tables(rows, out, baseline_method=selected[0].label,
                         predecessors=comparison_predecessors(selected) if protocol in (
                             FORMAL_TWO_STAGE_GAP_PROTOCOL, FORMAL_TWO_STAGE_PROTOCOL,
                         ) else None)
            draw_tables(rows, out, ("spot",), methods=methods, baseline_method=selected[0].label)
            manifest.update(status="complete")
            save_manifest()
            print(f"Finalized tables with zero training steps: {out}", flush=True)
            return out
        except Exception as error:
            manifest.update(status="failed", error=str(error))
            save_manifest()
            raise
