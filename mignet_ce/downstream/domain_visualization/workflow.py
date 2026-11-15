"""Domain-owned scheduling: bounded subprocesses and exact shared Stage 1 state."""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

from .variants import TIME_PAIRS, selected_variants


REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/run_wyt_deltaei_coarse_grain.py"
SCHEMA_VERSION = 1
STAGE1_ARTIFACTS = ("stage1_summary.json", "best_ei.pt", "metrics.csv", "config.json")
FINAL_ARTIFACTS = (
    "summary.json", "config.json", "best_ei.pt", "best_joint.pt", "metrics.csv",
    "checkpoint_consistency.json", "feature_manifest.json", "S_t.npy", "S_tp.npy",
    "assignments_t.csv", "assignments_tp.csv", "PIJ_micro_train.npy", "PIJ_macro_train.npy",
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    # Replace only this workflow's metadata; checkpoints/attempts remain immutable.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_digest() -> str:
    digest = hashlib.sha256()
    files = [RUNNER, Path(__file__), Path(__file__).with_name("variants.py")]
    files.extend(path for path in (REPO / "mignet_ce").rglob("*.py")
                 if path.relative_to(REPO / "mignet_ce").parts[0] != "downstream")
    files.extend((REPO / "wyt_deltaei_coarse_grain").rglob("*.py"))
    for path in sorted(files):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command(args, pair: tuple[str, str], variant, out: Path) -> list[str]:
    result = [sys.executable, "-B", str(RUNNER), "--method", "formal_two_stage", "--k", "40",
              "--seed", str(args.seed), "--epochs", str(args.epochs), "--device", args.device,
              "--out-dir", str(out), "--grn-feature-method", "legacy_nmf", "--cci-feature-method", "laplacian_hks",
              "--alpha-cci", "0.20", "--maturity-id-column", "unit_id", "--maturity-column", "pseudotime",
              "--lambda-gap", "1", "--ei-loss-retain-ratio", "1", "--ei-retain-ratio", "0.85",
              "--retain-floor-ratio", "0.5", "--checkpoint-retain-ratio", "0.5",
              "--low-signal-pair-weight", "0.1", "--stage1-ratio", "0.45"]
    for side, stage in zip(("t", "tp"), pair):
        stem = f"spot_heart_{stage}"
        paths = {
            "h5ad": args.data_root / "spot/heart" / f"{stem}.h5ad",
            "cci": args.cci_root / "spot" / f"{stem}_CCI_total.npz",
            "cci-index": args.cci_root / "spot" / f"{stem}_index.tsv",
            "grn": args.grn_root / "spot" / stem / "grn_edges.csv",
            "maturity": args.developmental_root / "spot" / f"heart_{stage}_features.csv",
        }
        for flag, path in paths.items():
            result.extend((f"--{flag}-{side}", str(path.resolve())))
    return [*result, *variant.flags]


def parsed_command(args, pair, variant):
    from scripts.run_wyt_deltaei_coarse_grain import build_argparser

    return build_argparser().parse_args(command(args, pair, variant, Path("unused"))[3:])


def preflight(args) -> dict:
    """Resolve and validate every training input before creating output folders."""
    from mignet_ce.io.h5ad_h5py import read_h5ad_axis_names
    from wyt_deltaei_coarse_grain.training_config import build_training_config
    import numpy as np
    import pandas as pd
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment; use the intended CUDA server.")
    input_paths = set()
    variants = selected_variants(args)
    for pair in TIME_PAIRS:
        parsed = parsed_command(args, pair, variants[0])
        for key in ("h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "cci_index_t", "cci_index_tp",
                    "grn_t", "grn_tp", "maturity_t", "maturity_tp"):
            input_paths.add(getattr(parsed, key))
    missing = sorted(str(path) for path in input_paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing formal training inputs:\n  " + "\n  ".join(missing))
    for stage in sorted({stage for pair in TIME_PAIRS for stage in pair}):
        path = args.developmental_root / "spot" / f"heart_{stage}_features.csv"
        frame = pd.read_csv(path, dtype={"unit_id": str})
        if not {"unit_id", "pseudotime"} <= set(frame):
            raise ValueError(f"Maturity CSV requires unit_id and pseudotime: {path}")
        ids = frame["unit_id"]
        if ids.isna().any() or ids.duplicated().any() or ids.str.strip().eq("").any():
            raise ValueError(f"Invalid maturity unit IDs: {path}")
        obs, _ = read_h5ad_axis_names(args.data_root / "spot/heart" / f"spot_heart_{stage}.h5ad")
        if set(ids) != set(map(str, obs)) or not np.isfinite(pd.to_numeric(frame["pseudotime"], errors="raise")).all():
            raise ValueError(f"Maturity values or H5AD unit IDs disagree: {path}")
    configurations = {}
    for variant in variants:
        configuration = asdict(build_training_config(parsed_command(args, TIME_PAIRS[0], variant)))
        for field in ("out_dir", "stage1_only", "stage1_source"):
            configuration.pop(field)
        configurations[variant.name] = configuration
    return {
        "schema_version": SCHEMA_VERSION, "experiment": getattr(args, "experiment", None) or "factorial",
        "method": "formal_two_stage", "input_scale": "spot", "k": 40,
        "time_pairs": [list(pair) for pair in TIME_PAIRS], "seed": args.seed, "epochs": args.epochs,
        "device": args.device, "data_root": str(args.data_root.resolve()),
        "input_sha256": {str(path): file_digest(path) for path in sorted(input_paths)},
        "training_code_sha256": training_digest(), "configurations": configurations,
        "stage1_groups": {variant.name: variant.stage1_group for variant in variants},
    }


def _execute(job: dict) -> Path:
    """Reuse only verified complete attempts; append a new attempt after a failure."""
    parent = job["parent"]
    parent.mkdir(parents=True, exist_ok=True)
    identity = {"command": job["command"], "cache_sha256": job["cache_sha256"],
                "source_sha256": job.get("source_sha256")}
    required = STAGE1_ARTIFACTS if job["stage1"] else FINAL_ARTIFACTS
    attempts = sorted(parent.glob("attempt_*.json"))
    for metadata in reversed(attempts):
        try:
            context = json.loads(metadata.read_text(encoding="utf-8"))
            run = metadata.with_suffix("")
            if (context.get("identity") == identity and context.get("status") == "complete"
                    and set(context["artifacts"]) == set(required)
                    and all(file_digest(run / name) == context["artifacts"][name] for name in required)):
                print(f"[reuse] {run}", flush=True)
                return run
        except (OSError, ValueError, KeyError):
            continue
    number = max((int(path.stem.rsplit("_", 1)[1]) for path in attempts), default=0) + 1
    run = parent / f"attempt_{number:04d}"
    metadata, log_path = run.with_suffix(".json"), run.with_suffix(".log")
    actual = list(job["command"])
    actual[actual.index("--out-dir") + 1] = str(run)
    context = {"identity": identity, "command": actual, "status": "running"}
    save_json(metadata, context)
    print(f"[start] {run}", flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(actual, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if process.returncode:
            raise RuntimeError(f"Training exited {process.returncode}. See {log_path}\n" +
                               "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
        context.update(status="complete", artifacts={name: file_digest(run / name) for name in required})
        save_json(metadata, context)
    except BaseException as error:
        context.update(status="failed", error=str(error))
        save_json(metadata, context)
        raise
    print(f"[done] {run}", flush=True)
    return run


def _execute_pair(job: dict) -> dict:
    ready = job["stage1_ready"]
    with job["stage1_lock"]:
        if not ready.done():
            try:
                ready.set_result(_execute(job["stage1_job"]))
            except Exception as error:
                ready.set_exception(error)
    source_run = ready.result()
    source = source_run / "best_ei.pt"
    stage2 = dict(job["stage2_job"])
    stage2["command"] = [*stage2["command"], "--stage1-source", str(source)]
    stage2["source_sha256"] = file_digest(source)
    run = _execute(stage2)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage1_checkpoint_sha256") != file_digest(source) or file_digest(run / "best_ei.pt") != file_digest(source):
        raise ValueError(f"Stage 2 did not use its assigned Stage 1 checkpoint: {run}")
    return {"variant": job["key"][0], "time_pair": list(job["key"][1]),
            "stage1_run": str(source_run.resolve()), "stage2_run": str(run.resolve()),
            "artifacts": {name: file_digest(run / name) for name in FINAL_ARTIFACTS}}


def build_jobs(args, root: Path, caches: dict) -> list[dict]:
    variants = selected_variants(args)
    groups = {}
    for variant in variants:
        groups.setdefault(variant.stage1_group, variant)
    shared = {(group, pair): (Lock(), Future()) for group in groups for pair in TIME_PAIRS}
    jobs = []
    for variant in variants:
        for pair in TIME_PAIRS:
            key = "_to_".join(pair)
            cache, digest = caches[pair]
            source = groups[variant.stage1_group]
            source_command = command(args, pair, source, Path("unused"))
            stage1 = {"stage1": True, "parent": root / "stage1" / variant.stage1_group / key,
                      "command": [*source_command, "--stage1-only", "--prepared-cache", str(cache)],
                      "cache_sha256": digest}
            stage2 = {"stage1": False, "parent": root / "runs" / variant.name / key,
                      "command": [*command(args, pair, variant, Path("unused")), "--prepared-cache", str(cache)],
                      "cache_sha256": digest}
            lock, ready = shared[variant.stage1_group, pair]
            jobs.append({"key": (variant.name, pair), "stage1_job": stage1, "stage2_job": stage2,
                         "stage1_lock": lock, "stage1_ready": ready})
    return jobs


def run(args) -> Path:
    from filelock import FileLock
    from joblib.externals import cloudpickle
    from mignet_ce.coarse_frontends import CoarseFrontendRequest, prepare_coarse_input

    plan = preflight(args)
    variants = selected_variants(args)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".running.lock", timeout=0):
        config_path = root / "run_config.json"
        if args.mode == "resume":
            if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != plan:
                raise ValueError("Resume requires identical training inputs, code and settings.")
        else:
            if any(path.name not in {".running.lock", "previews", "history_audits"} for path in root.iterdir()):
                raise FileExistsError(f"Use a fresh output directory or --mode resume: {root}")
            save_json(config_path, plan)
        manifest = {"schema_version": SCHEMA_VERSION, "status": "running", "run_config_sha256": file_digest(config_path),
                    "experiment": getattr(args, "experiment", None) or "factorial",
                    "workers": args.workers, "schedule": "variant_time_pair_stage1_then_stage2",
                    "stage1_groups": plan["stage1_groups"], "results": [], "errors": []}
        manifest_path = root / "run_manifest.json"
        save_json(manifest_path, manifest)
        try:
            caches = {}
            cache_dir = root / "_prepared"
            cache_dir.mkdir(exist_ok=True)
            for pair in TIME_PAIRS:
                cache = cache_dir / ("_to_".join(pair) + ".pkl")
                checksum = cache.with_suffix(".sha256")
                if cache.is_file() and checksum.is_file():
                    if file_digest(cache) != checksum.read_text().strip():
                        raise ValueError(f"Prepared input cache checksum mismatch: {cache}")
                else:
                    if cache.exists() or checksum.exists():
                        raise ValueError(f"Incomplete input cache; use a fresh output directory: {cache}")
                    parsed = parsed_command(args, pair, variants[0])
                    request = CoarseFrontendRequest(**{field.name: getattr(parsed, field.name)
                                                       for field in fields(CoarseFrontendRequest) if hasattr(parsed, field.name)})
                    print(f"[prepare] {' -> '.join(pair)}", flush=True)
                    prepared = prepare_coarse_input("formal_two_stage", request)
                    prepared.validate()
                    with cache.open("xb") as handle:
                        cloudpickle.dump(prepared, handle)
                    checksum.write_text(file_digest(cache) + "\n", encoding="utf-8")
                    del prepared
                caches[pair] = (cache, file_digest(cache))
            jobs = build_jobs(args, root, caches)
            # Prepare shared parent folders before threads touch them (also on OneDrive/Windows).
            for job in jobs:
                for stage in ("stage1_job", "stage2_job"):
                    job[stage]["parent"].mkdir(parents=True, exist_ok=True)
            stage1_count = len({item.stage1_group for item in variants}) * len(TIME_PAIRS)
            print(f"[schedule] {len(jobs)} tasks, {args.workers} workers; "
                  f"{stage1_count} shared Stage 1 runs, {len(jobs)} Stage 2 runs", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending = {executor.submit(_execute_pair, job): job["key"] for job in jobs}
                for future in as_completed(pending):
                    try:
                        manifest["results"].append(future.result())
                    except Exception as error:
                        manifest["errors"].append({"task": str(pending[future]), "error": str(error)})
                    save_json(manifest_path, manifest)
            if manifest["errors"]:
                raise RuntimeError("Domain experiments failed:\n" + "\n".join(item["error"] for item in manifest["errors"]))
            manifest["status"] = "complete"
            save_json(manifest_path, manifest)
        except BaseException as error:
            manifest.update(status="failed", error=str(error))
            save_json(manifest_path, manifest)
            raise
    return root
