"""Shared pair preparation, reusable Stage 1 jobs and isolated subprocess runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys

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
        raise RuntimeError(f"Training failed with exit {result.returncode}. Inspect {log_file}")
    print(f"[done] {parent.relative_to(job['root'])}", flush=True)
    return run_dir


def _parallel(jobs: list[dict], workers: int) -> dict:
    completed = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute, job): job["key"] for job in jobs}
        for future in as_completed(futures):
            try:
                completed[futures[future]] = future.result()
            except Exception as error:
                errors.append(str(error))
    if errors:
        raise RuntimeError("\n".join(errors))
    return completed


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
    plan = {"protocol": "formal_two_stage_v1", "manifest_version": 2, "input_sha256": inputs,
            "training_code_sha256": _training_code_digest(),
            "epochs": args.epochs, "seed": args.seed, "device": args.device}
    root.mkdir(parents=True, exist_ok=True)
    lock_file = root / ".running.lock"
    lock = FileLock(lock_file, timeout=0)
    lock.acquire()
    round_file = None
    try:
        config_file = root / "run_config.json"
        if args.resume:
            saved = json.loads(config_file.read_text(encoding="utf-8"))
            if saved.get("manifest_version") != 2:
                raise ValueError("This folder uses the previous whole-matrix manifest. Start branch rounds in a fresh output folder.")
            if saved != plan:
                raise ValueError("Saved inputs, code or training settings differ from this run.")
        else:
            if any(path != lock_file for path in root.iterdir()):
                raise FileExistsError(f"Use a fresh output folder or --resume: {root}")
            config_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        report_dir = root / "reports" / report_name(selected)
        report_dir.mkdir(parents=True, exist_ok=True)
        round_manifest = {
            "status": "running", "shared_config": str(config_file),
            "branches": {variant.name: list(variant.flags) for variant in selected},
            "workers": args.workers,
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

        stage1_jobs = []
        for variant in required_stage1_variants(selected):
            for pair in TIME_PAIRS:
                cache, checksum = caches[pair]
                cmd = command(args, pair, variant, Path("unused"))
                stage1_jobs.append({"key": (variant.name, pair), "root": root, "stage1": True,
                    "parent": root / "stage1" / variant.name / "_to_".join(pair),
                    "command": [*cmd, "--stage1-only", "--prepared-cache", str(cache)],
                    "cache_sha256": checksum, "source_sha256": None})
        stage1_results = _parallel(stage1_jobs, args.workers)

        stage2_jobs = []
        for variant in selected:
            for pair in TIME_PAIRS:
                cache, checksum = caches[pair]
                source = stage1_results[variant.stage1_group, pair] / "best_ei.pt"
                cmd = command(args, pair, variant, Path("unused"))
                stage2_jobs.append({"key": (variant.name, pair), "root": root, "stage1": False,
                    "parent": root / "stage2" / variant.name / "_to_".join(pair),
                    "command": [*cmd, "--stage1-source", str(source), "--prepared-cache", str(cache)],
                    "cache_sha256": checksum, "source_sha256": file_digest(source)})
        stage2_results = _parallel(stage2_jobs, args.workers)
        rows = []
        for variant in selected:
            for pair in TIME_PAIRS:
                run_dir = stage2_results[variant.name, pair]
                rows.extend(checkpoint_rows(run_dir, method=variant.label, scale="spot", pair=pair, k=40))
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                source = stage1_results[variant.stage1_group, pair] / "best_ei.pt"
                if summary["stage1_checkpoint_sha256"] != file_digest(source):
                    raise ValueError(f"Restored Stage 1 checkpoint hash mismatch: {run_dir}")
        table_grid(rows, ("spot",), methods)
        write_tables(rows, report_dir, baseline_method=VARIANTS[0].label)
        draw_tables(rows, report_dir, ("spot",), methods=methods, baseline_method=VARIANTS[0].label)
        round_manifest.update(status="complete", results=[
            {"variant": variant.name, "time_pair": list(pair),
             "stage1_run": str(stage1_results[variant.stage1_group, pair]),
             "stage2_run": str(stage2_results[variant.name, pair])}
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
