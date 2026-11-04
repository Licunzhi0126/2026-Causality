"""Isolated organ/hierarchy jobs, resumable provenance, and incremental tables.

This is independent retraining within each organ, not frozen Heart-to-Lung transfer.
Each column is an independent learned coarse graining; their gains need not add up.
No training algorithm or checkpoint eligibility rule is modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from mignet_ce.io.organ_paths import resolve_organ_paths

PROTOCOL = "organ_independent_trained_deltaei_v1"
METHOD_ALIASES = {
    "two_stage": "maturity_cci_grn_two_stage",
    "maturity_cci_grn_two_stage": "maturity_cci_grn_two_stage",
    "shr_method_10": "shr_method_10",
}
HIERARCHIES = {
    "spot_to_k150": ("Spot -> K150", "spot", 150),
    "k150_to_k40": ("K150 -> K40", "seurat_k150", 40),
    "spot_to_k40": ("Spot -> K40", "spot", 40),
}
REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class GeneralizationConfig:
    method: str
    data_root: Path
    developmental_root: Path
    output_root: Path
    organs: tuple[str, ...] = ("lung", "brain")
    pairs: tuple[str, ...] = ("11.5:12.5", "12.5:13.5", "11.5:13.5")
    hierarchies: tuple[str, ...] = tuple(HIERARCHIES)
    epochs: int = 1500
    nmf_max_iter: int = 300
    seed: int = 42
    device: str = "auto"
    dpi: int = 300
    runner_extra_args: tuple[str, ...] = ()

    @property
    def canonical_method(self) -> str:
        return METHOD_ALIASES[self.method]

    @property
    def method_tag(self) -> str:
        return "two_stage" if self.canonical_method == "maturity_cci_grn_two_stage" else "shr_method_10"

    @property
    def method_root(self) -> Path:
        return self.output_root.resolve() / self.method_tag

    def time_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs = []
        for raw in self.pairs:
            parts = raw.replace("->", ":").split(":")
            if len(parts) != 2 or not all(part.replace(".", "", 1).isdigit() for part in parts):
                raise ValueError(f"Invalid time pair: {raw!r}; expected e.g. 11.5:12.5")
            source, target = parts
            if not (math.isfinite(float(source)) and float(source) < float(target)):
                raise ValueError(f"Time pairs must run forward: {raw!r}")
            pairs.append((source, target))
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("Time pairs must be nonempty and unique.")
        return tuple(pairs)

    def validate(self) -> None:
        if self.method not in METHOD_ALIASES:
            raise ValueError(f"Unsupported method: {self.method}")
        for values, allowed in ((self.organs, {"heart", "lung", "brain"}),
                                (self.hierarchies, set(HIERARCHIES))):
            if not values or len(set(values)) != len(values) or not set(values) <= allowed:
                raise ValueError(f"Invalid or duplicate experiment selection: {values}")
        self.time_pairs()
        if self.epochs < 2 or self.nmf_max_iter < 1 or self.dpi < 1 or self.seed < 0:
            raise ValueError("epochs >= 2, positive NMF iterations/DPI, and nonnegative seed are required.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Invalid device")
        protected = {
            "--method", "--k", "--out-dir", "--seed", "--epochs", "--device", "--nmf-max-iter",
            "--nmf-components", "--grn-feature-method", "--cci-feature-method", "--alpha-cci", "--lambda-dev",
            "--maturity-id-column", "--maturity-column", "--maturity-normalization", "--maturity-direction",
        }
        allowed_overrides = {
            "--hidden-dim", "--mid-dim", "--gnn-layers", "--macro-layers", "--knn-k", "--local-dims",
            "--local-graph-mode", "--temperature", "--align-temperature", "--lr", "--log-every",
            "--lambda-align", "--lambda-ei", "--lambda-var", "--lambda-local", "--lambda-sharp",
            "--lambda-proto", "--lambda-closure", "--lambda-within", "--lambda-inter", "--lambda-retain",
            "--stage1-ratio", "--ei-retain-ratio", "--lambda-ei-constraint", "--lambda-retain-norm",
            "--retain-floor-ratio", "--lambda-retain-floor", "--inter-margin", "--active-usage-threshold",
            "--keff-min", "--checkpoint-keff-min", "--lambda-keff", "--lambda-dead-usage", "--min-usage",
            "--low-signal-pair-weight", "--low-signal-threshold-bits", "--checkpoint-retain-ratio",
            "--attention-heads", "--attention-ffn-dim", "--attention-dropout", "--attention-beta-init",
            "--cross-time-beta-init", "--cross-time-gate-bias-init", "--cosine-eta-min",
        }
        for value in self.runner_extra_args:
            name = value.split("=", 1)[0]
            if name in protected or name.startswith(("--h5ad", "--cci-", "--grn-t", "--maturity-")):
                raise ValueError(f"runner-extra-args cannot override experiment identity/features: {name}")
            if name.startswith("--") and name not in allowed_overrides:
                raise ValueError(f"Unknown or abbreviated trainer override: {name}")

    def contract(self) -> dict:
        return {
            "protocol": PROTOCOL, "method": self.canonical_method,
            "data_root": str(self.data_root.resolve()),
            "developmental_root": str(self.developmental_root.resolve()),
            "organs": list(self.organs), "pairs": [list(p) for p in self.time_pairs()],
            "hierarchies": list(self.hierarchies), "epochs": self.epochs,
            "nmf_max_iter": self.nmf_max_iter, "seed": self.seed, "device": self.device,
            "runner_extra_args": list(self.runner_extra_args),
            "features": {"grn": "legacy_nmf", "cci": "laplacian_hks", "alpha_cci": .20},
            "checkpoint_policy": "native; Method 10 checkpoint_keff_min=1e-6; training floor=0.30*K",
            "experiment": "independent within-organ retraining, independent hierarchy columns",
            "formal_profile": self.epochs == 1500 and self.nmf_max_iter == 300 and not self.runner_extra_args,
        }


def planned_jobs(cfg: GeneralizationConfig) -> list[dict]:
    jobs = []
    for organ in cfg.organs:
        for source, target in cfg.time_pairs():
            for hierarchy in cfg.hierarchies:
                label, scale, k = HIERARCHIES[hierarchy]
                job_id = f"{organ}/{hierarchy}/{source}_to_{target}"
                inputs = {}
                for side, stage in (("t", source), ("tp", target)):
                    paths = resolve_organ_paths(cfg.data_root, scale, organ, stage)
                    for key in ("h5ad", "cci", "cci_index", "grn"):
                        inputs[f"{key}_{side}"] = str(paths[key].resolve())
                    inputs[f"maturity_{side}"] = str(
                        (cfg.developmental_root / scale / f"{organ}_{stage}_features.csv").resolve())
                jobs.append({"job_id": job_id, "organ": organ, "hierarchy": hierarchy,
                             "column": label, "input_scale": scale, "K": k,
                             "time_pair": f"{source}->{target}", "method": cfg.canonical_method,
                             "inputs": inputs, "status": "pending", "error": "",
                             "trained_delta_EI": None, "attempts": []})
    return jobs


def training_command(cfg: GeneralizationConfig, job: dict, run_dir: Path) -> list[str]:
    command = [sys.executable, "-B", "-u", str(REPO_ROOT / "scripts/run_wyt_deltaei_coarse_grain.py"),
               "--method", cfg.canonical_method]
    options = {"h5ad": "h5ad", "cci": "cci", "cci_index": "cci-index", "grn": "grn", "maturity": "maturity"}
    for key, flag in options.items():
        for side in ("t", "tp"):
            command.extend([f"--{flag}-{side}", job["inputs"][f"{key}_{side}"]])
    command.extend([
        "--k", str(job["K"]), "--out-dir", str(run_dir), "--seed", str(cfg.seed),
        "--epochs", str(cfg.epochs), "--device", cfg.device,
        "--nmf-components", "5", "--nmf-max-iter", str(cfg.nmf_max_iter),
        "--grn-feature-method", "legacy_nmf", "--cci-feature-method", "laplacian_hks",
        "--alpha-cci", "0.20", "--lambda-dev", "0.05",
        "--maturity-id-column", "unit_id", "--maturity-column", "pseudotime",
    ])
    if cfg.canonical_method == "shr_method_10":
        command.extend(["--checkpoint-keff-min", "1e-6", "--keff-min", f"{.30 * job['K']:g}"])
    return [*command, *cfg.runner_extra_args]


def scientific_fingerprint() -> dict:
    # Include all training/frontend/shared numerical sources, not historical experiment folders.
    paths = {REPO_ROOT / "scripts/run_wyt_deltaei_coarse_grain.py", Path(__file__),
             Path(__file__).with_name("tables.py")}
    for folder in ("wyt_deltaei_coarse_grain", "mignet_ce/coarse_frontends", "mignet_ce/grn_representation",
                   "mignet_ce/pij/compare/_shared"):
        paths.update((REPO_ROOT / folder).glob("*.py"))
    for name in ("mignet_ce/metrics.py", "mignet_ce/networks/light_cci_grn.py", "mignet_ce/io/h5ad_h5py.py",
                 "mignet_ce/io/organ_paths.py", "mignet_ce/networks/wyt_cci_regsim.py",
                 "mignet_ce/downstream/analysis/dynamic_closure/analysis.py"):
        paths.add(REPO_ROOT / name)
    return {str(p.relative_to(REPO_ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def input_signature(job: dict) -> list[dict]:
    result = []
    for name, raw in sorted(job["inputs"].items()):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        stat = path.stat()
        result.append({"name": name, "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def execute_job(command: Sequence[str], log_path: Path) -> int:
    """Stream to disk so an OOMing child cannot fill the parent's memory with logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(list(command), cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def classify_failure(returncode: int, log_path: Path) -> tuple[str, str]:
    tail = ""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 65536))
            tail = handle.read().decode("utf-8", errors="replace")
    lower = tail.lower()
    if any(marker in lower for marker in ("out of memory", "outofmemoryerror", "memoryerror",
                                          "unable to allocate", "can't allocate memory", "cannot allocate memory")):
        status = "oom"
    elif returncode in {-9, 137, -1073741819}:
        status = "killed"  # SIGKILL/exit 137 alone does not prove OOM.
    elif ("no eligible" in lower or "no valid checkpoint" in lower or "no stage 1 epoch" in lower
          or "no valid non-collapsed checkpoint" in lower):
        status = "no_checkpoint"
    else:
        status = "failed"
    return status, f"returncode={returncode}\n{tail[-4000:]}".strip()


def _next_attempt(root: Path, job: dict) -> Path:
    base = root / "runs" / job["job_id"]
    index = 1
    while (base / f"attempt_{index:04d}").exists():
        index += 1
    return base / f"attempt_{index:04d}"


def _overall_status(jobs: list[dict]) -> str:
    return "complete" if all(job["status"] == "success" for job in jobs) else "partial"


def _save(cfg: GeneralizationConfig, state: dict) -> None:
    from .tables import export_tables
    state["updated_at"] = _utc()
    write_json(cfg.method_root / "manifest.json", state)
    export_tables(cfg.method_root, state, dpi=cfg.dpi)


def run(cfg: GeneralizationConfig, *, mode: str = "run", dry_run: bool = False) -> dict:
    cfg.validate()
    if mode not in {"run", "resume", "render"}:
        raise ValueError(f"Unknown mode: {mode}")
    jobs = planned_jobs(cfg)
    root, manifest = cfg.method_root, cfg.method_root / "manifest.json"
    if dry_run:
        issues = []
        for job in jobs:
            try:
                input_signature(job)
            except FileNotFoundError as exc:
                issues.append(str(exc))
            print(shlex.join(training_command(cfg, job, root / "runs" / job["job_id"] / "attempt_0001")))
        for issue in dict.fromkeys(issues):
            print(f"[MISSING] {issue}")
        return {"status": "partial" if issues else "ready", "issues": issues}
    from .tables import inspect_training_run
    if mode in {"resume", "render"}:
        if not manifest.is_file():
            raise FileNotFoundError(f"No saved experiment to {mode}: {manifest}")
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state.get("protocol") != PROTOCOL or state.get("method") != cfg.canonical_method:
            raise ValueError("Saved manifest protocol/method does not match this command.")
        if mode == "resume" and state["contract"] != cfg.contract():
            raise ValueError("Resume contract mismatch; use identical settings or a new output root.")
        jobs = state["jobs"]
        if mode == "render":
            for job in jobs:
                if job["status"] == "running" and not job.get("run_dir"):
                    job.update(status="interrupted", error="Running job has no saved attempt.", trained_delta_EI=None)
                if job["status"] in {"success", "running"} and job.get("run_dir"):
                    try:
                        job.update(inspect_training_run(job, state["contract"], root))
                        job.update(status="success", error="")
                    except Exception as exc:
                        job.update(status="invalid_output", error=str(exc), trained_delta_EI=None)
            state["status"] = _overall_status(jobs)
            _save(cfg, state)
            return {"status": state["status"], "table_root": str(root / "tables")}
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Output exists; use --mode resume or a new --output-root: {root}")
        state = {"protocol": PROTOCOL, "method": cfg.canonical_method, "contract": cfg.contract(),
                 "created_at": _utc(), "status": "running", "jobs": jobs}
    # Initial reports are persisted before touching any potentially large input data.
    state["status"] = "running"
    _save(cfg, state)
    code = scientific_fingerprint()
    try:
        for job in jobs:
            try:
                signature = input_signature(job)
            except FileNotFoundError as exc:
                job.update(status="missing_input", error=str(exc), trained_delta_EI=None)
                _save(cfg, state)
                continue
            fingerprint = _digest({"contract": state["contract"], "inputs": signature, "code": code})
            if mode == "resume" and job.get("fingerprint") == fingerprint and job.get("run_dir"):
                try:
                    job.update(inspect_training_run(job, state["contract"], root))
                    job.update(status="success", error="")
                    _save(cfg, state)
                    print(f"[generalization] reuse {job['job_id']}", flush=True)
                    continue
                except Exception:
                    pass  # A new numbered attempt preserves the old partial/invalid directory.
            run_dir = _next_attempt(root, job)
            log_path = root / "logs" / job["job_id"] / f"{run_dir.name}.log"
            command = training_command(cfg, job, run_dir)
            attempt = {"run_dir": str(run_dir), "command": command, "log_path": str(log_path),
                       "fingerprint": fingerprint, "inputs": signature, "started_at": _utc()}
            job["attempts"].append(attempt)
            job.update(status="running", error="", trained_delta_EI=None, run_dir=str(run_dir),
                       fingerprint=fingerprint, log_path=str(log_path))
            _save(cfg, state)
            print(f"[generalization] run {job['job_id']} -> {log_path}", flush=True)
            try:
                returncode = execute_job(command, log_path)
                attempt["returncode"] = returncode
                if returncode:
                    status, error = classify_failure(returncode, log_path)
                    job.update(status=status, error=error)
                else:
                    job.update(inspect_training_run(job, state["contract"], root))
                    job.update(status="success", error="")
            except KeyboardInterrupt:
                job.update(status="interrupted", error="Interrupted by user; previous attempts retained.")
                raise
            except Exception as exc:
                job.update(status="invalid_output" if run_dir.exists() else "failed", error=str(exc),
                           trained_delta_EI=None)
            finally:
                attempt.update(finished_at=_utc(), status=job["status"], error=job["error"])
                _save(cfg, state)
            print(f"[generalization] {job['job_id']}: {job['status']}", flush=True)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    else:
        state["status"] = _overall_status(jobs)
    finally:
        _save(cfg, state)
    return {"status": state["status"], "table_root": str(root / "tables")}
