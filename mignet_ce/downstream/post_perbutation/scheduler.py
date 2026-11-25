from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict, dataclass
import getpass
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

from .config import FAMILIES, PROTOCOL, Settings, Task, file_digest, read_json, write_json


MODULE = "mignet_ce.downstream.post_perbutation"
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
GIB = 1 << 30


class LockBusy(RuntimeError):
    pass


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise LockBusy(f"Another scheduler is using {path}") from error
        yield


def run_lock(root: Path):
    return _file_lock(root / ".run.lock")


def _smi(query: str, fields: str) -> list[list[str]]:
    result = subprocess.run(["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False, timeout=8)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi query failed: {result.stderr.strip() or result.stdout.strip()}")
    return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines())]


@dataclass(frozen=True)
class CardMemory:
    used: int
    total: int
    per_pid: dict[int, int]


def _nvidia_memory():
    cards = _smi("gpu", "index,uuid,memory.used,memory.total")
    processes = _smi("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    by_card = {}
    for uuid, pid, used in (row for row in processes if len(row) == 3):
        if pid.isdigit() and used.isdigit():
            by_card.setdefault(uuid, {})[int(pid)] = int(used) << 20
    return {uuid: (index, CardMemory(int(used) << 20, int(total) << 20, by_card.get(uuid, {})))
            for index, uuid, used, total in cards}


class GpuAdmission:
    """One card-wide admission lock and OS-backed leases for all launchers of this user."""

    def __init__(self, devices: dict, budget_gib: float, initial: int, maximum: int):
        self.target = int(budget_gib * GIB)
        self.margin = GIB
        self.initial, self.maximum = initial, maximum
        self.limit = {device: initial for device in devices}
        self.estimated = {}
        self.poll_time = 0.
        self.cards = {}
        self.device_peaks = {}
        if set(devices) == {"cpu"}:
            return
        self.refresh(force=True)
        user = sha256(getpass.getuser().encode()).hexdigest()[:12]
        self.root = Path(tempfile.gettempdir()) / f"post-perbutation-gpu-{user}"
        for device, info in devices.items():
            token = info["visible_device"]
            raw_uuid = info["uuid"]
            normalized = raw_uuid.lower().removeprefix("gpu-")
            matches = [uuid for uuid in self.snapshot if uuid.lower().removeprefix("gpu-") == normalized]
            if not matches:
                matches = [uuid for uuid, (index, _) in self.snapshot.items() if index == token or uuid == token]
            if len(matches) != 1 or matches[0] in self.cards.values():
                raise RuntimeError(f"Cannot map {device} ({token}) to a unique GPU in nvidia-smi")
            self.cards[device] = matches[0]
            memory = self.snapshot[matches[0]][1]
            if self.target > memory.total or self.target <= self.margin:
                raise ValueError(f"Invalid per-card budget {budget_gib:g} GiB for {device}")

    def refresh(self, *, force: bool = False) -> None:
        if force or (self.cards and time.monotonic() - self.poll_time >= 1.):
            self.snapshot = _nvidia_memory()
            self.poll_time = time.monotonic()
            for device in self.cards:
                self.device_peaks[device] = max(self.device_peaks.get(device, 0), self._memory(device).used)

    def _memory(self, device: str) -> CardMemory:
        return self.snapshot[self.cards[device]][1]

    def estimate(self, device: str, family: str) -> int:
        profile = "existence" if family == "existence" else "training"
        initial_peak = min(4 * GIB, (self.target - 2 * self.margin) // self.initial)
        return self.estimated.get((device, profile), max(initial_peak, GIB // 2))

    def _active(self, folder: Path) -> list[dict]:
        active = []
        for path in folder.glob("slot-*.lock"):
            try:
                with _file_lock(path):
                    pass
            except LockBusy:
                active.append(read_json(path.with_suffix(".json")))
        return active

    def admit(self, slot, work) -> bool:
        device = slot.device
        if slot.index >= self.limit[device] and not work.retry:
            return False
        if device not in self.cards:
            return True
        folder = self.root / self.cards[device]
        try:
            with _file_lock(folder / "admit.lock"):
                active = self._active(folder)
                cap = min([self.maximum, *(item.get("maximum", self.maximum) for item in active)])
                target = min([self.target, *(item.get("target", self.target) for item in active)])
                if (work.retry and active) or any(item["solo"] for item in active) or len(active) >= cap:
                    return False
                memory = self._memory(device)
                predicted = memory.used + sum(max(0, item["peak"] - memory.per_pid.get(item["pid"], 0)) for item in active)
                pid = slot.process.pid if slot.process and slot.process.poll() is None else None
                peak = self.estimate(device, work.task.family)
                if predicted + max(0, peak - memory.per_pid.get(pid, 0)) > target - self.margin:
                    return False
                lease = ExitStack()
                try:
                    for index in range(self.maximum):
                        path = folder / f"slot-{index}.lock"
                        try:
                            lease.enter_context(_file_lock(path))
                        except LockBusy:
                            continue
                        state = path.with_suffix(".json")
                        write_json(state, {"pid": pid, "peak": peak, "solo": work.retry,
                                           "target": self.target, "maximum": self.maximum})
                        slot.lease = lease
                        slot.lease_record = state
                        slot.peak = memory.per_pid.get(pid, 0)
                        return True
                except BaseException:
                    lease.close()
                    raise
        except LockBusy:
            return False
        return False

    def started(self, slot) -> None:
        if slot.lease_record is not None:
            state = read_json(slot.lease_record)
            write_json(slot.lease_record, {**state, "pid": slot.process.pid})

    def sample(self, slot) -> None:
        if slot.device in self.cards and slot.process is not None:
            slot.peak = max(slot.peak, self._memory(slot.device).per_pid.get(slot.process.pid, 0))

    def finished(self, slot, family: str, result: dict) -> None:
        if slot.device not in self.cards:
            return
        metrics = result.setdefault("metrics", {})
        metrics["peak_device_process_bytes"] = slot.peak
        if result["status"] == "oom":
            self.limit[slot.device] = max(1, self.limit[slot.device] - 1)
        elif result["status"] == "complete" and metrics.get("peak_cuda_reserved_bytes") is not None:
            peak = max(slot.peak, metrics["peak_cuda_reserved_bytes"] + GIB // 2)
            profile = "existence" if family == "existence" else "training"
            key = (slot.device, profile)
            self.estimated[key] = max(self.estimated.get(key, 0), peak)
            supported = max(self.initial, min(self.maximum, (self.target - self.margin) // self.estimated[key]))
            self.limit[slot.device] = min(supported, self.limit[slot.device] + 1)

    def release(self, slot) -> None:
        if slot.lease is not None:
            slot.lease.close()
            slot.lease = None
            slot.lease_record = None
        slot.peak = 0


def device_tokens(devices: list[str]) -> dict[str, str]:
    if devices == ["cpu"]:
        return {"cpu": ""}
    if not devices or len(set(devices)) != len(devices) or any(not re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", item) for item in devices):
        raise ValueError("Specify distinct cuda:N devices, or cpu")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return {item: item[5:] for item in devices}
    visible = [token.strip() for token in inherited.split(",") if token.strip() and token.strip() != "-1"]
    if any(int(item[5:]) >= len(visible) for item in devices):
        raise ValueError(f"Requested GPUs exceed CUDA_VISIBLE_DEVICES={inherited!r}")
    tokens = {item: visible[int(item[5:])] for item in devices}
    if len(set(tokens.values())) != len(tokens):
        raise ValueError(f"CUDA_VISIBLE_DEVICES maps multiple requested devices to one GPU: {inherited!r}")
    return tokens


def probe_devices(tokens: dict[str, str]) -> dict:
    detected = {}
    for device, token in tokens.items():
        if device == "cpu":
            detected[device] = {"visible_device": "cpu"}
            continue
        process = subprocess.run(
            [sys.executable, "-u", "-m", MODULE, "_probe"],
            env={**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": token},
            capture_output=True, text=True, check=False,
        )
        if process.returncode:
            raise RuntimeError(f"{device} ({token}) is unavailable in this Python environment: {process.stderr.strip()}")
        detected[device] = {"visible_device": token, **json.loads(process.stdout)}
    return detected


def gpu_probe() -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected a single visible CUDA device, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({"name": properties.name, "total_bytes": properties.total_memory,
                      "uuid": str(getattr(properties, "uuid", ""))}))


def worker_loop(device: str) -> None:
    from .formal import run_training

    cache = {}
    for line in sys.stdin:
        job = json.loads(line)
        directory = Path(job["directory"])
        result = {"id": job["id"], "signature": job["signature"], "baseline": job.get("baseline"),
                  "pid": os.getpid()}
        restart = False
        with (directory / "worker.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                print(f"condition={job['id']} gpu={job['device']} visible={os.getenv('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}")
                try:
                    started = time.perf_counter()
                    if job["family"] == "existence":
                        from .symmetry_existence_pipeline import calculate_batch
                        output = calculate_batch(Path(job["baseline"]), tuple(map(Path, job["seurat"])),
                                                 Settings(**job["settings"]), job["jobs"], device=device, cache=cache)
                    else:
                        output = run_training(job, device, cache)
                    result.update(output, status="complete", seconds=time.perf_counter() - started)
                except Exception as error:
                    message = str(error)
                    lowered = message.lower().replace("_", " ")
                    oom = device.startswith("cuda") and (type(error).__name__ == "OutOfMemoryError"
                                                          or ("cuda" in lowered and "out of memory" in lowered))
                    result.update(status="oom" if oom else "failed", error=message)
                    traceback.print_exc()
                    if oom:
                        import torch
                        try:
                            free, total = torch.cuda.mem_get_info(device)
                            result["metrics"] = {"cuda_free_bytes_on_error": free, "cuda_total_bytes": total,
                                                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                        except RuntimeError:
                            traceback.print_exc()
                if device.startswith("cuda") and result["status"] != "oom":
                    import torch
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        traceback.print_exc()
                        restart = True
        if restart:
            result["restart_worker"] = True
        write_json(directory / "result.json", result)
        if restart or result["status"] == "oom":
            return


@dataclass
class Work:
    task: Task
    directory: Path
    payload: dict
    retry: bool = False


@dataclass
class Slot:
    name: str
    device: str
    token: str
    log: Path
    index: int = 0
    process: subprocess.Popen | None = None
    work: Work | None = None
    lease: ExitStack | None = None
    lease_record: Path | None = None
    peak: int = 0

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self.stop()
        env = {**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": self.token, "PYTHONUNBUFFERED": "1"}
        with self.log.open("a", encoding="utf-8") as handle:
            local_device = "cpu" if self.device == "cpu" else "cuda:0"
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", MODULE, "_worker", local_device],
                env=env, stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8",
            )

    def send(self, payload: dict) -> None:
        self.start()
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def stop(self, terminate: bool = False) -> None:
        if self.process is None:
            return
        if terminate and self.process.poll() is None:
            self.process.terminate()
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None


def _matches_result(result: dict, record: dict, signature: str, baseline: Path | None) -> bool:
    return (result.get("id") == record["id"] and result.get("signature") == signature
            and result.get("baseline") == (str(baseline) if record["family"] in FAMILIES and baseline else None)
            and (record["family"] not in FAMILIES or baseline is not None))


def _restore_result(record: dict, signature: str, baseline: Path | None) -> dict | None:
    directory = record.get("directory")
    if not directory:
        return None
    path = Path(directory)
    try:
        result = read_json(path / "result.json")
        if result.get("status") != "complete" or not _matches_result(result, record, signature, baseline):
            return None
        if not result.get("artifacts") or any(file_digest(path / name) != digest for name, digest in result["artifacts"].items()):
            return None
        return {**record, **result, "metrics": {**result.get("metrics", {}), **record.get("metrics", {})}}
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Scheduler:
    def __init__(self, root: Path, settings: Settings, inputs, tasks: tuple[Task, ...],
                 seurat, tokens: dict[str, str], workers_per_gpu: int, cpu_workers: int, resume: bool,
                 *, shared_root: Path | None = None, max_workers_per_gpu: int | None = None, budget_gib: float = 18):
        self.root, self.settings, self.tasks = root, settings, tasks
        self.shared_root = root if shared_root is None else shared_root
        self.formal_inputs = inputs
        self.inputs = {key: str(value) for key, value in vars(inputs).items()}
        self.seurat = seurat
        self.cpu_workers = cpu_workers
        self.futures = {}
        self.ready = {name: deque() for name in ("p0", "cci", "symmetry", "existence")}
        self.cci_pending = deque()
        self.oom_wait = deque()
        self.last_family = "symmetry"
        self.errors = {}
        self.baseline = None
        self.baseline_dispatched = False
        self.baseline_lock = None
        self.baseline_waited = False
        self.baseline_finished = False
        self.existence_rows = None
        self.report_existence = False
        self.existence_parts = {}
        self.existence_failed = 0
        self.existence_total = 2 * (settings.random_partitions + 1) if seurat is not None else 0
        self.last_progress = None

        identity = {"protocol": PROTOCOL, "settings": asdict(settings), "inputs": self.inputs,
                    "input_sha256": {key: file_digest(Path(path)) for key, path in self.inputs.items()}}
        self.signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        baseline_identity = {**identity, "settings": {name: getattr(settings, name) for name in
                                                      ("organ", "source_time", "target_time", "k", "epochs", "train_seed")}}
        self.baseline_key = sha256(json.dumps(baseline_identity, sort_keys=True).encode()).hexdigest()
        self.baseline_registry = self.shared_root / "baseline" / self.baseline_key[:12]
        self.manifest = root / "run_manifest.json"
        self.state = root / "conditions.json"
        if self.manifest.exists():
            if not resume:
                raise ValueError(f"Run already exists: {root}. Use --resume or a new output root.")
            if read_json(self.manifest)["signature"] != self.signature:
                raise ValueError("Run inputs or settings changed. Use a new output root.")
        devices = probe_devices(tokens)
        maximum = max_workers_per_gpu or workers_per_gpu
        if set(tokens) == {"cpu"}:
            maximum = workers_per_gpu
        self.admission = GpuAdmission(devices, budget_gib, workers_per_gpu, maximum)
        write_json(self.manifest, {**identity, "signature": self.signature, "devices": devices,
                                   "workers_per_gpu": workers_per_gpu, "max_workers_per_gpu": maximum,
                                   "gpu_budget_gib": budget_gib, "cpu_workers": cpu_workers})
        logs = root / "workers"
        logs.mkdir(exist_ok=True)
        self.slots = [Slot(f"{device.replace(':', '-')}-{index}", device, token,
                           logs / f"{device.replace(':', '-')}-{index}.log", index=index)
                      for device, token in tokens.items() for index in range(maximum)]
        records = read_json(self.state) if self.state.exists() else {}
        legacy, prior = {}, {}
        if root != self.shared_root and (self.shared_root / "run_manifest.json").is_file():
            legacy = read_json(self.shared_root / "run_manifest.json")
            if (self.shared_root / "conditions.json").is_file():
                prior = read_json(self.shared_root / "conditions.json")
            if not records and resume and legacy.get("signature") == self.signature:
                selected = {task.id for task in tasks}
                records = {key: value for key, value in prior.items() if key in selected}
        shared = self.baseline_registry / "state.json"
        cached = read_json(shared) if shared.is_file() else {}
        p0 = self.restore_baseline(cached) or _restore_result(records.get("P0", {}), self.signature, None)
        if p0 is None and prior and all(legacy.get(key) == baseline_identity[key]
                                       for key in ("protocol", "inputs", "input_sha256")):
            settings_match = all(legacy.get("settings", {}).get(key) == value
                                 for key, value in baseline_identity["settings"].items())
            if settings_match:
                p0 = _restore_result(prior.get("P0", {}), legacy["signature"], None)
        if p0:
            self.baseline = Path(p0["directory"])
            self.baseline_finished = True
        self.records = {}
        active = {task.id for task in tasks}
        for key, record in records.items():
            restored = p0 if key == "P0" else _restore_result(record, self.signature, self.baseline)
            if restored is not None:
                self.records[key] = restored
            elif key not in active:
                self.records[key] = {**record, "status": "incomplete", "metrics": {}}
        for task in tasks:
            if task.id not in self.records:
                self.records[task.id] = {**asdict(task), "id": task.id, "status": "pending"}
            if task.family == "p0" and p0 is not None:
                self.records[task.id] = p0
            if self.records[task.id]["status"] == "complete":
                continue
            if task.family == "cci":
                self.cci_pending.append(task)
        self.write_state()

    def restore_baseline(self, state: dict) -> dict | None:
        if state and state.get("baseline_key") != self.baseline_key:
            raise ValueError(f"P0 identity collision at {self.baseline_registry}")
        record = state.get("record", {})
        signature = state.get("signature")
        return _restore_result(record, signature, None) if signature else None

    def poll_baseline(self) -> None:
        from .formal import prepare_on_cpu

        if self.baseline_lock is not None:
            return
        path = self.baseline_registry / "state.json"
        state = read_json(path) if path.is_file() else {}
        if self.baseline_finished and (self.baseline is None or state.get("status") == "complete"):
            return
        lock = ExitStack()
        try:
            lock.enter_context(run_lock(self.baseline_registry))
        except LockBusy:
            self.baseline_waited = True
            return
        self.baseline_lock = lock
        state = read_json(path) if path.is_file() else {}
        shared = self.restore_baseline(state)
        if shared is not None:
            self.records["P0"] = shared
            self.baseline = Path(shared["directory"])
            self.finish_baseline("complete")
        elif self.baseline is not None:
            self.finish_baseline("complete")
        elif state.get("status") == "failed" and self.baseline_waited:
            self.update(Task("p0"), "failed", error=state["record"].get("error", "Shared P0 failed"))
            self.finish_baseline("failed")
        else:
            task = Task("p0")
            work = self.make_work(task, {})
            self.update(task, "staging", directory=str(work.directory))
            self.write_baseline("staging")
            self.submit("stage_p0", prepare_on_cpu, self.formal_inputs, self.settings, work.directory, context=work)

    def write_baseline(self, status: str) -> None:
        record = self.records["P0"]
        write_json(self.baseline_registry / "state.json", {
            "status": status, "baseline_key": self.baseline_key,
            "signature": record.get("signature", self.signature), "record": record,
        })

    def finish_baseline(self, status: str) -> None:
        self.baseline_finished = True
        self.write_baseline(status)
        self.baseline_lock.close()
        self.baseline_lock = None
        self.write_state()

    def write_state(self) -> None:
        write_json(self.state, self.records)

    def update(self, task: Task, status: str, **fields) -> None:
        self.records[task.id] = {**self.records[task.id], **fields, "status": status}
        self.write_state()
        if status in {"failed", "blocked", "oom_wait"}:
            print(f"[{status}] {task.id} {fields.get('device', '')} {fields.get('error', '')}", flush=True)

    def display_progress(self) -> None:
        entries = {}
        for name, group in (
            ("P0", [task for task in self.tasks if task.family == "p0"]),
            ("CCI", [task for task in self.tasks if task.family == "cci"]),
            ("Symmetry-perturbation", [task for task in self.tasks if task.family in FAMILIES]),
        ):
            if group:
                statuses = [self.records[task.id]["status"] for task in group]
                entries[name] = {"complete": statuses.count("complete"), "total": len(group),
                                 "failed": statuses.count("failed") + statuses.count("blocked"),
                                 "infeasible": statuses.count("infeasible")}
        if self.existence_total:
            complete = self.existence_total if self.report_existence else sum(len(part["rows"]) for part in self.existence_parts.values())
            entries["Symmetry-existence"] = {"complete": complete, "total": self.existence_total,
                                             "failed": self.existence_failed, "infeasible": 0}
        if entries == self.last_progress:
            return
        self.last_progress = entries
        write_json(self.root / "progress.json", entries)
        labels = []
        for name, entry in entries.items():
            done = entry["complete"] + entry["failed"] + entry["infeasible"]
            total = entry["total"]
            marks = 10 * done // total
            suffix = "".join(f" {key}={entry[key]}" for key in ("failed", "infeasible") if entry[key])
            labels.append(f"{name} [{'#' * marks}{'.' * (10 - marks)}] {done}/{total}{suffix}")
        print(" | ".join(labels), flush=True)

    def make_work(self, task: Task, payload: dict, retry: bool = False) -> Work:
        folder = self.baseline_registry if task.family == "p0" else self.root / "runs" / task.id
        folder.mkdir(parents=True, exist_ok=True)
        number = 1 + max((int(path.name[8:]) for path in folder.glob("attempt-*") if path.name[8:].isdigit()), default=0)
        directory = folder / f"attempt-{number:04d}"
        directory.mkdir()
        return Work(task, directory, payload, retry)

    def submit(self, kind: str, function, *args, context=None) -> None:
        future = self.pool.submit(function, *args)
        self.futures[future] = (kind, context)

    def start_baseline_dependents(self) -> None:
        from .formal import build_reference

        if self.baseline_dispatched or self.baseline is None:
            return
        self.baseline_dispatched = True
        self.symmetry_root = self.root / "plans" / self.baseline.name
        self.reference = self.symmetry_root / "reference.npz"
        if self.seurat is not None:
            self.queue_existence()
        if any(task.family in FAMILIES and self.records[task.id]["status"] == "pending" for task in self.tasks):
            self.submit("reference", build_reference, self.baseline, self.settings, self.reference)

    def queue_existence(self) -> None:
        from .symmetry_existence_pipeline import batches

        self.existence_identity = {"baseline": str(self.baseline), "seurat_sha256": [file_digest(path) for path in self.seurat]}
        manifest = self.root / "symmetry_existence_manifest.json"
        report = self.root / "symmetry_existence.json"
        if manifest.is_file() and report.is_file() and read_json(manifest) == self.existence_identity:
            self.report_existence = len(read_json(report)) == self.existence_total
        if self.report_existence:
            return
        digest = sha256(json.dumps(self.existence_identity, sort_keys=True).encode()).hexdigest()[:12]
        self.parts_root = self.root / "plans" / "existence" / digest
        self.existence_batches = batches(self.settings)
        for index, tasks in enumerate(self.existence_batches):
            task_id = Task("existence", level=index)
            part = self.parts_root / f"{task_id.id}.json"
            if part.is_file():
                result = read_json(part)
                if self.existence_matches(tasks, result):
                    self.existence_parts[task_id.id] = result
                    continue
            self.ready["existence"].append(self.make_work(task_id, {
                "baseline": str(self.baseline), "seurat": list(map(str, self.seurat)), "jobs": tasks,
            }))
        self.finish_existence()

    @staticmethod
    def existence_matches(tasks, result: dict) -> bool:
        return [(row.get("matched_to"), row.get("repeat")) for row in result.get("rows", [])] == [
            tuple(task) for task in tasks
        ]

    def finish_existence(self) -> None:
        if len(self.existence_parts) != len(self.existence_batches):
            return
        parts = [self.existence_parts[key] for key in sorted(self.existence_parts)]
        write_json(self.root / "symmetry_existence.json", [row for part in parts for row in part["rows"]])
        write_json(self.root / "symmetry_existence_manifest.json", self.existence_identity)
        write_json(self.root / "symmetry_existence_execution.json", [
            {"device": part["device"], "groups": len(part["rows"]), "seconds": part["seconds"]} for part in parts
        ])
        self.report_existence = True

    def collect_existence(self, work: Work, result: dict, device: str) -> None:
        if not self.existence_matches(work.payload["jobs"], result):
            self.existence_error(work, "Worker returned an unexpected partition batch", device)
            return
        part = {"device": device, "rows": result["rows"], "seconds": result["seconds"]}
        write_json(self.parts_root / f"{work.task.id}.json", part)
        self.existence_parts[work.task.id] = part
        self.finish_existence()

    def existence_error(self, work: Work, error: str, device: str) -> None:
        if device != "cpu":
            work.payload["cpu_only"] = True
            self.ready["existence"].append(work)
        else:
            self.errors[work.task.id] = error
            self.existence_failed += len(work.payload["jobs"])

    def stage_existence(self) -> None:
        from .symmetry_existence_pipeline import calculate_batch

        queue = self.ready["existence"]
        if not queue:
            return
        for _ in range(max(0, self.cpu_workers - len(self.futures))):
            if not queue:
                break
            work = queue.pop()
            self.submit("existence", calculate_batch, self.baseline, self.seurat, self.settings,
                        work.payload["jobs"], context=work)

    def pending_symmetry(self, family: str | None = None):
        return [task for task in self.tasks if task.family in FAMILIES and (family is None or task.family == family)
                and self.records[task.id]["status"] == "pending"]

    def poll_cpu(self) -> None:
        from .symmetry_perturbation_pipeline import Calibration

        for future, (kind, context) in list(self.futures.items()):
            if not future.done():
                continue
            del self.futures[future]
            try:
                result = future.result()
            except Exception as error:
                detail = context.task.id if isinstance(context, Work) else str(context or "")
                self.errors[f"{kind}:{detail}"] = str(error)
                print(f"[{kind} error] {error}", flush=True)
                if kind in {"stage_cci", "stage_p0"}:
                    (context.directory / "worker.log").write_text(traceback.format_exc(), encoding="utf-8")
                    self.update(context.task, "failed", error=str(error), directory=str(context.directory))
                    if kind == "stage_p0":
                        self.finish_baseline("failed")
                elif kind == "existence":
                    self.existence_error(context, str(error), "cpu")
                elif kind == "cci_plan":
                    while self.cci_pending:
                        self.update(self.cci_pending.popleft(), "blocked", error=str(error))
                elif kind in {"reference", "symmetry_plan"}:
                    for task in self.pending_symmetry(context if kind == "symmetry_plan" else None):
                        self.update(task, "blocked", error=str(error))
                elif kind == "endpoint":
                    self.calibration.collect(self.submit, self.pending_symmetry, *context, {"rho": 0.0, "error": str(error)})
                continue
            if kind == "cci_plan":
                self.cci_audit = result
            elif kind in {"stage_cci", "stage_p0"}:
                context.payload.update(result)
                self.ready[context.task.family].append(context)
            elif kind == "existence":
                self.collect_existence(context, result, "cpu")
            elif kind == "reference":
                write_json(self.symmetry_root / "baseline_metrics.json", result)
                self.calibration = Calibration(self.baseline, self.reference, self.settings, self.symmetry_root)
                self.calibration.submit(self.submit, self.pending_symmetry)
            elif kind == "endpoint":
                self.calibration.collect(self.submit, self.pending_symmetry, *context, result)
            elif kind == "symmetry_plan":
                for task in self.pending_symmetry(context):
                    plan = result["conditions"][task.id]
                    if "error" in plan:
                        self.update(task, "infeasible", error=plan["error"],
                                    metrics={name: value for name, value in plan.items() if name != "error"})
                    else:
                        self.ready["symmetry"].append(self.make_work(task, {**plan, "baseline": str(self.baseline),
                                                                           "reference": str(self.reference)}))

    def stage_cci(self) -> None:
        from .cci_pipeline import prepare_condition

        # Keep formal input preparation ahead without retaining an unbounded number of full CCI states.
        limit = max(self.cpu_workers, 2 * sum(self.admission.limit.values()))
        staged = sum(kind == "stage_cci" for kind, _ in self.futures.values()) + len(self.ready["cci"])
        capacity = min(self.cpu_workers - len(self.futures), limit - staged)
        for _ in range(capacity):
            if not self.cci_pending or self.cci_audit is None:
                return
            task = self.cci_pending.popleft()
            if not self.cci_audit.get("region_edge_counts", {}).get(task.topology):
                self.update(task, "infeasible", error="CCI region has no eligible edges")
                continue
            work = self.make_work(task, {})
            self.update(task, "staging", directory=str(work.directory))
            self.submit("stage_cci", prepare_condition, self.formal_inputs, self.cci_plan / "regions.npz",
                        task, self.settings, work.directory, context=work)

    def poll_workers(self) -> None:
        for slot in self.slots:
            work = slot.work
            if work is None:
                continue
            self.admission.sample(slot)
            result_path = work.directory / "result.json"
            if result_path.is_file():
                result = read_json(result_path)
            elif slot.process.poll() is not None:
                result = {"id": work.task.id, "signature": self.signature, "baseline": work.payload.get("baseline"),
                          "status": "failed", "error": f"Worker exited with code {slot.process.returncode}. See {slot.log}"}
            else:
                continue
            slot.work = None
            task = work.task
            self.admission.finished(slot, task.family, result)
            self.admission.release(slot)
            if result.pop("restart_worker", False) or result["status"] == "oom":
                slot.stop()
            if task.family == "existence":
                if result["status"] == "complete" and result.get("id") == task.id and result.get("signature") == self.signature:
                    self.collect_existence(work, result, slot.device)
                else:
                    self.existence_error(work, result.get("error", "Mismatched worker result"), slot.device)
                continue
            if not _matches_result(result, self.records[task.id], self.signature, self.baseline):
                self.update(task, "failed", error="Mismatched worker result")
                if task.family == "p0":
                    self.finish_baseline("failed")
                continue
            status = result.pop("status")
            if status == "complete":
                metrics = {**result.pop("metrics", {}), "worker_seconds": result.get("seconds")}
                self.update(task, "complete", metrics=metrics, **result)
                if task.family == "p0":
                    self.baseline = work.directory
                    self.finish_baseline("complete")
            elif status == "oom" and not work.retry:
                self.update(task, "oom_wait", **result)
                self.oom_wait.append(work)
            else:
                self.update(task, "failed", **result)
                if task.family == "p0":
                    self.finish_baseline("failed")

    def start_work(self, slot: Slot, work: Work) -> None:
        task = work.task
        slot.work = work
        if task.family != "existence":
            self.update(task, "running", directory=str(work.directory), device=slot.device, worker=slot.name,
                        solo_retry=work.retry, error="", metrics={})
        if task.family == "p0":
            self.write_baseline("running")
        job = {**work.payload, "id": task.id, "family": task.family, "device": slot.device,
               "signature": self.signature, "inputs": self.inputs, "settings": asdict(self.settings),
               "directory": str(work.directory)}
        try:
            slot.send(job)
            self.admission.started(slot)
        except OSError as error:
            slot.work = None
            slot.stop(terminate=True)
            self.admission.release(slot)
            if task.family == "existence":
                self.existence_error(work, str(error), slot.device)
            else:
                self.update(task, "failed", error=str(error))
                if task.family == "p0":
                    self.finish_baseline("failed")

    def can_start(self, slot: Slot, work: Work) -> bool:
        return self.admission.admit(slot, work)

    def dispatch(self) -> None:
        reserved = {slot.device for slot in self.slots if slot.work and slot.work.retry}
        if self.oom_wait and not reserved:
            device = min(dict.fromkeys(slot.device for slot in self.slots),
                         key=lambda value: sum(slot.work is not None for slot in self.slots if slot.device == value))
            reserved.add(device)
            members = [slot for slot in self.slots if slot.device == device]
            if not any(slot.work for slot in members):
                # Release allocator/callback caches of the idle siblings before the solo attempt.
                for slot in members:
                    slot.stop()
                retry = self.oom_wait[0]
                candidate = Work(retry.task, retry.directory, retry.payload, retry=True)
                if self.can_start(members[0], candidate):
                    self.oom_wait.popleft()
                    self.start_work(members[0], self.make_work(retry.task, retry.payload, retry=True))
        for slot in self.slots:
            if slot.device in reserved or slot.work is not None:
                continue
            if self.ready["p0"] and self.can_start(slot, self.ready["p0"][0]):
                self.start_work(slot, self.ready["p0"].popleft())
                continue
            first = "cci" if self.last_family == "symmetry" else "symmetry"
            second = self.last_family
            for family in (first, second, "existence"):
                queue = self.ready.get(family)
                if family == "existence" and slot.device == "cpu":
                    continue
                if queue and not queue[0].payload.get("cpu_only") and self.can_start(slot, queue[0]):
                    if family != "existence":
                        self.last_family = family
                    self.start_work(slot, queue.popleft())
                    break

    def run(self) -> int:
        from .cci import plan_cci
        from .reporting import render_report

        os.environ.update(THREAD_ENV)
        self.cci_audit = None
        self.cci_plan = self.root / "plans" / "cci"
        self.pool = ProcessPoolExecutor(max_workers=self.cpu_workers, mp_context=mp.get_context("spawn"))
        try:
            self.poll_baseline()
            if self.cci_pending:
                self.submit("cci_plan", plan_cci, Path(self.inputs["cci_t"]), self.settings, self.cci_plan)
            while True:
                self.admission.refresh()
                self.poll_workers()
                self.poll_baseline()
                self.start_baseline_dependents()
                self.poll_cpu()
                self.stage_cci()
                self.dispatch()
                self.stage_existence()
                self.display_progress()
                if not (not self.baseline_finished or self.futures or self.oom_wait or self.cci_pending or any(self.ready.values())
                        or any(slot.work for slot in self.slots)):
                    break
                time.sleep(0.15)
        except KeyboardInterrupt:
            for task in self.tasks:
                if self.records[task.id]["status"] in {"running", "staging", "oom_wait"}:
                    self.update(task, "interrupted")
            raise
        finally:
            for slot in self.slots:
                slot.stop(terminate=slot.work is not None)
                self.admission.release(slot)
            self.pool.shutdown(wait=True, cancel_futures=True)
            if self.baseline_lock is not None:
                self.baseline_lock.close()
                self.baseline_lock = None
        if self.baseline is None:
            for task in self.pending_symmetry():
                self.update(task, "blocked", error="P0 did not complete")
            if self.seurat is not None:
                self.errors["existence"] = "P0 did not complete"
                self.existence_failed = self.existence_total
        p0 = self.records["P0"]
        if self.baseline is not None:
            baseline_metrics = self.root / "plans" / self.baseline.name / "baseline_metrics.json"
            if baseline_metrics.is_file():
                p0["metrics"].update({f"fixed_{name}": value for name, value in read_json(baseline_metrics).items()})
            cci_audit = self.cci_plan / "cci_preflight.json"
            if cci_audit.is_file():
                p0["metrics"].update(read_json(cci_audit)["baseline_topology"])
            self.write_state()
        write_json(self.root / "errors.json", self.errors)
        if self.admission.cards:
            write_json(self.root / "gpu_usage.json", {
                device: {"uuid": uuid, "budget_bytes": self.admission.target,
                         "sampled_card_peak_bytes": self.admission.device_peaks.get(device, 0),
                         "final_worker_limit": self.admission.limit[device],
                         "estimated_process_peak_bytes": {family: peak for (card, family), peak in self.admission.estimated.items()
                                                          if card == device}}
                for device, uuid in self.admission.cards.items()
            })
        self.display_progress()
        render_report(self.root, list(self.records.values()), include_existence=self.report_existence)
        selected = [self.records[task.id]["status"] for task in self.tasks]
        print("Conditions: " + ", ".join(f"{state}={selected.count(state)}" for state in sorted(set(selected))), flush=True)
        return int(bool(self.errors) or any(status in {"failed", "blocked"} for status in selected))
