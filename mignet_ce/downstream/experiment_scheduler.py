"""Bounded CPU preparation and per-card training using the shared worker slots."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from time import time_ns
from types import SimpleNamespace

from .post_perbutation.config import read_json, write_json
from .post_perbutation.scheduler import (
    GIB, GpuAdmission, Slot, THREAD_ENV, device_tokens, probe_devices, worker_loop,
)


MODULE = "mignet_ce.downstream.experiment_scheduler"


@dataclass(frozen=True)
class Execution:
    devices: tuple[str, ...] = ("cpu",)
    cpu_workers: int = 16
    workers_per_gpu: int = 3
    gpu_budget_gib: float = 18.0

    def __post_init__(self):
        from math import isfinite

        device_tokens(list(self.devices))
        trainers = 1 if self.devices == ("cpu",) else len(self.devices) * self.workers_per_gpu
        # Leave room for the launcher, multiprocessing tracker and GPU probe.
        if not 1 <= self.cpu_workers <= min(61 if os.name == "nt" else 64, 64 - trainers - 3):
            raise ValueError("CPU preparation and training must fit in the 64-process allocation")
        if not 1 <= self.workers_per_gpu <= 3:
            raise ValueError("Use one to three training processes per GPU")
        if not isfinite(self.gpu_budget_gib) or not 0 < self.gpu_budget_gib <= 18:
            raise ValueError("Per-card project budget must be in (0, 18] GiB")


def add_execution_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--devices", nargs="+", help="CPU or a list of visible CUDA devices, e.g. cuda:0 cuda:1")
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--gpu-budget-gib", type=float, default=18.0)


def execution_from_args(args):
    if args.devices is None:
        return None
    return Execution(tuple(args.devices), args.cpu_workers, args.workers_per_gpu, args.gpu_budget_gib)


@dataclass(frozen=True)
class Job:
    id: str
    prepare: object
    args: tuple


@dataclass(frozen=True)
class Training:
    function: object
    args: tuple
    retry_outputs: tuple[Path, ...] = ()


def dump_transfer(path, value):
    """Send data through this invocation's private staging directory."""
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_transfer(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def save_prepared(path, prepared):
    dump_transfer(path, (replace(prepared, macro_pij_builder=None), prepared.macro_pij_builder.formal_assets))


def load_prepared(path):
    from wyt_deltaei_coarse_grain.complete_combined import build_macro_pij_builder

    prepared, (stage_t, stage_tp, enhancer, alpha, residual) = load_transfer(path)
    return replace(prepared, macro_pij_builder=build_macro_pij_builder(
        stage_t, stage_tp, grn_enhancer=enhancer, alpha_cci=alpha, target_residual_block=residual))


def _prepare(job, directory):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return job.prepare(directory, *job.args)


def _cpu_training(call):
    return call.function("cpu", *call.args)


def _worker(job, device):
    import torch

    call = load_transfer(job["call"])
    if job.get("retry"):
        for path in call.retry_outputs:
            if path.exists():
                path.rename(path.with_name(f"{path.name}.failed-{time_ns()}"))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    output = call.function("cuda" if device.startswith("cuda") else "cpu", *call.args)
    if device.startswith("cuda"):
        output = {**output, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return {"output": output, "metrics": {"peak_cuda_reserved_bytes": output.get("peak_cuda_reserved_bytes")}}


def _report_progress(root, group, total, results, futures, ready, active, cached, admission,
                     project_peaks=None, retries=0):
    status = {"total": total, "done": len(results) + cached, "reused": cached,
              "failed": sum("error" in result or result.get("training_status") in {
                  "DIRECT_STATE_FAILED", "BLOCKED_UPSTREAM_IMPORT"} for result in results.values()),
              "cpu": len(futures), "ready": len(ready), "gpu": len(active), "oom_retries": retries}
    if admission and admission.cards:
        status["cards"] = {}
        for device in admission.cards:
            memory = admission._memory(device)
            local = [slot for slot in active if slot.device == device]
            values = [memory.per_pid.get(slot.process.pid) for slot in local]
            usage = sum(values) if all(value is not None for value in values) else None
            if usage is not None:
                project_peaks[device] = max(usage, project_peaks.get(device, 0))
            status["cards"][device] = {
                "total_used_gib": round(memory.used / GIB, 3),
                "project_used_gib": None if usage is None else round(usage / GIB, 3), "active": len(local)}
    path = root / "progress.json"
    existing = read_json(path) if path.is_file() else {}
    if status != existing.get(group):
        write_json(path, {**existing, group: status})
    counts = tuple(key for key in status if key != "cards")
    if any(status[key] != existing.get(group, {}).get(key) for key in counts):
        marks = 10 * status['done'] // max(1, total)
        print(f"{group} [{'#' * marks}{'.' * (10 - marks)}] {status['done']}/{total} "
              f"cpu={status['cpu']} ready={status['ready']} gpu={status['gpu']} "
              f"reused={cached} failed={status['failed']} retry={retries}", flush=True)


def run_jobs(root, group, jobs, execution, *, cached=0, fail_fast=False, on_result=None):
    """Stage a bounded window on CPU and consume its training calls as devices free up."""
    from collections import deque
    from time import sleep

    root.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    total = len(jobs) + cached
    if not jobs:
        _report_progress(root, group, total, {}, {}, (), (), cached, None)
        return []
    tokens = device_tokens(list(execution.devices))
    cuda = execution.devices != ("cpu",)
    admission = GpuAdmission(probe_devices(tokens), execution.gpu_budget_gib, execution.workers_per_gpu,
                             execution.workers_per_gpu, owned_budget=True) if cuda else None
    os.environ.update(THREAD_ENV)
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_root = root / "worker_logs"
    log_root.mkdir(exist_ok=True)
    slots = [Slot(f"{device}-{index}", device, token, log_root / f"{device.replace(':', '-')}-{index}.log",
                  index=index, worker_module=MODULE)
             for device, token in tokens.items() for index in range(execution.workers_per_gpu)] if cuda else []
    futures, active, results, stages = {}, {}, {}, {}
    waiting, ready = deque(enumerate(jobs)), deque()
    solo = {device: deque() for device in tokens}
    project_peaks, reserved_peaks = {}, {}
    retries = 0
    window = execution.cpu_workers + max(1, len(slots))

    def finish(index, result):
        if on_result is not None:
            on_result(index, result)
        results[index] = result
        stages.pop(index).cleanup()

    def start(slot, index, directory, call, retry):
        payload = directory / ("retry.pkl" if retry else "training.pkl")
        dump_transfer(payload, call)
        slot.send({"call": str(payload), "id": jobs[index].id, "family": "training", "retry": retry,
                   "signature": group, "device": slot.device, "directory": str(directory)})
        admission.started(slot)
        active[slot.name] = index, directory, call, retry
    with TemporaryDirectory(prefix="formal-stage-", dir=root) as temporary:
        with ProcessPoolExecutor(max_workers=min(execution.cpu_workers, len(jobs)),
                                 mp_context=mp.get_context("spawn")) as pool:
            try:
                while waiting or futures or ready or active or any(solo.values()):
                    if admission:
                        admission.refresh()
                    for future in [item for item in futures if item.done()]:
                        index, directory, stage = futures.pop(future)
                        try:
                            value = future.result()
                        except Exception as error:
                            finish(index, {"error": str(error), "stage": stage})
                        else:
                            if isinstance(value, Training):
                                ready.append((index, directory, value))
                            else:
                                finish(index, value)
                    for slot in slots:
                        if slot.name not in active:
                            continue
                        index, directory, call, retry = active[slot.name]
                        result_path = directory / "result.json"
                        admission.sample(slot)
                        if not result_path.is_file() and slot.process.poll() is None:
                            continue
                        result = read_json(result_path) if result_path.is_file() else {
                            "status": "failed", "error": f"Worker exited early; see {slot.log}"}
                        admission.finished(slot, "training", result)
                        reserved = result.get("metrics", {}).get("peak_cuda_reserved_bytes")
                        if reserved is not None:
                            reserved_peaks[slot.device] = max(reserved, reserved_peaks.get(slot.device, 0))
                        with (log_root / f"{jobs[index].id.replace(':', '_').replace('/', '_')}.log").open("a", encoding="utf-8") as stream:
                            log = directory / "worker.log"
                            if log.is_file():
                                stream.write(log.read_text(encoding="utf-8"))
                        slot.stop()
                        admission.release(slot)
                        del active[slot.name]
                        if result["status"] == "oom" and not retry and call.retry_outputs:
                            result_path.rename(directory / "oom_result.json")
                            solo[slot.device].append((index, directory, call))
                            retries += 1
                            print(f"[OOM] {jobs[index].id} {slot.device} queued for a solo retry", flush=True)
                        else:
                            finish(index, result["output"] if result["status"] == "complete" else {
                                "error": result["error"], "stage": "train", "oom": result["status"] == "oom"})
                    if fail_fast and any("error" in result for result in results.values()):
                        break
                    if cuda:
                        for slot in slots:
                            if slot.name in active:
                                continue
                            siblings = [active[other.name] for other in slots
                                        if other.device == slot.device and other.name in active]
                            if solo[slot.device]:
                                work = SimpleNamespace(task=SimpleNamespace(family="training"), retry=True)
                                if not siblings and slot.index == 0 and admission.admit(slot, work):
                                    start(slot, *solo[slot.device].popleft(), retry=True)
                                continue
                            if not ready or any(item[3] for item in siblings):
                                continue
                            work = SimpleNamespace(task=SimpleNamespace(family="training"), retry=False)
                            if admission.admit(slot, work):
                                start(slot, *ready.popleft(), retry=False)
                    else:
                        while ready and len(futures) < execution.cpu_workers:
                            index, directory, call = ready.popleft()
                            futures[pool.submit(_cpu_training, call)] = index, directory, "train"
                    while waiting and len(futures) < execution.cpu_workers and len(stages) < window:
                        index, job = waiting.popleft()
                        stages[index] = TemporaryDirectory(dir=temporary, prefix=f"{index}-")
                        directory = Path(stages[index].name)
                        futures[pool.submit(_prepare, job, directory)] = index, directory, "prepare"
                    _report_progress(root, group, total, results, futures, ready,
                                     [slot for slot in slots if slot.name in active], cached, admission, project_peaks, retries)
                    if active or (cuda and ready) or any(solo.values()):
                        sleep(0.2)
                    elif futures:
                        wait(futures, timeout=0.2, return_when="FIRST_COMPLETED")
            finally:
                for slot in slots:
                    slot.stop(terminate=True)
                    admission.release(slot)
    if admission is not None:
        usage_path = root / "gpu_usage.json"
        usage = read_json(usage_path) if usage_path.is_file() else {}
        usage[group] = {device: {
            "gpu_uuid": uuid, "project_budget_bytes": admission.target, "maximum_workers": execution.workers_per_gpu,
            "card_sample_peak_bytes": admission.device_peaks.get(device),
            "project_sample_peak_bytes": project_peaks.get(device) or None,
            "max_worker_reserved_bytes": reserved_peaks.get(device),
        } for device, uuid in admission.cards.items()}
        write_json(usage_path, usage)
    return [results.get(index, {"error": "Cancelled after earlier failure", "stage": "schedule"})
            for index in range(len(jobs))]


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "_worker":
    worker_loop(sys.argv[2], handler=_worker)
