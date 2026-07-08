from __future__ import annotations

import json
import os
import queue as queue_mod
import threading
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Mapping

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - depends on optional runtime package
    tqdm = None


@dataclass
class ProgressEvent:
    event: str
    status: str = "update"
    phase: str | None = None
    message: str | None = None

    organ: str | None = None
    pair: str | None = None
    lower_layer: str | None = None
    upper_layer: str | None = None
    stage: str | int | None = None
    layer: str | None = None
    space: str | None = None
    t0: str | int | None = None
    t1: str | int | None = None
    time_pair: str | None = None

    current: int | None = None
    total: int | None = None
    advance: int | None = None
    unit: str | None = None

    shape: tuple[int, int] | list[int] | None = None
    nnz: int | None = None
    estimated_bytes: int | None = None
    workers: int | None = None

    backend: str | None = None
    device: str | None = None
    dtype: str | None = None
    cuda_available: bool | None = None
    cuda_memory_allocated: int | None = None
    cuda_memory_reserved: int | None = None
    fallback_reason: str | None = None

    elapsed_seconds: float | None = None
    pid: int = field(default_factory=os.getpid)
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressReporter:
    def emit(self, event: str, **kwargs) -> None:
        raise NotImplementedError


class NullProgressReporter(ProgressReporter):
    def emit(self, event: str, **kwargs) -> None:
        return


class QueueProgressReporter(ProgressReporter):
    def __init__(self, queue, default_scope: Mapping[str, object] | None = None):
        self.queue = queue
        self.default_scope = dict(default_scope or {})

    def emit(self, event: str, **kwargs) -> None:
        payload = dict(self.default_scope)
        payload.update(kwargs)
        self.queue.put(ProgressEvent(event=event, **payload).to_dict())


_CURRENT_REPORTER: ProgressReporter = NullProgressReporter()


def set_progress_reporter(reporter: ProgressReporter | None) -> None:
    global _CURRENT_REPORTER
    _CURRENT_REPORTER = reporter or NullProgressReporter()


def get_progress_reporter() -> ProgressReporter:
    return _CURRENT_REPORTER


def emit_progress(event: str, **kwargs) -> None:
    _CURRENT_REPORTER.emit(event, **kwargs)


def get_progress_queue_or_none():
    reporter = get_progress_reporter()
    return getattr(reporter, "queue", None)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _prune_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _prune_none(item)
            for key, item in value.items()
            if item is not None and item != {}
        }
    if isinstance(value, (list, tuple)):
        return [_prune_none(item) for item in value]
    return value


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{value}B"


class ProgressMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        total_pairs: int,
        log_path: Path | None,
        refresh_interval: float = 0.5,
        process_safe: bool = False,
    ):
        self.enabled = bool(enabled)
        self.total_pairs = int(total_pairs)
        self.log_path = Path(log_path) if log_path is not None else None
        self.refresh_interval = float(refresh_interval)
        self.process_safe = bool(process_safe)
        self._manager = None
        self._queue = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle = None
        self._bars: dict[str, object] = {}

    def __enter__(self) -> "ProgressMonitor":
        if not self.enabled:
            set_progress_reporter(NullProgressReporter())
            return self
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.log_path.open("w", encoding="utf-8")
        if self.process_safe:
            self._manager = Manager()
            self._queue = self._manager.Queue()
        else:
            self._queue = queue_mod.Queue()
        set_progress_reporter(QueueProgressReporter(self._queue))
        self._create_bars()
        self._thread = threading.Thread(target=self._consume, name="mignet-progress-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            set_progress_reporter(NullProgressReporter())
            return
        self.drain()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.drain()
        set_progress_reporter(NullProgressReporter())
        self._close_bars()
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    @property
    def queue(self):
        return self._queue

    def drain(self) -> None:
        if self._queue is None:
            return
        while True:
            try:
                event = self._queue.get_nowait()
            except Exception:
                break
            self._handle_event(event)

    def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue_mod.Empty:
                continue
            except Exception:
                continue
            self._handle_event(event)

    def _create_bars(self) -> None:
        if tqdm is None:
            return
        self._bars["pairs"] = tqdm(
            total=self.total_pairs,
            desc="Pairs",
            unit="pair",
            mininterval=self.refresh_interval,
            position=0,
            leave=True,
        )

    def _close_bars(self) -> None:
        for bar in self._bars.values():
            close = getattr(bar, "close", None)
            if close is not None:
                close()
        self._bars.clear()

    def _reset_bar(self, key: str, *, desc: str, total: int, unit: str, position: int, postfix: str | None = None) -> None:
        if tqdm is None:
            return
        existing = self._bars.get(key)
        if existing is not None:
            existing.close()
        bar = tqdm(
            total=max(0, int(total)),
            desc=desc,
            unit=unit,
            mininterval=self.refresh_interval,
            position=position,
            leave=False,
        )
        if postfix:
            bar.set_postfix_str(postfix)
        self._bars[key] = bar

    def _add_or_reset_bar(self, key: str, *, desc: str, total: int, unit: str, position: int) -> None:
        if tqdm is None:
            return
        existing = self._bars.get(key)
        if existing is None:
            self._reset_bar(key, desc=desc, total=total, unit=unit, position=position)
            return
        existing.total = int(existing.total or 0) + max(0, int(total))
        existing.refresh()

    def _advance_bar(self, key: str, advance: int | None = None) -> None:
        bar = self._bars.get(key)
        if bar is not None:
            bar.update(int(advance or 1))

    def _set_postfix(self, key: str, text: str) -> None:
        bar = self._bars.get(key)
        if bar is not None and text:
            bar.set_postfix_str(text)

    def _write_jsonl(self, event: Mapping[str, Any]) -> None:
        if self._handle is None:
            return
        payload = _prune_none(dict(event))
        self._handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        self._handle.flush()

    def _handle_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        self._write_jsonl(event)
        name = str(event.get("event", ""))

        if name == "pair_start":
            pair_label = " ".join(
                value
                for value in (str(event.get("organ") or ""), str(event.get("pair") or ""))
                if value
            )
            self._reset_bar("pair_phase", desc="Pair phases", total=5, unit="phase", position=1, postfix=pair_label)
            for key in ("network_layers", "cci_scan_files", "cci_build_files", "pij_kernels", "pij_export"):
                existing = self._bars.pop(key, None)
                if existing is not None:
                    existing.close()
            return

        if name in {"pair_phase_done", "pair_phase_skip"}:
            self._advance_bar("pair_phase", event.get("advance") or 1)
            stage = event.get("stage")
            if stage:
                self._set_postfix("pair_phase", str(stage))
            return

        if name in {"pair_done", "pair_error"}:
            self._advance_bar("pairs", event.get("advance") or 1)
            return

        if name == "network_layers_total":
            self._reset_bar("network_layers", desc="Network layers", total=int(event.get("total") or 0), unit="layer", position=2)
            return
        if name == "layer_build_done":
            self._advance_bar("network_layers", event.get("advance") or 1)
            return

        if name == "cci_scan_total":
            self._add_or_reset_bar("cci_scan_files", desc="CCI scan LR files", total=int(event.get("total") or 0), unit="file", position=3)
            return
        if name == "cci_scan_advance":
            self._advance_bar("cci_scan_files", event.get("advance") or 1)
            return
        if name == "cci_build_total":
            self._add_or_reset_bar("cci_build_files", desc="CCI build LR files", total=int(event.get("total") or 0), unit="file", position=4)
            return
        if name == "cci_build_advance":
            self._advance_bar("cci_build_files", event.get("advance") or 1)
            return

        if name == "pij_kernels_total":
            self._reset_bar("pij_kernels", desc="PIJ kernels", total=int(event.get("total") or 0), unit="kernel", position=5)
            return
        if name in {"pij_kernel_start", "pij_backend_selected", "pij_gpu_memory_check", "pij_gpu_memory_fallback"}:
            bits = []
            for field_name in ("space", "time_pair", "backend", "device", "dtype"):
                value = event.get(field_name)
                if value:
                    bits.append(str(value))
            shape = event.get("shape")
            if shape:
                bits.append("shape=" + "x".join(map(str, shape)))
            estimated = _format_bytes(event.get("estimated_bytes"))
            if estimated:
                bits.append(f"est={estimated}")
            if event.get("fallback_reason"):
                bits.append(f"fallback={event['fallback_reason']}")
            self._set_postfix("pij_kernels", " ".join(bits))
            return
        if name == "pij_kernel_done":
            self._advance_bar("pij_kernels", event.get("advance") or 1)
            return

        if name == "pij_export_total":
            self._reset_bar("pij_export", desc="PIJ export", total=int(event.get("total") or 0), unit="matrix", position=6)
            return
        if name == "pij_export_matrix_done":
            self._advance_bar("pij_export", event.get("advance") or 1)
            return

        if tqdm is None and name.endswith(("_start", "_done", "_error", "_fallback")):
            message = event.get("message") or name
            print(f"[progress] {message}", flush=True)
