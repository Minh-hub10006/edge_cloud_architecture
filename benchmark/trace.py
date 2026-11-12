"""Bounded asynchronous JSONL writer, adapted from legacy trace concepts."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
from typing import Any

from .config import RunConfig, finite_nonnegative


def frame_event(config: RunConfig, *, frame_id: int, phase: str, status: str,
                started_ns: int, finished_ns: int, durations_ms: dict[str, float | None],
                detections: int = 0, error: str | None = None) -> dict[str, Any]:
    if type(frame_id) is not int or frame_id < 0 or finished_ns < started_ns:
        raise ValueError("invalid frame/timing")
    if phase not in {"warmup", "measure"} or status not in {"ok", "error", "timeout", "dropped"}:
        raise ValueError("invalid phase/status")
    if type(detections) is not int or detections < 0:
        raise ValueError("invalid detections")
    stages = {key: None if value is None else finite_nonnegative(value, key)
              for key, value in durations_ms.items()}
    return {"schema_version": 1, **config.as_dict(), "frame_id": frame_id,
            "phase": phase, "status": status, "started_mono_ns": started_ns,
            "finished_mono_ns": finished_ns, "durations_ms": stages,
            "detections": detections, "error": error}


class JsonlWriter:
    """Never silently loses measurements or overwrites old results."""

    def __init__(self, path: Path, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("writer capacity must be positive")
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self._queue: queue.Queue[str | None] = queue.Queue(capacity)
        self._error: BaseException | None = None
        self._closed = False
        self.written = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="bench-jsonl")
        self._thread.start()

    def write(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("writer closed")
        if self._error is not None:
            raise RuntimeError("trace writer failed") from self._error
        line = json.dumps(event, allow_nan=False, separators=(",", ":"))
        try:
            self._queue.put_nowait(line)
        except queue.Full as exc:
            raise RuntimeError("trace queue full; measurement invalid, no silent drop") from exc

    def _run(self) -> None:
        try:
            while True:
                line = self._queue.get()
                if line is None:
                    break
                self._file.write(line + "\n")
                self.written += 1
            self._file.flush()
        except BaseException as exc:
            self._error = exc
        finally:
            self._file.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._thread.is_alive() and self._error is None:
                try:
                    self._queue.put(None, timeout=5)
                except queue.Full as exc:
                    raise RuntimeError("trace close queue timeout") from exc
            self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("trace drain timeout")
        if self._error is not None:
            raise RuntimeError("trace writer failed") from self._error
