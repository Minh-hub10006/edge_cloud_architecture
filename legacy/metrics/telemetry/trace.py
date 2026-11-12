"""Canonical per-frame trace shared by YOLO, DFINE and Blend.

Only durations captured on the same machine use ``perf_counter_ns``. Cloud
durations must be measured on the cloud and attached with ``observe_ms``; this
avoids subtracting unsynchronised edge/cloud clocks.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 3

STAGES = (
    "frame_read",
    "resize",
    "state_aggregate",
    "drl_decision",
    "model_switch",
    "preprocess",
    "edge_inference",
    "tensor_encode",
    # Transport-neutral stages make gRPC and Kafka D-FINE runs comparable.
    # Protocol-specific gRPC stages below remain for report compatibility.
    "transport_pack",
    "transport_enqueue",
    "transport_publish",
    "transport_result_decode",
    "transport_rtt",
    "transport_wire",
    "grpc_queue",
    "protobuf_pack",
    "grpc_transport",
    "grpc_uplink",
    "cloud_queue",
    "cloud_decode",
    "cloud_inference",
    "grpc_downlink",
    "grpc_rtt",
    "feedback_deserialize",
    "postprocess",
    "scene_metrics",
    "tracking",
    "e2e",
)


@dataclass
class FrameTrace:
    run_id: str
    frame_id: int | str
    mode: str
    backend: str = "unknown"
    status: str = "pending"
    durations_ms: dict[str, float | None] = field(
        default_factory=lambda: {stage: None for stage in STAGES}
    )
    attributes: dict[str, Any] = field(default_factory=dict)
    wall_started_ns: int = field(default_factory=time.time_ns)
    wall_finished_ns: int | None = None
    _starts_ns: dict[str, int] = field(default_factory=dict, repr=False)

    def start(self, stage: str) -> None:
        self._check_stage(stage)
        if stage in self._starts_ns:
            raise RuntimeError(f"stage already started: {stage}")
        self._starts_ns[stage] = time.perf_counter_ns()

    def stop(self, stage: str) -> float:
        self._check_stage(stage)
        started_ns = self._starts_ns.pop(stage, None)
        if started_ns is None:
            raise RuntimeError(f"stage not started: {stage}")
        duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self.durations_ms[stage] = duration_ms
        return duration_ms

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)

    def observe_ms(self, stage: str, duration_ms: float | None) -> None:
        """Attach a duration measured elsewhere, e.g. by the cloud process."""
        self._check_stage(stage)
        if duration_ms is None:
            self.durations_ms[stage] = None
            return
        value = float(duration_ms)
        if value < 0:
            raise ValueError(f"duration must be non-negative: {stage}={value}")
        self.durations_ms[stage] = value

    def finish(self, status: str = "ok", **attributes: Any) -> None:
        if self._starts_ns:
            open_stages = ", ".join(sorted(self._starts_ns))
            raise RuntimeError(f"cannot finish with open stages: {open_stages}")
        self.status = status
        self.attributes.update(attributes)
        self.wall_finished_ns = time.time_ns()

    def is_running(self, stage: str) -> bool:
        self._check_stage(stage)
        return stage in self._starts_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "frame_id": self.frame_id,
            "mode": self.mode.upper(),
            "backend": self.backend,
            "status": self.status,
            "wall_started_ns": self.wall_started_ns,
            "wall_finished_ns": self.wall_finished_ns,
            "durations_ms": {
                stage: (
                    round(value, 6) if value is not None else None
                )
                for stage, value in self.durations_ms.items()
            },
            "attributes": self.attributes,
        }

    @staticmethod
    def _check_stage(stage: str) -> None:
        if stage not in STAGES:
            raise KeyError(f"unknown benchmark stage: {stage}")


class JsonlTraceWriter:
    """Thread-safe, line-buffered writer with no per-frame fsync."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self.count = 0

    def write(self, trace: FrameTrace | dict[str, Any]) -> None:
        row = trace.to_dict() if isinstance(trace, FrameTrace) else dict(trace)
        payload = json.dumps(
            row, ensure_ascii=False, separators=(",", ":")
        )
        with self._lock:
            self._file.write(payload + "\n")
            self.count += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
