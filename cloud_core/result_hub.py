"""Cloud result normalization, per-stream tracking and bounded subscribers."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
import json
from time import monotonic_ns, time_ns
from typing import Any, Protocol
from uuid import uuid4

from contracts import DetectionResult

from .tracking import CloudByteTrack

StreamKey = tuple[str, str, str, int]
StreamLaneKey = tuple[str, str, str]


class ResultTracker(Protocol):
    def update(self, result: DetectionResult) -> DetectionResult: ...
    def reset(self, stream_key: StreamKey) -> None: ...


class SubmitStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    DEADLINE_EXCEEDED = "deadline_exceeded"


@dataclass(frozen=True, slots=True)
class ResultSubmit:
    status: SubmitStatus
    result: DetectionResult | None
    tracker_update_ms: float | None = None
    tracking_completed_mono_ns: int | None = None


def result_to_payload(result: DetectionResult) -> dict[str, Any]:
    """Serialize one tracked result using normalized source-frame boxes."""

    return {
        "schema_version": result.schema_version,
        "node_id": result.frame.node_id,
        "camera_id": result.frame.camera_id,
        "stream_id": result.frame.stream_id,
        "stream_epoch": result.frame.stream_epoch,
        "source_frame_id": result.frame.source_frame_id,
        "capture_utc_ns": result.frame.capture_utc_ns,
        "source_pts": result.frame.source_pts,
        "preview_pts": result.preview_pts,
        "model_mode": result.model_mode.value,
        "model_id": result.model_id,
        "confidence_calibration_id": result.confidence_calibration_id,
        "decision_id": result.decision_id,
        "request_id": result.request_id,
        "inference_ms": result.inference_ms,
        "produced_utc_ns": result.produced_utc_ns,
        "boxes": [
            {
                "left": box.left,
                "top": box.top,
                "right": box.right,
                "bottom": box.bottom,
                "confidence": box.confidence,
                "calibrated_confidence": box.calibrated_confidence,
                "class_id": box.class_id,
                "class_name": box.class_name,
                "track_id": box.track_id,
            }
            for box in result.boxes
        ],
        "attributes": dict(result.attributes),
    }


def result_to_json(result: DetectionResult) -> str:
    return json.dumps(result_to_payload(result), separators=(",", ":"), allow_nan=False)


class ResultHub:
    """Serializes each stream's tracking updates and publishes bounded events.

    Results must arrive in increasing source-frame order within a stream. A
    late result is discarded so it cannot rewind ByteTrack's Kalman state.
    Inference admission should therefore keep one ordered result lane per
    stream or perform its own bounded reorder before calling this hub.
    """

    def __init__(
        self, *, tracker: ResultTracker | None = None,
        subscriber_queue_size: int = 16, dedup_capacity_per_stream: int = 512,
    ) -> None:
        if subscriber_queue_size <= 0 or dedup_capacity_per_stream <= 0:
            raise ValueError("subscriber and dedup capacities must be positive")
        self.tracker = tracker or CloudByteTrack()
        self.subscriber_queue_size = subscriber_queue_size
        self.dedup_capacity_per_stream = dedup_capacity_per_stream
        self._last_frame_id: dict[StreamKey, int] = {}
        self._seen: dict[StreamKey, OrderedDict[int, None]] = {}
        # Serialize reconnect epochs through one camera lane. This prevents a
        # late old-epoch result racing a newly created tracker after reconnect.
        self._locks: dict[StreamLaneKey, asyncio.Lock] = {}
        self._subscribers: dict[str, tuple[StreamKey | None, asyncio.Queue[dict[str, Any]]]] = {}
        self.subscriber_dropped: int = 0
        # Per-stream display status for the multi-Edge live page; bounded to recent epochs.
        self._status: OrderedDict[StreamKey, dict[str, Any]] = OrderedDict()
        self.status_capacity = 64

    def subscribe(self, stream_key: StreamKey | None = None) -> tuple[str, asyncio.Queue[dict[str, Any]]]:
        subscriber_id = uuid4().hex
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.subscriber_queue_size)
        self._subscribers[subscriber_id] = (stream_key, queue)
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        self._subscribers.pop(subscriber_id, None)

    async def submit(self, result: DetectionResult, *, deadline_utc_ns: int | None = None,
                     publish_event: bool = True) -> ResultSubmit:
        key = result.frame.stream_key
        lock = self._locks.setdefault(key[:3], asyncio.Lock())
        async with lock:
            if deadline_utc_ns is not None and time_ns() >= deadline_utc_ns:
                return ResultSubmit(SubmitStatus.DEADLINE_EXCEEDED, None)
            frame_id = result.frame.source_frame_id
            seen = self._seen.setdefault(key, OrderedDict())
            if frame_id in seen:
                return ResultSubmit(SubmitStatus.DUPLICATE, None)
            last_frame_id = self._last_frame_id.get(key, -1)
            if frame_id <= last_frame_id:
                return ResultSubmit(SubmitStatus.OUT_OF_ORDER, None)
            # ByteTrack is synchronous CPU work; keep it off the event loop.
            tracking_started = monotonic_ns()
            tracked = await asyncio.to_thread(self.tracker.update, result)
            tracking_completed = monotonic_ns()
            self._last_frame_id[key] = frame_id
            seen[frame_id] = None
            while len(seen) > self.dedup_capacity_per_stream:
                seen.popitem(last=False)
            if deadline_utc_ns is not None and time_ns() >= deadline_utc_ns:
                return ResultSubmit(SubmitStatus.DEADLINE_EXCEEDED, None)
            self._record_status(key, tracked, tracking_completed)
            if publish_event:
                self._publish(key, result_to_payload(tracked))
            return ResultSubmit(SubmitStatus.ACCEPTED, tracked,
                                (tracking_completed - tracking_started) / 1e6,
                                tracking_completed)

    def _record_status(self, key: StreamKey, result: DetectionResult, at_ns: int) -> None:
        status = self._status.pop(key, None) or {"results": 0, "arrivals": deque(maxlen=30)}
        status["results"] += 1
        status["arrivals"].append(at_ns)
        status.update(last_ns=at_ns, model_mode=result.model_mode.value, model_id=result.model_id,
                      boxes=len(result.boxes), source_frame_id=result.frame.source_frame_id)
        self._status[key] = status  # most recent last
        while len(self._status) > self.status_capacity:
            self._status.popitem(last=False)

    def stream_status(self, now_ns: int | None = None) -> list[dict[str, Any]]:
        """Known streams, most recent first: idle age, result rate over the last 30 results."""
        now_ns = monotonic_ns() if now_ns is None else now_ns
        rows = []
        for (node_id, camera_id, stream_id, epoch), status in reversed(self._status.items()):
            arrivals = status["arrivals"]
            span = arrivals[-1] - arrivals[0]
            rows.append({"node_id": node_id, "camera_id": camera_id, "stream_id": stream_id,
                         "stream_epoch": epoch, "age_s": round((now_ns - status["last_ns"]) / 1e9, 2),
                         "results": status["results"], "model_mode": status["model_mode"],
                         "model_id": status["model_id"], "boxes": status["boxes"],
                         "source_frame_id": status["source_frame_id"],
                         "result_fps": round((len(arrivals) - 1) * 1e9 / span, 2) if span > 0 else None})
        return rows

    async def close_stream(self, stream_key: StreamKey) -> None:
        lock = self._locks.get(stream_key[:3])
        if lock is None:
            return
        async with lock:
            self.tracker.reset(stream_key)
            self._last_frame_id.pop(stream_key, None)
            self._seen.pop(stream_key, None)

    def _publish(self, stream_key: StreamKey, payload: dict[str, Any]) -> None:
        for target, queue in tuple(self._subscribers.values()):
            if target is not None and target != stream_key:
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                self.subscriber_dropped += 1
            queue.put_nowait(payload)
