"""Slow lane on the Edge (v4 step 1, 8/11): a metadata BUS, a short frame ring and the tasks that read them.

The fast lane (detector -> tracker -> result) never waits for anything here: it hands every frame to the ring and every
result to the BUS, both O(1) and non-blocking. Each slow task has its own bounded queue on the BUS; when a task falls
behind, its oldest events are dropped (counted), never the fast lane's frames. Slow tasks run at fixed low rates
(decision slow-lane-v4: no DRL control of their frequency; not real time except alerts).

Frames carry `hires` (full resolution) on every `hires_every`-th frame when the source provides it: the hardware path
reads 640x640 for analytics plus a low-rate full-resolution branch (media/hw_capture.py); on the CPU path the analytics
frame itself is full resolution.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from .fanout import BoundedLatestQueue


@dataclass(frozen=True, slots=True)
class SlowEvent:
    frame: Any    # edge_agent.CapturedFrame the result belongs to (pixels, hires, identity, source_size)
    result: Any   # contracts.DetectionResult with normalized boxes and track IDs


class MetadataBus:
    """Fan-out of results to slow tasks; every subscriber has its own drop-oldest queue."""

    def __init__(self) -> None:
        self.queues: dict[str, BoundedLatestQueue[SlowEvent]] = {}
        self.published = 0

    def subscribe(self, name: str, capacity: int = 8) -> BoundedLatestQueue[SlowEvent]:
        if name in self.queues:
            raise ValueError(f"duplicate slow task {name!r}")
        self.queues[name] = BoundedLatestQueue(capacity)
        return self.queues[name]

    def publish(self, event: SlowEvent) -> None:
        self.published += 1
        for queue in self.queues.values():
            queue.put_latest(event)


class FrameRing:
    """The last `seconds` of captured frames, oldest first (short clips for UC4, density frames), and the last
    `hires_seconds` of full-resolution frames (plate crops). 8/11: a 1080x1920 frame is 6.2 MB; they are only needed
    until the frame's result is back (~0.1-0.3 s), so they are kept 1 s, not 3 (93 -> ~19 MB per stream at 3 per s)."""

    def __init__(self, seconds: float = 3.0, hires_seconds: float = 1.0) -> None:
        if seconds <= 0 or hires_seconds <= 0:
            raise ValueError("ring seconds must be positive")
        self.span_ns, self.hires_span_ns = round(seconds * 1e9), round(hires_seconds * 1e9)
        self.frames: deque = deque()
        self.hires: deque = deque()

    def add(self, frame) -> None:
        if getattr(frame, "hires", None) is not None:
            self.hires.append(frame)
            if dataclasses.is_dataclass(frame) and frame.hires is not frame.pixels:
                frame = dataclasses.replace(frame, hires=None)  # the 3 s ring must not pin the big array
        self.frames.append(frame)
        now = frame.capture_mono_ns
        while self.frames and now - self.frames[0].capture_mono_ns > self.span_ns:
            self.frames.popleft()
        while self.hires and now - self.hires[0].capture_mono_ns > self.hires_span_ns:
            self.hires.popleft()

    def get(self, source_frame_id: int):
        for frame in reversed(self.frames):
            if frame.identity.source_frame_id == source_frame_id:
                return frame
        return None

    def hires_near(self, source_frame_id: int, max_gap: int = 2):
        """The full-resolution frame closest to source_frame_id within max_gap frames, or None."""
        best = None
        for frame in self.hires:
            gap = abs(frame.identity.source_frame_id - source_frame_id)
            if gap <= max_gap and (best is None or gap < best[0]):
                best = (gap, frame)
        return None if best is None else best[1]

    def sequence(self, count: int, every_ns: int):
        """Up to `count` frames spaced at least every_ns apart, ending with the newest (UC4 short clips)."""
        picked, last = [], None
        for frame in reversed(self.frames):
            if last is None or last - frame.capture_mono_ns >= every_ns:
                picked.append(frame)
                last = frame.capture_mono_ns
                if len(picked) == count:
                    break
        return picked[::-1]

    @property
    def megabytes(self) -> float:
        total = sum(getattr(frame.pixels, "nbytes", 0) for frame in self.frames)
        total += sum(frame.hires.nbytes for frame in self.hires if frame.hires is not frame.pixels)
        return total / 1e6


class SlowTask(Protocol):
    name: str

    async def run(self, queue: BoundedLatestQueue[SlowEvent], ring: FrameRing, stop: asyncio.Event) -> dict: ...


class SlowLane:
    def __init__(self, tasks, *, ring_seconds: float = 3.0, queue_capacity: int = 8, sink=None) -> None:
        self.sink = sink   # edge_agent.events.SlowSink: its retries run beside the tasks
        self.bus = MetadataBus()
        self.ring = FrameRing(ring_seconds)
        self.tasks = [(task, self.bus.subscribe(task.name, queue_capacity)) for task in tasks]

    def on_frame(self, frame) -> None:
        self.ring.add(frame)

    def on_result(self, frame, result) -> None:
        self.bus.publish(SlowEvent(frame, result))

    async def run(self, stop: asyncio.Event) -> dict:
        """Until `stop` (the media lane finished) and each task drained its queue; -> per-task stats."""
        async def guarded(task, queue):
            # 8/11: a crashed slow task stopped the whole agent (fast lane included). Now it only stops itself; its
            # queue keeps dropping its oldest events, and the error is reported in the run summary.
            try:
                return await task.run(queue, self.ring, stop)
            except Exception as exc:
                print(f"[slow lane] task {task.name} stopped: {type(exc).__name__}: {exc}", flush=True)
                return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        retries = getattr(self.sink, "run_retries", None)
        results = await asyncio.gather(*(guarded(task, queue) for task, queue in self.tasks),
                                       *([retries(stop)] if retries else []))
        out = {"published": self.bus.published, "ring_mb": round(self.ring.megabytes, 1),
               "tasks": {task.name: {**stats, "dropped": queue.dropped}
                         for (task, queue), stats in zip(self.tasks, results)}}
        if hasattr(self.sink, "stats"):
            out["sink"] = {**self.sink.stats,
                           "outbox_pending": len(self.sink.outbox.pending) if self.sink.outbox is not None else None}
        return out


async def next_event(queue: BoundedLatestQueue[SlowEvent], stop: asyncio.Event, poll_s: float = .1):
    """The next event, or None once `stop` is set and the queue is empty."""
    while True:
        try:
            return await asyncio.wait_for(queue.get(), poll_s)
        except asyncio.TimeoutError:
            if stop.is_set() and not queue.pending:
                return None


class RingCheckTask:
    """`--slow-tasks check`: proves the plumbing on a device. Counts events, how many found a full-resolution frame
    within 2 frames, its size, and the ring's memory."""

    name = "check"

    async def run(self, queue, ring, stop) -> dict:
        events = with_hires = 0
        size, ring_mb = None, 0.0
        while (event := await next_event(queue, stop)) is not None:
            events += 1
            frame = ring.hires_near(event.frame.identity.source_frame_id)
            if frame is not None:
                with_hires += 1
                size = list(frame.hires.shape)
            ring_mb = max(ring_mb, ring.megabytes)
        return {"events": events, "with_hires": with_hires, "hires_shape": size, "ring_mb_max": round(ring_mb, 1)}


def _tasks():
    from .crowd_task import CrowdTask, GatheringTask
    from .disorder_task import CrashTask, FightTask, LoiterTask
    from .plate_task import PlateTask
    from .slow_tasks import CongestionTask, CountTask
    return {"check": lambda sink, config: RingCheckTask(), "count": CountTask, "congestion": CongestionTask,
            "plates": PlateTask, "crowd": CrowdTask, "gathering": GatheringTask,
            "fight": FightTask, "crash": CrashTask, "loiter": LoiterTask}


def build_slow_lane(names: str, *, ring_seconds: float = 3.0, sink=None, config=None,
                    outbox_path=None) -> SlowLane | None:
    """sink(kind, payload): async, where counts and events go (the Kafka client's publish_slow); None = keep nothing."""
    names = [n.strip() for n in names.split(",") if n.strip()]
    tasks = _tasks()
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise ValueError(f"unknown slow tasks {unknown}; known: {sorted(tasks)}")
    if not names:
        return None
    if sink is not None and not hasattr(sink, "stats"):   # a bare transport send(kind, payload, wait)
        from .events import EventOutbox, SlowSink
        sink = SlowSink(sink, EventOutbox(outbox_path) if outbox_path else None)
    return SlowLane([tasks[n](sink, config) for n in names], ring_seconds=ring_seconds, sink=sink)
