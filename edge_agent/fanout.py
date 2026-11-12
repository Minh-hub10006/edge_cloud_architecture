"""Independent bounded media and analytics queues for one capture source."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .capture import CapturedFrame

T = TypeVar("T")


class BoundedLatestQueue(Generic[T]):
    """On overflow, discard the oldest queued item to preserve fresh video."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)
        self.dropped = 0
        self.last_batch_size = 1

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def put_latest(self, item: T) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(item)

    async def get(self) -> T:
        self.last_batch_size = 1
        return await self._queue.get()

    async def get_latest(self) -> T:
        item = await self.get()
        drained = 0
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self.dropped += 1
            drained += 1
        self.last_batch_size = drained + 1
        return item

    def get_nowait(self) -> T:
        return self._queue.get_nowait()


@dataclass(frozen=True, slots=True)
class CaptureStats:
    captured: int
    media_dropped: int
    analytics_dropped: int


class EdgeFanout:
    def __init__(self, *, media_capacity: int, analytics_capacity: int) -> None:
        self.media: BoundedLatestQueue[CapturedFrame] = BoundedLatestQueue(media_capacity)
        self.analytics: BoundedLatestQueue[CapturedFrame] = BoundedLatestQueue(analytics_capacity)
        self.captured = 0

    def publish(self, frame: CapturedFrame, *, analytics: bool = True) -> None:
        self.media.put_latest(frame)
        if analytics:
            self.analytics.put_latest(frame)
        self.captured += 1

    def snapshot(self) -> CaptureStats:
        return CaptureStats(self.captured, self.media.dropped, self.analytics.dropped)


async def capture_to_fanout(source: AsyncIterable[CapturedFrame], fanout: EdgeFanout) -> CaptureStats:
    async for frame in source:
        fanout.publish(frame)
    return fanout.snapshot()
