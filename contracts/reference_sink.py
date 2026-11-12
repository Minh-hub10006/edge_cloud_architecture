"""Bounded async sender for compact media identity sidecars."""

from __future__ import annotations

import asyncio

from contracts import MediaFrameRef


class MediaReferenceSink:
    """Keep Kafka/gRPC sidecar delivery off the media encoder write path."""

    def __init__(self, client, *, capacity: int = 2048) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.client = client
        self.queue: asyncio.Queue[MediaFrameRef] = asyncio.Queue(capacity)
        self.dropped = 0
        self.failed = 0
        self._closing = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("media reference sink already started")
        self._task = asyncio.create_task(self._run())

    async def submit(self, reference: MediaFrameRef) -> None:
        if self._closing.is_set():
            raise RuntimeError("media reference sink is closing")
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(reference)

    async def _run(self) -> None:
        while not self._closing.is_set() or not self.queue.empty():
            try:
                reference = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            try:
                last_error = None
                for attempt in range(3):
                    try:
                        await self.client.publish_media_reference(reference)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < 2:
                            await asyncio.sleep(0.1 * (attempt + 1))
                if last_error is not None:
                    self.failed += 1
            finally:
                self.queue.task_done()

    async def close(self, timeout_s: float = 3.0) -> None:
        self._closing.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout_s)
            except asyncio.TimeoutError:
                self.failed += 1
            finally:
                self._task = None
                while not self.queue.empty():
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped += 1
