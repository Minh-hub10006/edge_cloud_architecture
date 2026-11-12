"""Decode MediaMTX RTSP frames and join them to Edge MediaFrameRef sidecars.

Each stream has its own reader and bounded handoff queue. The tap never asks
the media plane to call inference/control code; it only feeds CloudFrameCache.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from threading import Event
from time import monotonic, time_ns
from typing import Any

from contracts import MediaFrameRef
from media.profile import StreamKey, media_path
from media.frame_identity import BAND_HEIGHT, read_identity

from .frame_cache import CloudFrameCache, CloudFrameTap, DecodedMediaFrame


@dataclass(frozen=True)
class MediaDisconnected:
    stream_key: StreamKey


def _put_latest(queue: asyncio.Queue, frame: DecodedMediaFrame | MediaDisconnected) -> None:
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(frame)


class RTSPFrameTapManager:
    """Lazily attach one PyAV decoder to each active RTSP stream identity."""

    def __init__(
        self, cache: CloudFrameCache, *, rtsp_base: str = "rtsp://127.0.0.1:8554",
        fps: int = 15, queue_size: int = 8,
        frame_converter: Callable[[Any], Any] | None = None,
        idle_timeout_s: float = 30.0,
        on_stream_idle: Callable[[StreamKey], Awaitable[None]] | None = None,
    ) -> None:
        if fps <= 0 or queue_size <= 0 or idle_timeout_s <= 0:
            raise ValueError("fps, queue_size and idle_timeout_s must be positive")
        self.tap = CloudFrameTap(cache)
        self.cache = cache
        self.rtsp_base = rtsp_base.rstrip("/")
        self.fps = fps
        self.queue_size = queue_size
        self.frame_converter = frame_converter
        self.idle_timeout_s = idle_timeout_s
        self.on_stream_idle = on_stream_idle
        self._lifecycle_lock = asyncio.Lock()
        self._last_reference_at: dict[StreamKey, float] = {}
        self._tasks: dict[StreamKey, asyncio.Task[None]] = {}
        self._stoppers: dict[StreamKey, Event] = {}

    async def publish_reference(self, reference: MediaFrameRef) -> None:
        async with self._lifecycle_lock:
            await self.tap.accept_reference(reference)
            key = reference.frame.stream_key
            self._last_reference_at[key] = monotonic()
            self.ensure_stream(key)

    def active_streams(self) -> dict[StreamKey, float]:
        """Return epochs with recent references and a live RTSP tap."""
        now = monotonic()
        return {
            key: last_reference
            for key, last_reference in self._last_reference_at.items()
            if now - last_reference < self.idle_timeout_s
            and key in self._tasks
            and not self._tasks[key].done()
            and key in self._stoppers
            and not self._stoppers[key].is_set()
        }

    def ensure_stream(self, stream_key: StreamKey) -> None:
        task = self._tasks.get(stream_key)
        if task is not None and not task.done() and not self._stoppers[stream_key].is_set():
            return
        url = f"{self.rtsp_base}/{media_path(stream_key)}"
        stopper = Event()
        queue: asyncio.Queue[DecodedMediaFrame] = asyncio.Queue(self.queue_size)
        loop = asyncio.get_running_loop()
        worker = asyncio.create_task(asyncio.to_thread(
            self._decode_forever, loop, queue, stopper, url, stream_key,
        ))
        consumer = asyncio.create_task(self._consume(queue, stopper))
        watchdog = asyncio.create_task(self._idle_watch(stream_key, stopper))
        self._stoppers[stream_key] = stopper
        self._tasks[stream_key] = asyncio.create_task(
            self._join(worker, consumer, watchdog, stream_key, stopper),
        )

    async def _idle_watch(self, key: StreamKey, stopper: Event) -> None:
        interval_s = min(1.0, self.idle_timeout_s / 4)
        while not stopper.is_set():
            await asyncio.sleep(interval_s)
            async with self._lifecycle_lock:
                if stopper.is_set():
                    return
                if monotonic() - self._last_reference_at[key] < self.idle_timeout_s:
                    continue
                stopper.set()
                await self.cache.drop_stream(key)
                if self.on_stream_idle is not None:
                    await self.on_stream_idle(key)
                return

    def _decode_forever(
        self, loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[DecodedMediaFrame], stopper: Event,
        url: str, stream_key: StreamKey,
    ) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for the Cloud RTSP frame tap (install av)") from exc
        last_error: str | None = None
        while not stopper.is_set():
            container = None
            connected = False
            try:
                container = av.open(url, mode="r", options={"rtsp_transport": "tcp"}, timeout=(3, 3))
                video = next((stream for stream in container.streams if stream.type == "video"), None)
                if video is None:
                    raise RuntimeError(f"RTSP source has no video track: {url}")
                connected = True
                first_frame = True
                for decoded in container.decode(video=video.index):
                    if stopper.is_set():
                        break
                    pixels = (
                        self.frame_converter(decoded)
                        if self.frame_converter is not None
                        else decoded.to_ndarray(format="bgr24")
                    )
                    try:
                        identity = read_identity(pixels, key=stream_key)
                    except ValueError as exc:
                        if str(exc) != last_error:
                            print(f"[Cloud RTSP tap] {stream_key}: {exc}; frame rejected", flush=True)
                        last_error = str(exc)
                        continue
                    if identity.padding:
                        continue
                    seq, pts_ns = identity.seq, identity.pts_ns
                    pixels = pixels[:-BAND_HEIGHT].copy()
                    if first_frame:
                        print(f"[Cloud RTSP tap] {stream_key}: verified frame seq={seq} "
                              f"pts_ns={pts_ns} fps={identity.fps}", flush=True)
                        first_frame = False
                    item = DecodedMediaFrame(
                        stream_key=stream_key, encoded_seq=seq,
                        media_pts_ns=pts_ns, pixels=pixels,
                        received_utc_ns=time_ns(),
                        media_fps=identity.fps,
                    )
                    loop.call_soon_threadsafe(_put_latest, queue, item)
                    last_error = None
                if not stopper.is_set():
                    stopper.wait(0.2)
            except Exception as exc:
                # Reconnect independently; a temporary RTSP interruption must
                # not tear down gRPC or other node/stream taps.
                message = f"{type(exc).__name__}: {exc}"
                if message != last_error:
                    print(f"[Cloud RTSP tap] {stream_key}: {message}; retrying", flush=True)
                last_error = message
                stopper.wait(0.5)
            finally:
                if connected and not stopper.is_set():
                    # Same FIFO as decoded frames: never wake a ticket as missing
                    # before the consumer has ingested the queued tail frames.
                    pending = asyncio.run_coroutine_threadsafe(
                        queue.put(MediaDisconnected(stream_key)), loop)
                    while not stopper.is_set():
                        try:
                            pending.result(timeout=0.1)
                            break
                        except FutureTimeout:
                            continue
                    if not pending.done():
                        pending.cancel()
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass

    async def _consume(self, queue: asyncio.Queue[DecodedMediaFrame], stopper: Event) -> None:
        while not stopper.is_set() or not queue.empty():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            try:
                if isinstance(frame, MediaDisconnected):
                    await self.cache.media_disconnected(frame.stream_key)
                    continue
                await self.tap.accept_decoded(
                    stream_key=frame.stream_key, encoded_seq=frame.encoded_seq,
                    media_pts_ns=frame.media_pts_ns, pixels=frame.pixels,
                    received_utc_ns=frame.received_utc_ns,
                    media_fps=frame.media_fps,
                )
                await self.cache.media_progress(frame.stream_key, frame.encoded_seq)
            finally:
                queue.task_done()

    async def _join(
        self, worker: asyncio.Task[Any], consumer: asyncio.Task[Any],
        watchdog: asyncio.Task[Any], key: StreamKey, stopper: Event,
    ) -> None:
        try:
            await asyncio.gather(worker, consumer, watchdog)
        finally:
            watchdog.cancel()
            if self._stoppers.get(key) is stopper:
                self._tasks.pop(key, None)
                self._stoppers.pop(key, None)
                self._last_reference_at.pop(key, None)

    async def close(self) -> None:
        for stopper in tuple(self._stoppers.values()):
            stopper.set()
        tasks = tuple(self._tasks.values())
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=6)
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()
