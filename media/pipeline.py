"""Connect the shared capture fan-out to the independent preview publisher."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from edge_agent import BoundedLatestQueue, CapturedFrame, EdgeFanout, FrameSource, capture_to_fanout

from .ffmpeg_publisher import FFmpegH264Publisher


class MediaCadence:
    """Select preview frames without changing the analytics branch."""

    def __init__(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period_ns = round(1_000_000_000 / fps)
        # Real cameras retain their real monotonic timestamps. Allow a bounded
        # early arrival without rebasing the grid or accepting capture bursts.
        self.early_tolerance_ns = min(self.period_ns // 4, 5_000_000)
        self._next_due_ns: int | None = None
        self._last_seen_ns = -1

    def accept(self, capture_mono_ns: int) -> bool:
        if capture_mono_ns < self._last_seen_ns:
            raise ValueError("capture monotonic time moved backwards")
        self._last_seen_ns = capture_mono_ns
        if self._next_due_ns is not None and capture_mono_ns < self._next_due_ns - self.early_tolerance_ns:
            return False
        if self._next_due_ns is None:
            self._next_due_ns = capture_mono_ns + self.period_ns
        else:
            # Advance the existing clock grid: rebasing on each late capture
            # accumulates jitter and unnecessarily reduces the preview FPS.
            periods = max(1, (capture_mono_ns - self._next_due_ns) // self.period_ns + 1)
            self._next_due_ns += periods * self.period_ns
        return True


@dataclass(frozen=True, slots=True)
class MediaRunStats:
    captured: int
    media_queue_dropped: int
    analytics_queue_dropped: int
    preview_fps_skipped: int
    encoded_frames: int


async def _publish_queue(
    queue: BoundedLatestQueue[CapturedFrame],
    publisher: FFmpegH264Publisher,
    stop: asyncio.Event,
) -> int:
    cadence = MediaCadence(publisher.profile.fps)
    skipped = 0
    while not stop.is_set() or queue.pending:
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if cadence.accept(frame.cadence_time_ns):
            await publisher.publish(frame)
        else:
            skipped += 1
    return skipped


async def run_media_pipeline(
    source: FrameSource,
    fanout: EdgeFanout,
    publisher: FFmpegH264Publisher,
) -> MediaRunStats:
    """Run one capture/decode source and a concurrent bounded media consumer."""

    stop = asyncio.Event()
    async with publisher:
        capture_task = asyncio.create_task(capture_to_fanout(source, fanout))
        media_task = asyncio.create_task(_publish_queue(fanout.media, publisher, stop))
        try:
            done, _ = await asyncio.wait(
                (capture_task, media_task), return_when=asyncio.FIRST_COMPLETED,
            )
            if media_task in done:
                error = media_task.exception()
                if error is not None:
                    raise error
                if not capture_task.done():
                    raise RuntimeError("media consumer stopped before capture")
            capture_stats = capture_task.result()
            stop.set()
            skipped = await media_task
            return MediaRunStats(
                captured=capture_stats.captured,
                media_queue_dropped=capture_stats.media_dropped,
                analytics_queue_dropped=capture_stats.analytics_dropped,
                preview_fps_skipped=skipped,
                encoded_frames=publisher.frames_submitted,
            )
        finally:
            stop.set()
            for task in (capture_task, media_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(capture_task, media_task, return_exceptions=True)
