"""One decoded source frame shared by media and analytics."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from contracts import FrameIdentity


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    identity: FrameIdentity
    pixels: Any
    capture_mono_ns: int
    # Selection clock only. File replay uses its stable source grid; latency
    # and deadlines always use the real capture_mono_ns above.
    cadence_ns: int | None = None
    # (width, height) of the camera when pixels are a stretched hardware-resized copy (media/hw_capture size); else None.
    source_size: tuple[int, int] | None = None
    # Slow lane (8/11): the full-resolution frame on every few frames (plate crops), else None. On the CPU path it is
    # the same array as pixels; on the hardware path a separate low-rate branch (media/hw_capture hires_every).
    hires: Any = None

    def __post_init__(self) -> None:
        if self.capture_mono_ns < 0:
            raise ValueError("capture_mono_ns must be non-negative")
        if self.cadence_ns is not None and self.cadence_ns < 0:
            raise ValueError("cadence_ns must be non-negative")

    @property
    def cadence_time_ns(self) -> int:
        return self.capture_mono_ns if self.cadence_ns is None else self.cadence_ns


class FrameSource(Protocol):
    """A capture/decoder implementation yields each source frame once."""

    def __aiter__(self) -> AsyncIterator[CapturedFrame]: ...


class SyntheticFrameSource:
    """Deterministic source for local Phase 2 tests and mock runs."""

    def __init__(
        self, *, node_id: str, camera_id: str, stream_id: str,
        stream_epoch: int, frame_count: int, start_utc_ns: int,
        start_mono_ns: int, frame_interval_ns: int,
        payload_factory: Callable[[int], Any] | None = None,
    ) -> None:
        if frame_count < 0 or frame_interval_ns <= 0 or start_mono_ns < 0:
            raise ValueError("invalid synthetic frame timing or count")
        self.node_id = node_id
        self.camera_id = camera_id
        self.stream_id = stream_id
        self.stream_epoch = stream_epoch
        self.frame_count = frame_count
        self.start_utc_ns = start_utc_ns
        self.start_mono_ns = start_mono_ns
        self.frame_interval_ns = frame_interval_ns
        self.payload_factory = payload_factory or (lambda frame_id: frame_id)

    async def __aiter__(self) -> AsyncIterator[CapturedFrame]:
        for frame_id in range(self.frame_count):
            offset = frame_id * self.frame_interval_ns
            yield CapturedFrame(
                identity=FrameIdentity(
                    node_id=self.node_id, camera_id=self.camera_id,
                    stream_id=self.stream_id, stream_epoch=self.stream_epoch,
                    source_frame_id=frame_id,
                    capture_utc_ns=self.start_utc_ns + offset,
                    source_pts=offset,
                ),
                pixels=self.payload_factory(frame_id),
                capture_mono_ns=self.start_mono_ns + offset,
            )
