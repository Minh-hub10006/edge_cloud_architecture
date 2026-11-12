"""Video source with separately measured decode and resize stages."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from benchmark.telemetry.trace import FrameTrace


class BenchmarkVideoSource:
    def __init__(
        self,
        path: str | Path,
        *,
        resize_wh: tuple[int, int] = (640, 480),
        loop: bool = True,
        prefetch: bool = False,
    ):
        self.path = str(path)
        self.resize_wh = resize_wh
        self.loop = loop
        self._prefetch_stream = None
        self.cap = None
        if prefetch:
            from video_stream import open_video
            self._prefetch_stream = open_video(self.path, prefetch=True, queue_size=3)
            self.source_fps = float(self._prefetch_stream.source_fps or 30.0)
        else:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise ValueError(f"unable to open video: {self.path}")
            self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._last_paced_at = 0.0

    def pace(self, target_fps: int | float, *, enabled: bool) -> None:
        """Pace input before E2E timing begins."""
        if not enabled or target_fps <= 0:
            return
        interval = 1.0 / float(target_fps)
        now = time.perf_counter()
        elapsed = now - self._last_paced_at
        if self._last_paced_at > 0 and elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_paced_at = time.perf_counter()

    def read(self, trace: FrameTrace):
        if self._prefetch_stream is not None:
            with trace.span("frame_read"):
                ok, frame, _raw = self._prefetch_stream.get_frame(0, throttle=False)
            trace.attributes["frame_read_scope"] = "prefetch_dequeue"
            # Resize happens in the producer; no local resize measurement.
            trace.attributes["resize_scope"] = "background_prefetch"
            return bool(ok and frame is not None), frame

        with trace.span("frame_read"):
            ok, raw = self.cap.read()
            if not ok and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, raw = self.cap.read()
        if not ok or raw is None:
            return False, None
        with trace.span("resize"):
            frame = cv2.resize(raw, self.resize_wh)
        return True, frame

    def release(self) -> None:
        if self._prefetch_stream is not None:
            self._prefetch_stream.release()
        elif self.cap is not None:
            self.cap.release()
