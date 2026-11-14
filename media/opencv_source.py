"""Decode a file or camera once, then yield shared Edge frames."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import math
import queue
import threading
from pathlib import Path
from time import monotonic_ns, time_ns

from contracts import FrameIdentity
from edge_agent import CapturedFrame

SEEK_LATE_FRAMES = 30   # 2 s at 15 fps: later than this, seek instead of decoding each late frame


class OpenCVFrameSource:
    def __init__(
        self, source: str | int, *, node_id: str, camera_id: str,
        stream_id: str, stream_epoch: int = 0, max_frames: int | None = None,
        realtime_file: bool = True, loop: bool = False, prefetch: int = 0, start_frame: int = 0,
        decoder: str | None = None, hw_size: int = 0, hires_every: int = 0,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        # Staged runs: begin this many frames into the file (frame IDs still start at 0).
        self.start_frame = start_frame
        if prefetch < 0 or (prefetch and realtime_file):
            raise ValueError("prefetch must be >= 0 and is only for unpaced file reading")
        # Decode up to `prefetch` frames ahead in a reader thread, so decoding frame
        # n+1 overlaps the consumer's work on frame n. Pixels and order are unchanged.
        self.prefetch = prefetch
        self.source = source
        self.node_id = node_id
        self.camera_id = camera_id
        self.stream_id = stream_id
        self.stream_epoch = stream_epoch
        self.max_frames = max_frames
        self.realtime_file = realtime_file
        # Benchmark only: rewind a file at EOF so long runs replay the same clip.
        # Frame IDs and PTS keep increasing; the content repeats.
        self.loop = loop
        self.loops_completed = 0
        self.frames_skipped = 0   # realtime file: frames dropped to stay on the clock, like a live camera
        # 12/11: frame 0's time = when the main stream started (set by the agent from its CopyPublisher), not when this
        # source finished opening its decoder: on 252 that took seconds and the video stayed ~3.5 s ahead of the boxes.
        self.clock_origin_mono_ns = None
        # v4 (7/11): a GStreamer hardware decoder (media/hw_capture.py), pixel-identical to OpenCV; None = OpenCV on the CPU.
        self.decoder = decoder
        # 8/11: with a decoder, also resize on the GPU to hw_size x hw_size (stretched); 0 = full frames.
        if hw_size and not decoder:
            raise ValueError("hw_size needs a hardware decoder")
        self.hw_size = hw_size
        # 8/11 slow lane: CapturedFrame.hires on every hires_every-th frame (0 = never)
        self.hires_every = hires_every

    async def __aiter__(self) -> AsyncIterator[CapturedFrame]:
        import cv2

        if self.decoder:
            from media.hw_capture import HwCapture
            cap = await asyncio.to_thread(HwCapture, self.source, decoder=self.decoder, size=self.hw_size,
                                          hires_every=self.hires_every)
        else:
            cap = await asyncio.to_thread(cv2.VideoCapture, self.source)
        if not cap.isOpened():
            await asyncio.to_thread(cap.release)
            raise RuntimeError(f"cannot open video source: {self.source}")
        reader = None
        source_size = (cap.width, cap.height) if self.hw_size else None
        try:
            if self.start_frame:
                # Seek, not sequential grab(): decoding thousands of frames flat out overheated Edge252.
                # Frame-exact on the DETRAC playlist (pixel hashes checked at 6 offsets up to 30000).
                await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                landed = await asyncio.to_thread(cap.get, cv2.CAP_PROP_POS_FRAMES)
                if landed != self.start_frame:
                    raise RuntimeError(f"seek to start_frame {self.start_frame} landed on {landed} in {self.source}")
            source_fps = await asyncio.to_thread(cap.get, cv2.CAP_PROP_FPS)
            source_fps = source_fps if math.isfinite(source_fps) and 0 < source_fps <= 240 else 15.0
            is_file = isinstance(self.source, str) and Path(self.source).is_file()
            started_mono_ns = self.clock_origin_mono_ns or monotonic_ns()
            last_pts = -1
            frame_id = 0

            def read_next(index):
                ok, pixels = cap.read()
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC) if ok else 0
                if not ok and self.loop and is_file and index > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.loops_completed += 1
                    ok, pixels = cap.read()
                    pts_ms = -1  # rewound file PTS restarts; use the monotonic fallback
                if not ok or not self.hires_every:
                    return ok, pixels, pts_ms, None
                if self.hw_size:  # the hardware reader's own full-resolution branch
                    return ok, pixels, pts_ms, cap.last_hires
                return ok, pixels, pts_ms, (pixels if index % self.hires_every == 0 else None)  # same array, no copy

            reader = None
            if self.prefetch:
                ahead = queue.Queue(maxsize=self.prefetch)
                stop = threading.Event()

                def run_reader():
                    index = 0
                    while not stop.is_set():
                        if self.max_frames is not None and index >= self.max_frames:
                            item = (False, None, 0, 0, 0, None)
                        else:
                            ok, pixels, pts_ms, hires = read_next(index)
                            item = (ok, pixels, pts_ms, monotonic_ns(), time_ns(), hires)
                        while not stop.is_set():
                            try:
                                ahead.put(item, timeout=.1)
                                break
                            except queue.Full:
                                continue
                        if not item[0]:
                            return
                        index += 1

                reader = threading.Thread(target=run_reader, name="frame-prefetch", daemon=True)
                reader.start()
            period_ns = 1_000_000_000 / source_fps
            while self.max_frames is None or frame_id < self.max_frames:
                if is_file and self.realtime_file:
                    due_ns = started_mono_ns + round(frame_id * period_ns)
                    delay_ns = due_ns - monotonic_ns()
                    if delay_ns > 0:
                        await asyncio.sleep(delay_ns / 1_000_000_000)
                    elif reader is None and -delay_ns > 2 * period_ns:
                        # 12/11 on 252 (overloaded): the file was read frame by frame at 9-14 fps while the main stream
                        # played at 15, so analytics fell 3-21 s behind the video and no box matched its frame. A live
                        # camera gives the newest frame: drop the late ones (their numbers still count).
                        late = int(-delay_ns // period_ns) - 1
                        # 14/11: more than 2 s late -> jump (seek) instead of decoding every late frame: on 252 the
                        # decoders burnt 4.3 cores catching up. The file position is the frame in the file (loops wrap).
                        pos = await asyncio.to_thread(cap.get, cv2.CAP_PROP_POS_FRAMES)
                        total = await asyncio.to_thread(cap.get, cv2.CAP_PROP_FRAME_COUNT)
                        if late >= SEEK_LATE_FRAMES and 0 <= pos + late < (total or 0) and (
                                self.max_frames is None or frame_id + late < self.max_frames):
                            if await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, pos + late) and \
                                    await asyncio.to_thread(cap.get, cv2.CAP_PROP_POS_FRAMES) == pos + late:
                                frame_id += late
                                self.frames_skipped += late
                                late = 0
                        for _ in range(late):
                            if self.max_frames is not None and frame_id + 1 >= self.max_frames:
                                break
                            ok, *_ = await asyncio.to_thread(read_next, frame_id)
                            if not ok:
                                break
                            frame_id += 1
                            self.frames_skipped += 1
                if reader is None:
                    ok, pixels, file_pts_ms, hires = await asyncio.to_thread(read_next, frame_id)
                    captured_mono_ns, captured_utc_ns = monotonic_ns(), time_ns()
                else:
                    ok, pixels, file_pts_ms, captured_mono_ns, captured_utc_ns, hires = await asyncio.to_thread(ahead.get)
                if not ok:
                    break
                pts_ns = round(file_pts_ms * 1_000_000) if math.isfinite(file_pts_ms) and file_pts_ms >= 0 else -1
                if pts_ns <= last_pts:
                    fallback = (round(frame_id * 1_000_000_000 / source_fps)
                                if is_file else captured_mono_ns - started_mono_ns)
                    pts_ns = max(fallback, last_pts + 1)
                last_pts = pts_ns
                yield CapturedFrame(
                    identity=FrameIdentity(
                        node_id=self.node_id, camera_id=self.camera_id,
                        stream_id=self.stream_id, stream_epoch=self.stream_epoch,
                        source_frame_id=frame_id, capture_utc_ns=captured_utc_ns,
                        source_pts=pts_ns,
                    ),
                    pixels=pixels,
                    capture_mono_ns=captured_mono_ns,
                    cadence_ns=round(frame_id * 1_000_000_000 / source_fps) if is_file else None,
                    source_size=source_size,
                    hires=hires,
                )
                frame_id += 1
        finally:
            if reader is not None:
                stop.set()
                await asyncio.to_thread(reader.join)
            await asyncio.to_thread(cap.release)
