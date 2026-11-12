"""FFmpeg H.264 publisher fed by decoded frames from the shared Edge capture."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
import shutil
from time import monotonic_ns, perf_counter_ns, time_ns  # perf_counter for durations: Windows monotonic ticks ~16 ms
from typing import Literal

from contracts import MediaFrameRef
from edge_agent import CapturedFrame

from .profile import MediaProfile, StreamKey
from .frame_identity import BAND_HEIGHT, frame_pts_ns, write_identity_band

OutputFormat = Literal["rtsp", "h264"]
MediaFrameRefSink = Callable[[MediaFrameRef], Awaitable[None]]


def build_ffmpeg_command(
    profile: MediaProfile, output: str, *,
    output_format: OutputFormat = "rtsp", ffmpeg_bin: str = "ffmpeg", frame_identity: bool = False,
) -> list[str]:
    if not output:
        raise ValueError("output is required")
    if output_format not in ("rtsp", "h264"):
        raise ValueError("output_format must be rtsp or h264")
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{profile.width}x{profile.height + (BAND_HEIGHT if frame_identity else 0)}",
        "-framerate", str(profile.fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-profile:v", "baseline", "-b:v", f"{profile.bitrate_kbps}k",
        "-maxrate", f"{profile.bitrate_kbps}k",
        "-bufsize", f"{profile.bitrate_kbps}k",
        "-g", str(profile.gop_frames),
        "-keyint_min", str(profile.gop_frames),
        "-bf", "0", "-sc_threshold", "0",
        # Pin trellis off: x264 core 164 (Edge246) already uses 0 under these flags, but core 155
        # (Ubuntu 20.04 Edges) enables trellis=1, so Edges would encode with different settings.
        "-trellis", "0",
    ]
    if output_format == "rtsp":
        command += ["-rtsp_transport", "tcp", "-f", "rtsp", output]
    else:
        command += ["-f", "h264", output]
    return command


def _frame_bytes(frame: CapturedFrame, profile: MediaProfile, identity=None, stages=None) -> bytes:
    """Raw bgr24 bytes for FFmpeg: resize straight into the output buffer and write the
    identity band below it, so the frame is copied once (tobytes) instead of three times."""
    import cv2
    import numpy as np

    pixels = frame.pixels
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("media frames must be HxWx3 uint8 BGR arrays")
    started = perf_counter_ns()
    height, width = profile.height, profile.width
    out = np.empty((height + (BAND_HEIGHT if identity is not None else 0), width, 3), dtype=np.uint8)
    image = out[:height]  # leading rows: contiguous view
    if pixels.shape[:2] != (height, width):
        shrinking = pixels.shape[0] > height or pixels.shape[1] > width
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        written = cv2.resize(pixels, (width, height), dst=image, interpolation=interpolation)
        if written is not image and not np.shares_memory(written, out):
            image[...] = written  # OpenCV reallocated; keep the result correct
    else:
        image[...] = pixels
    resized = perf_counter_ns()
    if identity is not None:
        write_identity_band(out[height:], **identity)
    banded = perf_counter_ns()
    raw = out.tobytes()
    if stages is not None:
        stages.update(media_resize_ms=(resized - started) / 1e6, media_band_ms=(banded - resized) / 1e6,
                      media_tobytes_ms=(perf_counter_ns() - banded) / 1e6)
    return raw


class FFmpegH264Publisher:
    def __init__(
        self, *, profile: MediaProfile, output: str,
        stream_key: StreamKey, output_format: OutputFormat = "rtsp",
        ffmpeg_bin: str = "ffmpeg",
        on_frame_submitted: MediaFrameRefSink | None = None,
        frame_identity: bool = False,
    ) -> None:
        self.profile = profile
        self.output = output
        self.stream_key = stream_key
        self.output_format = output_format
        self.ffmpeg_bin = ffmpeg_bin
        self.on_frame_submitted = on_frame_submitted
        self.frame_identity = frame_identity
        if frame_identity and (profile.width < 128 or int(profile.fps) != profile.fps or profile.fps > 240):
            raise ValueError("frame identity requires width >=128 and integer FPS <=240")
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self.frames_submitted = 0
        self.padding_frames_submitted = 0

    @property
    def error_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    async def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("publisher is already started")
        if shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError(self.ffmpeg_bin)
        command = build_ffmpeg_command(
            self.profile, self.output, output_format=self.output_format,
            ffmpeg_bin=self.ffmpeg_bin, frame_identity=self.frame_identity,
        )
        self._process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self._process.stderr is not None
        self._stderr_task = asyncio.create_task(self._read_stderr(self._process.stderr))

    async def _read_stderr(self, stderr: asyncio.StreamReader) -> None:
        while line := await stderr.readline():
            self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    async def publish(self, frame: CapturedFrame) -> MediaFrameRef:
        if frame.identity.stream_key != self.stream_key:
            raise ValueError("frame belongs to another stream or epoch")
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("publisher is not started")
        if process.returncode is not None:
            raise RuntimeError(f"FFmpeg exited with {process.returncode}: {self.error_tail}")
        encoded_seq = self.frames_submitted + self.padding_frames_submitted
        identity = dict(key=self.stream_key, seq=encoded_seq, fps=self.profile.fps) if self.frame_identity else None
        prepared_started = perf_counter_ns()
        detail: dict[str, float] = {}
        raw = await asyncio.to_thread(_frame_bytes, frame, self.profile, identity, detail)
        prepared_ms = (perf_counter_ns() - prepared_started) / 1e6
        pipe_started = perf_counter_ns()
        await self._write(process, raw)
        self.frames_submitted += 1
        self.last_publish_stages = {
            **detail,
            "media_prepare_ms": prepared_ms,
            # Wait for FFmpeg to take the previous frame, then queue this one.
            "media_pipe_write_ms": (perf_counter_ns() - pipe_started) / 1e6,
        }
        reference = MediaFrameRef(
            frame=frame.identity,
            encoded_seq=encoded_seq,
            media_pts_ns=(frame_pts_ns(encoded_seq, int(self.profile.fps)) if self.frame_identity
                          else round(encoded_seq * 1_000_000_000 / self.profile.fps)),
            submitted_utc_ns=time_ns(),
        )
        if self.on_frame_submitted is not None:
            await self.on_frame_submitted(reference)
        return reference

    async def publish_padding(self, frame: CapturedFrame) -> None:
        """Keep a finite test stream alive after its last referenced frame.

        Padding follows all real frames and has no MediaFrameRef or inference
        ticket, so it cannot be mistaken for a captured source frame.
        """
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise RuntimeError(f"FFmpeg is not publishing: {self.error_tail}")
        identity = dict(key=self.stream_key, seq=self.frames_submitted + self.padding_frames_submitted,
                        fps=self.profile.fps, padding=True) if self.frame_identity else None
        raw = await asyncio.to_thread(_frame_bytes, frame, self.profile, identity)
        await self._write(process, raw)
        self.padding_frames_submitted += 1

    async def _write(self, process, raw: bytes) -> None:
        """Drain the previous frame, then queue this one without waiting for it.

        The next frame's resize then overlaps FFmpeg reading this one (bytes and
        order unchanged; at most one frame waits in the pipe buffer). A dead
        FFmpeg surfaces on the next write or on close.
        """
        try:
            await process.stdin.drain()
            process.stdin.write(raw)
        except (BrokenPipeError, ConnectionResetError) as exc:
            await process.wait()
            raise RuntimeError(f"FFmpeg stopped accepting frames: {self.error_tail}") from exc

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None:
            try:
                await process.stdin.drain()  # flush the last queued frame
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
            self._stderr_task = None
        if process.returncode:
            raise RuntimeError(f"FFmpeg exited with {process.returncode}: {self.error_tail}")

    async def __aenter__(self) -> FFmpegH264Publisher:
        await self.start()
        return self

    async def __aexit__(self, _type, _value, _traceback) -> None:
        await self.close()
