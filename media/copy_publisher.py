"""Main stream without re-encoding (v4, 6/11), like an IP camera that already gives H.264: the Edge only forwards it.

Testbed stand-in for a camera's main stream: `ffmpeg -re -c copy` of a file from scripts/make_main_video.py (H.264 without
B-frames, frame number in the identity band). Analytics keep reading frames from --video. Same interface as
FFmpegH264Publisher: publish() encodes nothing. media_pts_ns = file frame number / fps, so a reader finds a video frame's
boxes from the number in its band. With loop_frames (the file's frame count, set when the analytics source loops too) the
file restarts at its end and the number wraps with it; without, the file must be longer than the run (11/11: a 180 s
run over 140 s files ended every agent with "FFmpeg exited").
"""
from __future__ import annotations

import asyncio
from collections import deque
import shutil
from time import time_ns

from contracts import MediaFrameRef

from .profile import MediaProfile, StreamKey


MAIN_KEY = ("main", "v4")  # identity band tag of the prepared files


def build_copy_command(main_video, output, *, start_s=0.0, ffmpeg_bin="ffmpeg", loop=False):
    return [ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-re", *(("-stream_loop", "-1") if loop else ()),
            "-ss", f"{start_s:.3f}", "-i", str(main_video), "-c", "copy", "-an", "-f", "rtsp", "-rtsp_transport", "tcp", output]


class CopyPublisher:
    def __init__(self, *, profile: MediaProfile, output: str, stream_key: StreamKey, main_video, start_s=0.0,
                 ffmpeg_bin="ffmpeg", on_frame_submitted=None, loop_frames: int = 0) -> None:
        self.profile, self.output, self.stream_key = profile, output, stream_key
        self.loop_frames = int(loop_frames)
        self.main_video, self.start_s, self.ffmpeg_bin = main_video, start_s, ffmpeg_bin
        self.on_frame_submitted = on_frame_submitted
        self.frames_submitted = self.padding_frames_submitted = 0
        self._process = self._stderr_task = None
        self._stderr_tail = deque(maxlen=20)

    @property
    def error_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    async def start(self) -> None:
        if shutil.which(self.ffmpeg_bin) is None:
            raise FileNotFoundError(self.ffmpeg_bin)
        self._process = await asyncio.create_subprocess_exec(
            *build_copy_command(self.main_video, self.output, start_s=self.start_s, ffmpeg_bin=self.ffmpeg_bin,
                                loop=self.loop_frames > 0),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        self._stderr_task = asyncio.create_task(self._read_stderr(self._process.stderr))

    async def _read_stderr(self, stderr) -> None:
        while line := await stderr.readline():
            self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    async def publish(self, frame) -> MediaFrameRef:
        if frame.identity.stream_key != self.stream_key:
            raise ValueError("frame belongs to another stream or epoch")
        if self._process is None:
            raise RuntimeError("publisher is not started")
        if self._process.returncode is not None:
            raise RuntimeError(f"FFmpeg exited with {self._process.returncode}: {self.error_tail}")
        seq = self.frames_submitted
        self.frames_submitted += 1
        # the frame's own number in the file, not how many were published: 12/11 a source that drops late frames (an
        # overloaded Edge) would otherwise draw frame k's boxes on a later frame of the video
        file_frame = round(self.start_s * self.profile.fps) + frame.identity.source_frame_id
        if self.loop_frames > 0:
            file_frame %= self.loop_frames
        reference = MediaFrameRef(frame=frame.identity, encoded_seq=seq,
                                  media_pts_ns=round(file_frame * 1_000_000_000 / self.profile.fps), submitted_utc_ns=time_ns())
        if self.on_frame_submitted is not None:
            await self.on_frame_submitted(reference)
        return reference

    async def publish_padding(self, frame) -> None:
        self.padding_frames_submitted += 1  # the file keeps playing by itself

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()
