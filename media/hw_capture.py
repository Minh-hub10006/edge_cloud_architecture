"""Hardware decode for the analytics stream (decision 7/11): a cv2.VideoCapture stand-in on Qualcomm Edges.

GStreamer (system environment, conda off) demuxes and decodes on the SoC's video block (246: v4l2h264dec, 252: qtic2vdec)
and writes NV12 to a pipe; each frame drops its row padding and is converted by FFmpeg swscale nv12->bgr24 POINT, the
conversion OpenCV's own reader uses, so frames are pixel-identical to cv2.VideoCapture (246, vn9: max diff 0) at about
half the CPU. Only what media/opencv_source.py uses is implemented: read, get(FPS/POS_MSEC/POS_FRAMES), set(POS_FRAMES),
isOpened, release.

size (8/11, accepted): qtivtransform engine=gles resizes on the GPU straight to size x size BGR (stretched, like the
models' INTER_AREA resize): ~0.23 instead of ~2 cores per stream; VN F1 0.5548 -> 0.5434 for RT-DETR. Frames are then
not the camera's shape: CapturedFrame.source_size carries it (edge_agent/control_v1.unstretch for the Edge YOLO).
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import threading
import time
from collections import deque

import re

import numpy as np

# 12/11 on 252 (3 Edges x 4 cams, qtic2vdec): a camera's decoder delivered no frame for ~55 s at start and then from
# ~110 s to the end, with no error; read() waited forever and the agent was killed at its run timeout. A read now waits
# at most STALL_S, logs what the pipeline said, and restarts the decoder at the frame it was on.
STALL_S = float(os.getenv("V4_CAPTURE_STALL_S", "10"))
SYSTEM_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "GST_DEBUG": "1"}
CAP_PROP_POS_MSEC, CAP_PROP_POS_FRAMES, CAP_PROP_FPS = 0, 1, 5  # cv2 property ids


def gst_source(source: str, decoder: str, io_mode: str = "") -> list[str]:
    """io_mode (V4L2 decoders): capture-io-mode = auto | mmap | userptr | dmabuf ... (8/11: which memory the decoded
    frames land in; reading the default uncached buffers cost 0.62 cores per 1080x1920 stream)."""
    head = (["rtspsrc", f"location={source}", "protocols=tcp", "latency=200", "!", "rtph264depay"]
            if str(source).startswith("rtsp://") else ["filesrc", f"location={source}", "!", "qtdemux"])
    dec = [decoder] + ([f"capture-io-mode={io_mode}"] if io_mode and decoder.startswith("v4l2") else [])
    return head + ["!", "h264parse", "!", *dec, "!", "video/x-raw,format=NV12"]


def _interrupt_when_parent_dies() -> None:  # runs in the child before exec (Linux)
    """If the Edge Agent dies, the kernel sends this GStreamer SIGINT: it ends with EOS and closes its decoder session.
    7/11: sessions left by killed runs are the likely cause of 246's decoder failing to open with ENOMEM."""
    import ctypes
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGINT)  # PR_SET_PDEATHSIG


def stop_gracefully(proc, timeout_s: float = 5.0, force: bool = True) -> bool:
    """SIGINT (EOS, the decoder flushes and closes), keep reading the pipe so it is never blocked, then wait; force only
    if it does not finish and force is set. -> True when it ended by itself (its decoder session was closed).
    12/11 on 246: msm_vidc "max limit 16 already running 16 sessions" with no decoder running - killed decoders leave
    ghost sessions until a reboot; restarts therefore never force (force=False) and give up instead."""
    if proc is None or proc.poll() is not None:
        return True
    proc.send_signal(signal.SIGINT)
    end = time.monotonic() + timeout_s
    if proc.stdout is not None:
        os.set_blocking(proc.stdout.fileno(), False)
        while proc.poll() is None and time.monotonic() < end:
            try:
                if not proc.stdout.read(1 << 20):
                    time.sleep(.02)
            except (BlockingIOError, TypeError):
                time.sleep(.02)
    try:
        proc.wait(timeout=max(.1, end - time.monotonic()))
        return True
    except subprocess.TimeoutExpired:
        if not force:
            return False
        print("[HwCapture] decoder did not stop on SIGINT: terminating it (may leave a decoder session until reboot)",
              flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return False


def find_stride(log: str, width: int, height: int) -> int:
    """Row stride from the per-buffer sizes GStreamer's identity prints ("... (3317760 bytes, ..."): one NV12 frame each."""
    import re
    sizes = {int(n) for n in re.findall(r"\((\d+) bytes", log)}
    if len(sizes) != 1:
        raise ValueError(f"decoder buffer sizes {sorted(sizes)}: expected one frame size")
    size = sizes.pop()
    stride, rest = divmod(size, height * 3 // 2)
    if rest or not width <= stride <= width + 512:
        raise ValueError(f"a {size}-byte buffer is not an NV12 frame of {width}x{height} with row padding")
    return stride


def transform_chain(size: int, engine_property: bool | None = None, height: int | None = None) -> list[str]:
    """engine=gles where qtivtransform has that property (246); 252's older plugin has none and uses its own default
    (8/11: "no property engine in element videotransform0"). size x size, or size x height (full-resolution branch)."""
    if engine_property is None:
        info = subprocess.run(["gst-inspect-1.0", "qtivtransform"], env=SYSTEM_ENV, capture_output=True, text=True)
        engine_property = bool(re.search(r"^\s+engine\s+:", info.stdout, re.M))
    engine = ["engine=gles"] if engine_property else []
    return ["!", "qtivtransform", *engine, "!", f"video/x-raw,format=BGR,width={size},height={height or size}"]


def bgr_from_buffer(buf, width: int, height: int):
    """One width x height BGR frame out of a larger GPU buffer (246: 1638400 bytes for 640x640). engine gles packs the
    pixels at the start (tail zero); other engines pad every row. shortcut: an all-black bottom part reads as packed."""
    raw = np.frombuffer(buf, np.uint8)
    n = width * height * 3
    if not raw[n:].any():
        return raw[:n].reshape(height, width, 3).copy()
    row = len(raw) // height
    return raw.reshape(height, row)[:, :width * 3].reshape(height, width, 3).copy()


def small_bgr(buf, size: int):
    return bgr_from_buffer(buf, size, size)


def buffer_size(log: str, minimum: int, what: str) -> int:
    sizes = {int(n) for n in re.findall(r"\((\d+) bytes", log)}
    if len(sizes) != 1 or min(sizes) < minimum:
        raise RuntimeError(f"{what} buffer sizes {sorted(sizes)}: expected one, >= {minimum}")
    return sizes.pop()


def nv12_to_bgr(buf, width: int, height: int, stride: int):
    """Decoder rows (padded to `stride`) are written straight into the frame's own planes: one copy instead of two
    (8/11 laptop: 22.9 -> 15.6 ms per 1080x1920 frame, same pixels)."""
    import av
    rows = np.frombuffer(buf, dtype=np.uint8).reshape(height * 3 // 2, stride)
    frame = av.VideoFrame(width, height, "nv12")
    for plane, part in zip(frame.planes, (rows[:height], rows[height:])):
        np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)[:part.shape[0], :width] = part[:, :width]
    return frame.reformat(format="bgr24", interpolation="POINT").to_ndarray()


class HwCapture:
    def __init__(self, source: str, *, decoder: str = "v4l2h264dec", io_mode: str | None = None, size: int = 0,
                 hires_every: int = 0):
        """hires_every (with size, 8/11 slow lane): a second branch of the same decoder gives the full-resolution frame
        of every hires_every-th frame (GPU colour convert, videorate drops the rest before any copy) -> last_hires."""
        import cv2
        probe = cv2.VideoCapture(source)  # only the stream's size and rate
        self.width, self.height = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = probe.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)   # tells a decoder failure from the file's end
        probe.release()
        if not self.width or not self.height:
            raise RuntimeError(f"cannot read the size of {source}")
        self.source, self.decoder = source, decoder
        self.io_mode = os.getenv("V4_DECODER_IO", "") if io_mode is None else io_mode
        self.size = size
        self.hires_every = hires_every if size else 0
        self.last_hires = None
        if self.hires_every:
            rate = self.fps / self.hires_every
            if self.hires_every < 1 or abs(rate - round(rate)) > 1e-6:
                raise ValueError(f"hires_every {hires_every} must divide the source rate {self.fps}")
            self.hires_rate = round(rate)
        self.head = gst_source(source, decoder, self.io_mode)
        self.chain = self.head + (transform_chain(size) if size else [])
        log = self._probe(self.chain)
        if size:
            self.stride, self.frame_bytes = 0, buffer_size(log, size * size * 3, "qtivtransform")
            if self.hires_every:
                self.hires_chain = transform_chain(self.width, height=self.height)
                self.hires_bytes = buffer_size(self._probe(self.head + self.hires_chain), self.width * self.height * 3,
                                               "full-resolution qtivtransform")
        else:
            self.stride = find_stride(log, self.width, self.height)
            self.frame_bytes = self.stride * self.height * 3 // 2
        self.index, self._proc, self._err = 0, None, deque(maxlen=20)
        self.stalls = self.failures = self.restarts = 0
        self.dead = False   # gave up: no new decoder session will be opened (read and set return False)
        self._start()

    def _probe(self, chain) -> str:
        """One retry after a hang or a failure (12/11 on 246/252, agents opening decoders at once: a probe hung 60 s, or
        failed with "Device or resource busy" / "poll error", and its camera never started). Out of memory (ENOMEM, -12)
        is not retried: only a reboot of the Edge frees it."""
        try:
            return self._probe_once(chain)
        except (TimeoutError, RuntimeError) as exc:
            log = str(exc).split(". Check the device")[0]   # the pipeline's own words, not the hint after them
            if not isinstance(exc, TimeoutError) and re.search(r"ENOMEM|-12\b|Cannot allocate memory", log):
                raise
            print(f"[HwCapture] {exc}; retrying once in 3 s", flush=True)
            time.sleep(3)
            try:
                return self._probe_once(chain)
            except TimeoutError as again:
                raise RuntimeError(f"{again} (after one retry)") from again
            except RuntimeError as again:
                raise RuntimeError(f"{again} (after one retry)") from again

    def _probe_once(self, chain) -> str:
        # identity prints each buffer's size. Stop after 30 frames like scripts/hw_decode_check.py (run 4 times on 246): the
        # V4L2 decoder holds several frames, and an EOS after only 5 reached it before any output ("device busy", no preroll).
        p = subprocess.Popen(["gst-launch-1.0", "-e", "-v", *chain, "!", "identity", "silent=false", "eos-after=30", "!",
                              "fakesink", "sync=false"], env=SYSTEM_ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, preexec_fn=_interrupt_when_parent_dies)
        try:
            out, err = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            # 10/11: subprocess.run's timeout SIGKILLs, and a killed decoder session stays a ghost in msm_vidc (no new
            # session opens until a reboot). SIGINT first so GStreamer closes it, kill only if that does not end it.
            p.send_signal(signal.SIGINT)
            try:
                out, err = p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
            raise TimeoutError(f"hardware decoder probe hung for 60 s (stopped with SIGINT): "
                               f"{' | '.join((out + err).strip().splitlines()[-3:])}")
        if p.returncode:
            log = out + err
            tail = " | ".join(log.strip().splitlines()[-4:])
            hint = ""
            if "link has no sink" in log or "no element" in log:
                hint = f" -> GStreamer has no '{self.decoder}' element: the decoder device could not be opened"
            raise RuntimeError(f"hardware decoder probe failed (exit {p.returncode}){hint}: {tail}. Check the device "
                               f"(an ENOMEM on /dev/video32 needs a reboot of the Edge)")
        return out + err

    def _start(self) -> None:
        self._buf = bytearray(self.frame_bytes)  # reused for every frame: nv12_to_bgr copies it into the frame
        # Frames go through their own pipe, never stdout: 13/11 on 252 qtic2vdec's library printed 64 KiB of log lines
        # ("QC2TargetSpec: Video Core Features...") to stdout ahead of the frames; every picture the model got was moved
        # by 65536 bytes (colours rotated, last 34 rows lost) and every box on 252 was wrong. Library logs -> /dev/null.
        frames_read, frames_write = os.pipe()
        cmd, pass_fds, hires_read = (["gst-launch-1.0", "-e", "-q", *self.chain, "!", "fdsink", f"fd={frames_write}"],
                                     (frames_write,), None)
        if self.hires_every:
            hires_read, hires_write = os.pipe()
            cmd = ["gst-launch-1.0", "-e", "-q", *self.head, "!", "tee", "name=t",
                   "t.", "!", "queue", *self.chain[len(self.head):], "!", "fdsink", f"fd={frames_write}",
                   "t.", "!", "queue", "!", "videorate", "drop-only=true", f"max-rate={self.hires_rate}",
                   *self.hires_chain, "!", "fdsink", f"fd={hires_write}"]
            pass_fds = (frames_write, hires_write)
        self._proc = subprocess.Popen(cmd, env=SYSTEM_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      pass_fds=pass_fds, preexec_fn=_interrupt_when_parent_dies)
        os.close(frames_write)   # the child holds it; EOF when the pipeline ends
        self._proc.stdout = os.fdopen(frames_read, "rb", buffering=0)   # read / skip / stop_gracefully read it as before
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()
        self.index = 0
        if hires_read is not None:
            os.close(hires_write)  # the child holds it; EOF when the pipeline ends
            self._hires, self._hires_cv = {}, threading.Condition()
            threading.Thread(target=self._read_hires, args=(os.fdopen(hires_read, "rb", buffering=0), self._hires,
                                                            self._hires_cv), daemon=True).start()

    def _read_hires(self, pipe, frames: dict, cv) -> None:
        """Full-resolution frame k of the branch = analytics frame k * hires_every (videorate keeps 0, n, 2n, ...).
        frames / cv belong to this pipeline only: 12/11 the reader of a stopped pipeline (restart, loop rewind) wrote its
        end marker into the next pipeline's dict and full-resolution frames stopped for good."""
        k, buf = 0, bytearray(self.hires_bytes)
        with pipe:
            while True:
                view, got = memoryview(buf), 0
                while got < self.hires_bytes:
                    n = pipe.readinto(view[got:])
                    if not n:
                        with cv:
                            frames[None] = None  # end marker: readers stop waiting
                            cv.notify_all()
                        return
                    got += n
                frame = bgr_from_buffer(buf, self.width, self.height)
                with cv:
                    frames[k] = frame
                    for old in [key for key in frames if key is not None and key < k - 8]:
                        del frames[old]
                    cv.notify_all()
                k += 1

    def _hires_for(self, index: int, timeout_s: float = 2.0):
        if not self.hires_every or index % self.hires_every:
            return None
        k = index // self.hires_every
        with self._hires_cv:
            self._hires_cv.wait_for(lambda: k in self._hires or None in self._hires, timeout_s)
            return self._hires.pop(k, None)

    def _drain_stderr(self, proc) -> None:
        for line in proc.stderr:
            self._err.append(line.decode("utf-8", errors="replace").rstrip())

    @property
    def error_tail(self) -> str:
        return "\n".join(self._err)

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _readable(self, timeout_s: float) -> bool:
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout_s)
        return bool(ready)

    MAX_RESTARTS = 3   # per capture (camera run): stalls and failures together

    def _restart_at_current_frame(self, why: str) -> bool:
        """Stop the decoder gracefully (never killed: a killed one leaks a session, 16 per Edge), wait for the driver,
        open a new one at the frame we were on. -> False (and dead) when over budget or the old one would not stop."""
        tail = " | ".join(list(self._err)[-3:])
        if self.restarts >= self.MAX_RESTARTS:
            print(f"[HwCapture] {why} at frame {self.index}: {tail}; already restarted {self.restarts} times, giving up "
                  f"(camera stops)", flush=True)
            self.dead = True
            return False
        self.restarts += 1
        print(f"[HwCapture] {why} at frame {self.index} (restart {self.restarts}/{self.MAX_RESTARTS}): {tail}",
              flush=True)
        proc, self._proc = self._proc, None
        if not stop_gracefully(proc, timeout_s=15, force=False):
            print("[HwCapture] the old decoder did not stop in 15 s: not opening another session, camera stops "
                  "(reboot the Edge if this repeats)", flush=True)
            self._proc, self.dead = proc, True
            return False
        time.sleep(2)   # let msm_vidc close the session before a new one opens
        if not self.set(CAP_PROP_POS_FRAMES, self.index):
            print(f"[HwCapture] restart could not reach frame {self.index}: camera stops", flush=True)
            self.dead = True
            return False
        return True

    def _restart_after_stall(self) -> bool:
        self.stalls += 1
        alive = self._proc is not None and self._proc.poll() is None
        return self._restart_at_current_frame(f"no frame for {STALL_S:.0f} s (gst {'alive' if alive else 'exited'})")

    def _restart_after_failure(self) -> bool:
        """The pipeline ended. Before the file's last frame that is a decoder failure (12/11: 246 and 252 decoders exited
        at 22 s / 91 s and the camera stopped): restart there, within the same budget as stalls.
        -> True when restarted, False at the real end of the file or when giving up."""
        if not self.frame_count or self.index >= self.frame_count - 5:   # the decoder may give a few frames fewer
            return False
        self.failures += 1
        code = self._proc.poll() if self._proc is not None else None
        return self._restart_at_current_frame(f"decoder ended (exit {code}) before the end of the file "
                                              f"({self.frame_count} frames)")

    def read(self):
        if self.dead:
            return False, None
        buf = self._buf
        view, got = memoryview(buf), 0
        while got < self.frame_bytes:
            if not self._readable(STALL_S):
                if not self._restart_after_stall():
                    return False, None
                buf = self._buf
                view, got = memoryview(buf), 0
                continue
            n = self._proc.stdout.readinto(view[got:])
            if not n:
                if not self._restart_after_failure():
                    return False, None  # end of file, or the decoder keeps failing (error_tail says why)
                buf = self._buf
                view, got = memoryview(buf), 0
                continue
            got += n
        self.last_hires = self._hires_for(self.index)
        self.index += 1
        return True, (small_bgr(buf, self.size) if self.size else nv12_to_bgr(buf, self.width, self.height, self.stride))

    def get(self, prop: int) -> float:
        if prop == CAP_PROP_FPS:
            return self.fps
        if prop == CAP_PROP_POS_FRAMES:
            return float(self.index)
        if prop == CAP_PROP_POS_MSEC:
            return (self.index - 1) * 1000 / self.fps if self.fps else 0.0  # nominal time of the frame just read
        return 0.0

    def set(self, prop: int, value) -> bool:
        if prop != CAP_PROP_POS_FRAMES or self.dead:   # gave up: the loop rewind must not open a new session either
            return False
        self.release()  # rewind = restart; a start frame is reached by decoding and dropping (cheap on the video block)
        self._start()
        target = int(value)
        while self.index < target:
            if not self._skip_frame():
                return False
            self.index += 1
        return True

    def _skip_frame(self) -> bool:
        """Drop exactly one frame. 12/11: this used one read(frame_bytes) on the unbuffered pipe, which returns at most
        ~64 KB: every restart at a frame > 0 (new that day: decoder stall recovery) stopped at once with the index back
        at 0 and the byte stream cut mid-frame - the camera went on with shifted garbage pictures and no boxes."""
        left = self.frame_bytes
        while left:
            if not self._readable(STALL_S):
                return False
            chunk = self._proc.stdout.read(min(left, 1 << 20))
            if not chunk:
                return False
            left -= len(chunk)
        return True

    def release(self) -> None:
        # never forced (14/11: 3 decoders on 252 were terminated at the end of a run -> ghost sessions until a reboot);
        # one still running after 15 s gets SIGINT again when this process exits, and ends at the file's end anyway
        proc, self._proc = self._proc, None
        if not stop_gracefully(proc, timeout_s=15, force=False):
            print("[HwCapture] decoder did not stop in 15 s after SIGINT: left to finish by itself", flush=True)
