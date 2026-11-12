"""Edge x264 quality on the real media path: cv2 INTER_AREA resize -> raw BGR pipe -> the
publisher's exact FFmpeg command (h264 file) -> decode with cv2 -> PSNR vs the same resized frames.

  encode:  python -B scripts/x264_quality.py encode --video V --out /tmp/edge.h264
  score:   python -B scripts/x264_quality.py score --video V --h264 /tmp/edge.h264
Scoring any host's .h264 on one machine removes decoder/tool differences; the reference is
cv2 decode + resize (OpenCV is pinned on every Edge, so it is bit-identical across hosts).
"""
from __future__ import annotations

import argparse
import math
import subprocess

import cv2
import numpy as np

from media.ffmpeg_publisher import build_ffmpeg_command
from media.profile import MediaProfile

PROFILE = MediaProfile(width=1280, height=720, fps=15, bitrate_kbps=2500, gop_frames=15)


def frames(video, count):
    capture = cv2.VideoCapture(video)
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                return
            yield cv2.resize(frame, (PROFILE.width, PROFILE.height), interpolation=cv2.INTER_AREA)
    finally:
        capture.release()


def encode(args):
    command = build_ffmpeg_command(PROFILE, args.out, output_format="h264")
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    n = 0
    for frame in frames(args.video, args.frames):
        proc.stdin.write(frame.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg exit {proc.returncode}")
    print(f"ENCODED frames={n} -> {args.out}")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def score(args):
    decoded = cv2.VideoCapture(args.h264)
    y, bgr = [], []
    for ref in frames(args.video, args.frames):
        ok, test = decoded.read()
        if not ok:
            break
        bgr.append(psnr(ref, test))
        y.append(psnr(cv2.cvtColor(ref, cv2.COLOR_BGR2YUV)[:, :, 0], cv2.cvtColor(test, cv2.COLOR_BGR2YUV)[:, :, 0]))
    decoded.release()
    if not y:
        raise SystemExit("no decoded frames")
    worst = int(np.argmin(y))
    print(f"SCORE frames={len(y)} psnr_y_mean={np.mean(y):.2f} psnr_y_p5={np.percentile(y, 5):.2f} "
          f"psnr_y_min={min(y):.2f}@{worst} psnr_bgr_mean={np.mean(bgr):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "score"):
        s = sub.add_parser(name)
        s.add_argument("--video", required=True)
        s.add_argument("--frames", type=int, default=600)
        if name == "encode":
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--h264", required=True)
    args = p.parse_args()
    encode(args) if args.cmd == "encode" else score(args)


if __name__ == "__main__":
    main()
